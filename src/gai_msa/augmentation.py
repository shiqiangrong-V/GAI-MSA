import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from .constants import LABEL2ID, ID2LABEL
from .utils import nonspace_char_len, normalize_text


PROMPT_TEMPLATE = (
    "You are given a Chinese social-media post and its sentiment label. "
    "Rewrite the post as a natural short Chinese social-media expression that "
    "preserves the core meaning and exactly the same sentiment category. "
    "Change the wording and sentence structure, but do not introduce new events, "
    "entities, or a different emotional polarity. Keep the rewritten text concise "
    "and return only the rewritten text. Original text: {text}. "
    "Sentiment label: {label}."
)


def generate_one(client, cfg, text, label, max_retries=5):
    prompt = PROMPT_TEMPLATE.format(text=text, label=label)
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=cfg["openai_model"],
                input=prompt,
                temperature=float(cfg["temperature"]),
                top_p=float(cfg["top_p"]),
                max_output_tokens=int(cfg["max_output_tokens"]),
            )
            return normalize_text(response.output_text.strip())
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(
        f"OpenAI generation failed after retries: {last_err}"
    )


def generate_candidates(train_df, cfg, output_csv, seed=42):
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    rng = np.random.default_rng(seed)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame()
    if output_csv.exists() and output_csv.stat().st_size > 0:
        existing = pd.read_csv(output_csv)

    existing_counts = (
        existing.groupby("label").size().to_dict()
        if len(existing) else {}
    )
    write_header = (
        not output_csv.exists() or output_csv.stat().st_size == 0
    )

    with open(output_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_index", "source_text", "label", "candidate_text"
            ],
        )
        if write_header:
            writer.writeheader()

        for label, quota in cfg["candidate_quotas"].items():
            quota = int(quota)
            remaining = quota - int(existing_counts.get(label, 0))
            if remaining <= 0:
                continue

            subset = train_df[train_df["label"] == label]
            if subset.empty:
                raise ValueError(
                    f"No training examples for label={label}"
                )

            source_indices = subset.index.to_numpy()
            cursor = 0

            while remaining > 0:
                if cursor % len(source_indices) == 0:
                    shuffled = rng.permutation(source_indices)
                idx = int(shuffled[cursor % len(shuffled)])
                cursor += 1

                source_text = str(train_df.loc[idx, "text"])
                n = min(
                    int(cfg["max_candidates_per_source"]),
                    remaining,
                )
                for _ in range(n):
                    candidate = generate_one(
                        client, cfg, source_text, label
                    )
                    writer.writerow({
                        "source_index": idx,
                        "source_text": source_text,
                        "label": label,
                        "candidate_text": candidate,
                    })
                    f.flush()
                    remaining -= 1
                    if remaining <= 0:
                        break


class FrozenSentimentFilter:
    def __init__(self, checkpoint_dir, encoder_name, device):
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder_name, use_fast=True
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                checkpoint_dir
            ).to(device)
        )
        self.model.eval()

    @torch.no_grad()
    def predict(
        self, texts, max_length=256, batch_size=64
    ):
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            all_probs.extend(probs.cpu().numpy())
        probs = np.asarray(all_probs)
        return (
            probs.argmax(axis=1),
            probs.max(axis=1),
            probs,
        )


def filter_candidates(
    train_df,
    candidates_df,
    cfg,
    sentiment_checkpoint,
    encoder_name,
    device,
    max_length=256,
):
    sem_model = SentenceTransformer(
        cfg["semantic_model"], device=str(device)
    )
    sentiment_filter = FrozenSentimentFilter(
        sentiment_checkpoint, encoder_name, device
    )

    original_norm = set(
        train_df["text"].astype(str).map(normalize_text)
    )
    seen_global = set()
    rows = []

    for _, row in candidates_df.iterrows():
        src = normalize_text(row["source_text"])
        cand = normalize_text(row["candidate_text"])
        label = str(row["label"])

        nchar = nonspace_char_len(cand)
        if not (
            int(cfg["min_chars"]) <= nchar <= int(cfg["max_chars"])
        ):
            continue
        if not cand or cand in original_norm or cand in seen_global:
            continue
        if cand == src:
            continue

        rows.append({
            "source_index": int(row["source_index"]),
            "source_text": src,
            "label": label,
            "candidate_text": cand,
        })
        seen_global.add(cand)

    if not rows:
        return pd.DataFrame()

    tmp = pd.DataFrame(rows)

    src_emb = sem_model.encode(
        tmp["source_text"].tolist(),
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    cand_emb = sem_model.encode(
        tmp["candidate_text"].tolist(),
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    sims = np.sum(src_emb * cand_emb, axis=1)
    tmp["semantic_similarity"] = sims

    tmp = tmp[
        (tmp["semantic_similarity"] >= float(cfg["similarity_min"])) &
        (tmp["semantic_similarity"] <= float(cfg["similarity_max"]))
    ].reset_index(drop=True)

    if tmp.empty:
        return tmp

    pred, conf, _ = sentiment_filter.predict(
        tmp["candidate_text"].tolist(),
        max_length=max_length,
    )
    tmp["sentiment_pred_id"] = pred
    tmp["sentiment_pred"] = [ID2LABEL[int(i)] for i in pred]
    tmp["sentiment_confidence"] = conf

    expected = tmp["label"].map(LABEL2ID).to_numpy()
    keep = (
        (tmp["sentiment_pred_id"].to_numpy() == expected) &
        (
            tmp["sentiment_confidence"].to_numpy() >=
            float(cfg["sentiment_confidence_min"])
        )
    )
    return (
        tmp.loc[keep]
        .drop(columns=["sentiment_pred_id"])
        .reset_index(drop=True)
    )
