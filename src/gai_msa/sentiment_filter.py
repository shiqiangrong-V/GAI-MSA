from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .constants import LABEL2ID, ID2LABEL
from .data import load_csv
from .utils import ensure_dir, resolve_device, set_seed


class FilterDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(
            LABEL2ID[row["label"]], dtype=torch.long
        )
        return item


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        pred = logits.argmax(dim=-1)
        y_true.extend(batch["labels"].cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    return f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )


def train_filter(config, train_path, val_path, output_dir):
    set_seed(config["seed"])
    device = resolve_device(config["training"]["device"])
    output_dir = Path(output_dir)
    best_dir = output_dir / "best"
    ensure_dir(best_dir)

    encoder_name = config["model"]["encoder_name"]
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_name, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        encoder_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    train_df = load_csv(train_path)
    val_df = load_csv(val_path)

    train_ds = FilterDataset(
        train_df, tokenizer, config["model"]["max_length"]
    )
    val_ds = FilterDataset(
        val_df, tokenizer, config["model"]["max_length"]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    best = -1.0
    history = []
    for epoch in range(
        1, int(config["training"]["epochs"]) + 1
    ):
        model.train()
        losses = []
        for batch in tqdm(
            train_loader, desc=f"Filter epoch {epoch}"
        ):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            losses.append(float(out.loss.detach().cpu()))

        val_f1 = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_macro_f1": val_f1,
        })
        pd.DataFrame(history).to_csv(
            output_dir / "history.csv", index=False
        )

        if val_f1 > best:
            best = val_f1
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

    return best_dir
