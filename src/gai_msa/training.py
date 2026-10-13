from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import ID2LABEL
from .data import HFSentimentDataset, collate_hf, load_csv
from .metrics import compute_metrics, full_reports
from .model import build_variant
from .utils import ensure_dir, resolve_device, save_json, set_seed


def _move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items() if k != "text"}


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    y_true, y_pred, probs_all, texts = [], [], [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        texts.extend(batch["text"])
        t = _move_batch(batch, device)
        outputs = model(
            input_ids=t["input_ids"],
            attention_mask=t["attention_mask"],
            token_type_ids=t.get("token_type_ids"),
            labels=t.get("labels"),
        )
        probs = torch.softmax(outputs["logits"], dim=-1)
        preds = probs.argmax(dim=-1)

        y_true.extend(t["labels"].cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        probs_all.extend(probs.cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    return metrics, y_true, y_pred, probs_all, texts


def train_model(config, variant, train_path, val_path, output_dir):
    set_seed(config["seed"])
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    device = resolve_device(config["training"]["device"])
    tokenizer_name = config["model"]["encoder_name"]
    max_length = config["model"]["max_length"]

    train_df = load_csv(train_path)
    val_df = load_csv(val_path)

    train_ds = HFSentimentDataset(train_df, tokenizer_name, max_length)
    val_ds = HFSentimentDataset(val_df, tokenizer_name, max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_hf,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_hf,
    )

    model = build_variant(config, variant).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    history = []
    best_f1 = -1.0
    best_dir = output_dir / "best"
    ensure_dir(best_dir)

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            t = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=t["input_ids"],
                attention_mask=t["attention_mask"],
                token_type_ids=t.get("token_type_ids"),
                labels=t["labels"],
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")

        val_metrics, *_ = predict_loader(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        pd.DataFrame(history).to_csv(
            output_dir / "history.csv", index=False
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_dir / "model.pt")
            save_json(
                {
                    "variant": variant,
                    "encoder_name": config["model"]["encoder_name"],
                    "best_val_metrics": val_metrics,
                },
                best_dir / "metadata.json",
            )

    return best_dir


def load_trained_model(config, checkpoint_dir, variant, device=None):
    device = (
        resolve_device(config["training"]["device"])
        if device is None else device
    )
    model = build_variant(config, variant)
    state = torch.load(
        Path(checkpoint_dir) / "model.pt", map_location=device
    )
    model.load_state_dict(state)
    model.to(device)
    return model, device


def evaluate_model(config, checkpoint_dir, variant, test_path, output_dir):
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    model, device = load_trained_model(
        config, checkpoint_dir, variant
    )

    test_df = load_csv(test_path)
    ds = HFSentimentDataset(
        test_df,
        config["model"]["encoder_name"],
        config["model"]["max_length"],
    )
    loader = DataLoader(
        ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_hf,
    )

    metrics, y_true, y_pred, probs, texts = predict_loader(
        model, loader, device
    )
    save_json(metrics, output_dir / "metrics.json")

    report_df, cm_df = full_reports(y_true, y_pred)
    report_df.to_csv(output_dir / "classification_report.csv")
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    pred_df = pd.DataFrame({
        "text": texts,
        "true_id": y_true,
        "pred_id": y_pred,
        "true_label": [ID2LABEL[i] for i in y_true],
        "pred_label": [ID2LABEL[i] for i in y_pred],
    })
    for i, name in ID2LABEL.items():
        pred_df[f"prob_{name}"] = [row[i] for row in probs]
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    return metrics
