import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .constants import LABEL2ID, LABELS
from .utils import normalize_text


def load_csv(path):
    df = pd.read_csv(path)
    required = {"text", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str).map(normalize_text)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    bad = sorted(set(df["label"]) - set(LABELS))
    if bad:
        raise ValueError(f"Unknown labels in {path}: {bad}")
    return df


class HFSentimentDataset(Dataset):
    def __init__(self, df, tokenizer_name, max_length=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_length = int(max_length)

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
        item["labels"] = torch.tensor(LABEL2ID[row["label"]], dtype=torch.long)
        item["text"] = row["text"]
        return item


def collate_hf(batch):
    keys = [k for k in batch[0].keys() if k != "text"]
    out = {k: torch.stack([x[k] for x in batch]) for k in keys}
    out["text"] = [x["text"] for x in batch]
    return out
