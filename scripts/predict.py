import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from gai_msa.constants import ID2LABEL
from gai_msa.training import load_trained_model
from gai_msa.utils import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--variant", default="gai_msa"
    )
    p.add_argument("--text", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    model, device = load_trained_model(
        cfg, args.checkpoint, args.variant
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["encoder_name"],
        use_fast=True,
    )

    enc = tokenizer(
        args.text,
        max_length=cfg["model"]["max_length"],
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(
            out["logits"], dim=-1
        )[0].cpu().tolist()

    pred = max(
        range(len(probs)),
        key=probs.__getitem__,
    )
    print("prediction:", ID2LABEL[pred])
    print({
        ID2LABEL[i]: round(p, 6)
        for i, p in enumerate(probs)
    })


if __name__ == "__main__":
    main()
