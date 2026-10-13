import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from gai_msa.training import (
    evaluate_model,
    train_model,
)
from gai_msa.utils import ensure_dir, load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--train-original", required=True
    )
    p.add_argument(
        "--train-augmented", required=True
    )
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    ensure_dir(args.output_dir)

    variants = [
        "ernie",
        "ernie_cnn",
        "ernie_attention",
        "ernie_cnn_attention",
        "gai_msa",
    ]

    rows = []
    for variant in variants:
        train_path = (
            args.train_augmented
            if variant == "gai_msa"
            else args.train_original
        )
        ckpt_root = (
            Path(args.output_dir) /
            variant /
            "checkpoint"
        )
        best = train_model(
            cfg,
            variant,
            train_path,
            args.val,
            ckpt_root,
        )
        metrics = evaluate_model(
            cfg,
            best,
            variant,
            args.test,
            Path(args.output_dir) /
            variant /
            "test",
        )
        rows.append({
            "variant": variant,
            **metrics,
        })

    df = pd.DataFrame(rows)
    df.to_csv(
        Path(args.output_dir) /
        "ablation_summary.csv",
        index=False,
    )
    print(df)


if __name__ == "__main__":
    main()
