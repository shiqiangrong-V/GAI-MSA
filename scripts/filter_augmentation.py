import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from gai_msa.augmentation import filter_candidates
from gai_msa.data import load_csv
from gai_msa.utils import load_config, resolve_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--train", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument(
        "--sentiment-checkpoint", required=True
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    train_df = load_csv(args.train)
    candidates = pd.read_csv(args.candidates)

    filtered = filter_candidates(
        train_df=train_df,
        candidates_df=candidates,
        cfg=cfg["augmentation"],
        sentiment_checkpoint=args.sentiment_checkpoint,
        encoder_name=cfg["model"]["encoder_name"],
        device=resolve_device(
            cfg["training"]["device"]
        ),
        max_length=cfg["model"]["max_length"],
    )
    Path(args.output).parent.mkdir(
        parents=True, exist_ok=True
    )
    filtered.to_csv(args.output, index=False)
    if len(filtered):
        print(
            filtered["label"]
            .value_counts()
            .to_dict()
        )


if __name__ == "__main__":
    main()
