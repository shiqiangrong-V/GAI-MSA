import argparse
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from gai_msa.augmentation import generate_candidates
from gai_msa.data import load_csv
from gai_msa.utils import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--train", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    train_df = load_csv(args.train)
    generate_candidates(
        train_df,
        cfg["augmentation"],
        args.output,
        seed=cfg["seed"],
    )


if __name__ == "__main__":
    main()
