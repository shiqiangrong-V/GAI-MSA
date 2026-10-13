import argparse
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from gai_msa.sentiment_filter import train_filter
from gai_msa.utils import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    best = train_filter(
        cfg, args.train, args.val, args.output_dir
    )
    print(f"Best sentiment filter saved to: {best}")


if __name__ == "__main__":
    main()
