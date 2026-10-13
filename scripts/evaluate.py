import argparse
import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from gai_msa.training import evaluate_model
from gai_msa.utils import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--variant",
        default="gai_msa",
        choices=[
            "ernie",
            "ernie_cnn",
            "ernie_attention",
            "ernie_cnn_attention",
            "gai_msa",
        ],
    )
    p.add_argument("--test", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    metrics = evaluate_model(
        cfg,
        args.checkpoint,
        args.variant,
        args.test,
        args.output_dir,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
