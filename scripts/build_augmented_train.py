import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from gai_msa.data import load_csv
from gai_msa.utils import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--train", required=True)
    p.add_argument("--filtered", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    train = load_csv(args.train)
    filtered = pd.read_csv(args.filtered)

    selected_parts = []
    for label, target in (
        cfg["augmentation"]["retained_targets"].items()
    ):
        target = int(target)
        if target == 0:
            continue

        part = filtered[
            filtered["label"] == label
        ].copy()
        if len(part) < target:
            raise RuntimeError(
                f"Insufficient valid candidates for {label}: "
                f"{len(part)} < {target}. Regenerate more "
                "candidates or verify the filtering pipeline."
            )

        part["rank_score"] = (
            part["semantic_similarity"].astype(float) *
            part["sentiment_confidence"].astype(float)
        )
        part = (
            part.sort_values(
                [
                    "rank_score",
                    "sentiment_confidence",
                    "semantic_similarity",
                ],
                ascending=False,
            )
            .head(target)
        )
        selected_parts.append(part)

    selected = pd.concat(
        selected_parts, ignore_index=True
    )
    synthetic = selected[
        ["candidate_text", "label"]
    ].rename(columns={"candidate_text": "text"})

    augmented = pd.concat(
        [train[["text", "label"]], synthetic],
        ignore_index=True,
    )
    augmented = (
        augmented.sample(
            frac=1.0,
            random_state=cfg["seed"],
        )
        .reset_index(drop=True)
    )

    Path(args.output).parent.mkdir(
        parents=True, exist_ok=True
    )
    augmented.to_csv(args.output, index=False)

    print(
        "Original:",
        train["label"].value_counts().to_dict(),
    )
    print(
        "Synthetic retained:",
        synthetic["label"].value_counts().to_dict(),
    )
    print(
        "Final:",
        augmented["label"].value_counts().to_dict(),
    )
    print("Total:", len(augmented))


if __name__ == "__main__":
    main()
