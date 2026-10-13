import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True)
    p.add_argument(
        "--output",
        default="results/paper_metrics.csv",
    )
    args = p.parse_args()

    root = Path(args.results_root)
    rows = []
    for pth in root.rglob("metrics.json"):
        try:
            with open(
                pth, "r", encoding="utf-8"
            ) as f:
                m = json.load(f)
            rows.append({
                "run": str(
                    pth.parent.relative_to(root)
                ),
                **m,
            })
        except Exception:
            pass

    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(
        parents=True, exist_ok=True
    )
    df.to_csv(args.output, index=False)
    print(df)


if __name__ == "__main__":
    main()
