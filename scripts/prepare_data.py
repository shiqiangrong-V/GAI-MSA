import argparse
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from gai_msa.data import load_csv
from gai_msa.utils import ensure_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    ensure_dir(args.out_dir)
    for name, path in [
        ("train", args.train),
        ("val", args.val),
        ("test", args.test),
    ]:
        df = load_csv(path)
        df.to_csv(
            Path(args.out_dir) / f"{name}.csv",
            index=False,
        )
        print(
            name,
            len(df),
            df["label"].value_counts().to_dict(),
        )


if __name__ == "__main__":
    main()
