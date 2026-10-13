import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from gai_msa.baselines import (
    DPCNN,
    TextCNN,
    TextRCNN,
    Word2VecLSTM,
)
from gai_msa.classic_data import (
    ClassicDataset,
    build_vocab,
    train_word2vec,
)
from gai_msa.constants import LABEL2ID
from gai_msa.data import load_csv
from gai_msa.metrics import compute_metrics
from gai_msa.utils import (
    ensure_dir,
    load_config,
    resolve_device,
    save_json,
    set_seed,
)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    yt, yp = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=-1)
        yt.extend(y.cpu().tolist())
        yp.extend(pred.cpu().tolist())
    return compute_metrics(yt, yp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--model",
        choices=[
            "word2vec_lstm",
            "textcnn",
            "textrcnn",
            "dpcnn",
        ],
        required=True,
    )
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    ccfg = cfg["baselines"]["classic"]
    set_seed(cfg["seed"])
    ensure_dir(args.output_dir)
    device = resolve_device(
        cfg["training"]["device"]
    )

    train = load_csv(args.train)
    val = load_csv(args.val)
    test = load_csv(args.test)

    vocab = build_vocab(
        train["text"].tolist(),
        ccfg["max_vocab"],
    )
    with open(
        Path(args.output_dir) / "vocab.pkl",
        "wb",
    ) as f:
        pickle.dump(vocab, f)

    train_ds = ClassicDataset(
        train, vocab, LABEL2ID,
        ccfg["max_length"]
    )
    val_ds = ClassicDataset(
        val, vocab, LABEL2ID,
        ccfg["max_length"]
    )
    test_ds = ClassicDataset(
        test, vocab, LABEL2ID,
        ccfg["max_length"]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=ccfg["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=ccfg["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=ccfg["batch_size"],
        shuffle=False,
    )

    if args.model == "word2vec_lstm":
        matrix = train_word2vec(
            train["text"].tolist(),
            vocab,
            ccfg["embedding_dim"],
            cfg["seed"],
        )
        model = Word2VecLSTM(
            matrix,
            hidden_dim=ccfg["hidden_dim"],
        ).to(device)
    elif args.model == "textcnn":
        model = TextCNN(
            len(vocab),
            embedding_dim=ccfg["embedding_dim"],
            kernels=tuple(
                ccfg["textcnn_kernels"]
            ),
            filters=ccfg["textcnn_filters"],
        ).to(device)
    elif args.model == "textrcnn":
        model = TextRCNN(
            len(vocab),
            embedding_dim=ccfg["embedding_dim"],
            hidden_dim=ccfg["hidden_dim"],
        ).to(device)
    else:
        model = DPCNN(
            len(vocab),
            embedding_dim=ccfg["embedding_dim"],
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(ccfg["learning_rate"]),
    )
    best = -1.0
    best_path = (
        Path(args.output_dir) / "best.pt"
    )

    for epoch in range(
        1, int(ccfg["epochs"]) + 1
    ):
        model.train()
        for x, y in tqdm(
            train_loader,
            desc=f"{args.model} epoch {epoch}",
        ):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits, y
            )
            loss.backward()
            optimizer.step()

        metrics = evaluate(
            model, val_loader, device
        )
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            torch.save(
                model.state_dict(), best_path
            )

    model.load_state_dict(
        torch.load(best_path, map_location=device)
    )
    metrics = evaluate(
        model, test_loader, device
    )
    save_json(
        metrics,
        Path(args.output_dir) /
        "metrics.json",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
