import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from gai_msa.baselines import (
    BertBiGRU,
    TransformerBaseline,
)
from gai_msa.data import (
    HFSentimentDataset,
    collate_hf,
    load_csv,
)
from gai_msa.metrics import compute_metrics
from gai_msa.utils import (
    ensure_dir,
    load_config,
    resolve_device,
    save_json,
    set_seed,
)


def move(batch, device):
    return {
        k: v.to(device)
        for k, v in batch.items()
        if k != "text"
    }


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    yt, yp = [], []
    for batch in loader:
        b = move(batch, device)
        out = model(
            input_ids=b["input_ids"],
            attention_mask=b["attention_mask"],
            token_type_ids=b.get(
                "token_type_ids"
            ),
            labels=b["labels"],
        )
        pred = out["logits"].argmax(dim=-1)
        yt.extend(b["labels"].cpu().tolist())
        yp.extend(pred.cpu().tolist())
    return compute_metrics(yt, yp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--model",
        choices=["bert", "ernie", "bert_bigru"],
        required=True,
    )
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ensure_dir(args.output_dir)
    device = resolve_device(
        cfg["training"]["device"]
    )

    if args.model == "bert":
        name = cfg["baselines"]["transformer"][
            "bert_name"
        ]
        model = TransformerBaseline(name).to(device)
    elif args.model == "ernie":
        name = cfg["baselines"]["transformer"][
            "ernie_name"
        ]
        model = TransformerBaseline(name).to(device)
    else:
        name = cfg["baselines"]["transformer"][
            "bert_name"
        ]
        model = BertBiGRU(
            name,
            gru_hidden=cfg["baselines"][
                "transformer"
            ]["bigru_hidden"],
        ).to(device)

    train = load_csv(args.train)
    val = load_csv(args.val)
    test = load_csv(args.test)

    def make_loader(df, shuffle):
        ds = HFSentimentDataset(
            df,
            name,
            cfg["model"]["max_length"],
        )
        return DataLoader(
            ds,
            batch_size=cfg["training"][
                "batch_size"
            ],
            shuffle=shuffle,
            num_workers=cfg["training"][
                "num_workers"
            ],
            collate_fn=collate_hf,
        )

    train_loader = make_loader(train, True)
    val_loader = make_loader(val, False)
    test_loader = make_loader(test, False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            cfg["training"]["learning_rate"]
        ),
        weight_decay=float(
            cfg["training"]["weight_decay"]
        ),
    )

    best = -1.0
    best_path = (
        Path(args.output_dir) / "best.pt"
    )

    for epoch in range(
        1, int(cfg["training"]["epochs"]) + 1
    ):
        model.train()
        for batch in tqdm(
            train_loader,
            desc=f"{args.model} epoch {epoch}",
        ):
            b = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"],
                token_type_ids=b.get(
                    "token_type_ids"
                ),
                labels=b["labels"],
            )
            out["loss"].backward()
            optimizer.step()

        metrics = eval_model(
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
    metrics = eval_model(
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
