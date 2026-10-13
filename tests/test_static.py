from pathlib import Path

import yaml


def test_config_dimensions():
    cfg = yaml.safe_load(
        Path("config.yaml").read_text(
            encoding="utf-8"
        )
    )
    filters = cfg["model"]["cnn"][
        "filters_per_kernel"
    ]
    kernels = cfg["model"]["cnn"][
        "kernel_sizes"
    ]
    cnn_dim = filters * len(kernels)
    att_dim = cfg["model"]["hidden_size"]
    assert cnn_dim == 384
    assert att_dim == 768
    assert cnn_dim + att_dim == 1152


def test_labels():
    cfg = yaml.safe_load(
        Path("config.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert cfg["labels"] == [
        "happy",
        "angry",
        "sad",
        "fear",
        "surprise",
        "neutral",
    ]


def test_reported_augmentation_counts():
    cfg = yaml.safe_load(
        Path("config.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert sum(
        cfg["augmentation"][
            "candidate_quotas"
        ].values()
    ) == 11500
    assert sum(
        cfg["augmentation"][
            "retained_targets"
        ].values()
    ) == 8120
