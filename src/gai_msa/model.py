from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class GAIMSAConfig:
    encoder_name: str
    num_classes: int = 6
    kernel_sizes: tuple = (2, 3, 4)
    filters_per_kernel: int = 128
    attention_projection_dim: int = 128
    dropout: float = 0.30
    use_cnn: bool = True
    use_attention: bool = True


class AdditiveTokenAttention(nn.Module):
    def __init__(self, hidden_size=768, projection_dim=128):
        super().__init__()
        self.proj = nn.Linear(hidden_size, projection_dim)
        self.context = nn.Parameter(torch.empty(projection_dim))
        nn.init.normal_(self.context, mean=0.0, std=0.02)

    def forward(self, hidden, attention_mask):
        u = torch.tanh(self.proj(hidden))
        scores = torch.einsum("bla,a->bl", u, self.context)
        scores = scores.masked_fill(
            attention_mask == 0, torch.finfo(scores.dtype).min
        )
        alpha = torch.softmax(scores, dim=1)
        context = torch.einsum("bl,blh->bh", alpha, hidden)
        return context, alpha


class MultiScaleCNN(nn.Module):
    def __init__(self, hidden_size=768, kernel_sizes=(2, 3, 4), filters_per_kernel=128):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, filters_per_kernel, kernel_size=k)
            for k in kernel_sizes
        ])
        self.activation = nn.ReLU()
        self.output_dim = len(kernel_sizes) * filters_per_kernel

    def forward(self, hidden):
        x = hidden.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            z = self.activation(conv(x))
            pooled.append(torch.max(z, dim=2).values)
        return torch.cat(pooled, dim=1)


class GAIMSA(nn.Module):
    # ernie -> CLS only
    # ernie_cnn -> CNN branch
    # ernie_attention -> attention branch
    # ernie_cnn_attention -> CNN + attention, original train set
    # gai_msa -> CNN + attention, augmented train set
    def __init__(self, cfg: GAIMSAConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.encoder_name)
        hidden_size = int(self.encoder.config.hidden_size)

        self.cnn = None
        self.attention = None
        dims = []

        if cfg.use_cnn:
            self.cnn = MultiScaleCNN(
                hidden_size=hidden_size,
                kernel_sizes=tuple(cfg.kernel_sizes),
                filters_per_kernel=cfg.filters_per_kernel,
            )
            dims.append(self.cnn.output_dim)

        if cfg.use_attention:
            self.attention = AdditiveTokenAttention(
                hidden_size=hidden_size,
                projection_dim=cfg.attention_projection_dim,
            )
            dims.append(hidden_size)

        if not dims:
            dims = [hidden_size]

        self.feature_dim = sum(dims)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(self.feature_dim, cfg.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        hidden = outputs.last_hidden_state

        features = []
        attention_weights = None

        if self.cnn is not None:
            features.append(self.cnn(hidden))

        if self.attention is not None:
            att_feature, attention_weights = self.attention(hidden, attention_mask)
            features.append(att_feature)

        if not features:
            features = [hidden[:, 0, :]]

        fused = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        logits = self.classifier(self.dropout(fused))
        loss = (
            nn.functional.cross_entropy(logits, labels)
            if labels is not None else None
        )

        return {
            "loss": loss,
            "logits": logits,
            "features": fused,
            "attention_weights": attention_weights,
        }


def build_variant(config, variant):
    model_cfg = config["model"]
    use_cnn = variant in {"ernie_cnn", "ernie_cnn_attention", "gai_msa"}
    use_attention = variant in {
        "ernie_attention", "ernie_cnn_attention", "gai_msa"
    }

    return GAIMSA(GAIMSAConfig(
        encoder_name=model_cfg["encoder_name"],
        num_classes=model_cfg["num_classes"],
        kernel_sizes=tuple(model_cfg["cnn"]["kernel_sizes"]),
        filters_per_kernel=model_cfg["cnn"]["filters_per_kernel"],
        attention_projection_dim=model_cfg["attention"]["projection_dim"],
        dropout=model_cfg["dropout"],
        use_cnn=use_cnn,
        use_attention=use_attention,
    ))
