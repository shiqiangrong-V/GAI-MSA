import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
)


class TransformerBaseline(nn.Module):
    def __init__(self, model_name, num_classes=6):
        super().__init__()
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_classes
            )
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        labels=None,
    ):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.model(**kwargs)
        return {"loss": out.loss, "logits": out.logits}


class BertBiGRU(nn.Module):
    def __init__(
        self,
        model_name="bert-base-chinese",
        gru_hidden=128,
        num_classes=6,
        dropout=0.3,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.gru = nn.GRU(
            input_size=h,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            gru_hidden * 2, num_classes
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        labels=None,
    ):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**kwargs).last_hidden_state
        z, _ = self.gru(hidden)
        mask = attention_mask.unsqueeze(-1).bool()
        z = z.masked_fill(
            ~mask, torch.finfo(z.dtype).min
        )
        pooled = z.max(dim=1).values
        logits = self.classifier(self.dropout(pooled))
        loss = (
            nn.functional.cross_entropy(logits, labels)
            if labels is not None else None
        )
        return {"loss": loss, "logits": logits}


class Word2VecLSTM(nn.Module):
    def __init__(
        self,
        embedding_matrix,
        hidden_dim=128,
        num_classes=6,
        dropout=0.3,
    ):
        super().__init__()
        emb = torch.tensor(
            embedding_matrix, dtype=torch.float
        )
        self.embedding = nn.Embedding.from_pretrained(
            emb, freeze=False, padding_idx=0
        )
        self.lstm = nn.LSTM(
            emb.shape[1],
            hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        z = self.embedding(x)
        _, (h, _) = self.lstm(z)
        return self.fc(self.dropout(h[-1]))


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=200,
        kernels=(3, 4, 5),
        filters=100,
        num_classes=6,
        padding_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(
                embedding_dim, filters, kernel_size=k
            )
            for k in kernels
        ])
        self.fc = nn.Linear(
            filters * len(kernels), num_classes
        )

    def forward(self, x):
        z = self.embedding(x).transpose(1, 2)
        z = [
            torch.max(torch.relu(conv(z)), dim=2).values
            for conv in self.convs
        ]
        return self.fc(torch.cat(z, dim=1))


class TextRCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=200,
        hidden_dim=128,
        num_classes=6,
        padding_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_proj = nn.Linear(
            embedding_dim + hidden_dim * 2,
            hidden_dim * 2,
        )
        self.fc = nn.Linear(
            hidden_dim * 2, num_classes
        )

    def forward(self, x):
        emb = self.embedding(x)
        ctx, _ = self.rnn(emb)
        z = torch.tanh(
            self.fc_proj(torch.cat([emb, ctx], dim=-1))
        )
        pooled = z.max(dim=1).values
        return self.fc(pooled)


class DPCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=200,
        channels=250,
        num_classes=6,
        padding_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.region = nn.Conv1d(
            embedding_dim, channels, kernel_size=3, padding=1
        )
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1
        )
        self.fc = nn.Linear(channels, num_classes)

    def _block(self, x):
        residual = torch.max_pool1d(
            x, kernel_size=3, stride=2, padding=1
        )
        z = torch.relu(residual)
        z = self.conv1(z)
        z = torch.relu(z)
        z = self.conv2(z)
        return z + residual

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        region = self.region(emb)
        z = torch.relu(region)
        z = self.conv1(z)
        z = torch.relu(z)
        z = self.conv2(z) + region
        while z.size(-1) > 2:
            z = self._block(z)
        pooled = z.max(dim=2).values
        return self.fc(pooled)
