from collections import Counter

import jieba
import numpy as np
import torch
from torch.utils.data import Dataset


PAD = "<PAD>"
UNK = "<UNK>"


def tokenize(text):
    return [
        tok.strip()
        for tok in jieba.lcut(str(text))
        if tok.strip()
    ]


def build_vocab(texts, max_vocab=30000):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {PAD: 0, UNK: 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_length):
    ids = [
        vocab.get(t, 1)
        for t in tokenize(text)
    ][:max_length]
    ids += [0] * (max_length - len(ids))
    return ids


class ClassicDataset(Dataset):
    def __init__(
        self, df, vocab, label2id, max_length
    ):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(
            encode(
                row["text"], self.vocab, self.max_length
            ),
            dtype=torch.long,
        )
        y = torch.tensor(
            self.label2id[row["label"]],
            dtype=torch.long,
        )
        return x, y


def train_word2vec(
    texts, vocab, embedding_dim=200, seed=42
):
    from gensim.models import Word2Vec

    sentences = [tokenize(t) for t in texts]
    model = Word2Vec(
        sentences=sentences,
        vector_size=embedding_dim,
        window=5,
        min_count=1,
        workers=1,
        sg=1,
        seed=seed,
        epochs=10,
    )
    matrix = np.random.default_rng(seed).normal(
        0, 0.02, size=(len(vocab), embedding_dim)
    ).astype("float32")
    matrix[0] = 0
    for word, idx in vocab.items():
        if word in model.wv:
            matrix[idx] = model.wv[word]
    return matrix
