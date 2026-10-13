# GAI-MSA: Generative AI-Driven Multi-Scale Sentiment Analysis

This repository is a complete runnable implementation aligned with the manuscript:

**A Generative AI-Driven Framework for Multi-Scale Sentiment Analysis in Social Media Public Opinion Monitoring**

It implements the paper's main pipeline:

1. SMP2020 train/validation/test split
2. GPT-4o-mini training-only data augmentation
3. Length filtering
4. Semantic-similarity filtering with `paraphrase-multilingual-MiniLM-L12-v2`
5. Sentiment-consistency filtering with a frozen ERNIE classifier
6. ERNIE 3.0 Base Chinese encoder
7. Parallel multi-scale CNN and additive-attention branches
8. Feature concatenation and 6-class classifier
9. Training/evaluation with Accuracy and macro Precision/Recall/F1
10. Ablation variants
11. Baseline implementations and result export

## Paper configuration

- Generative model: `gpt-4o-mini`
- Temperature: `0.70`
- Top-p: `0.90`
- Max output tokens: `128`
- Up to 5 candidate rewrites per selected source example
- Length filter: 5–80 non-whitespace characters
- Semantic model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Cosine-similarity range: `[0.78, 0.98]`
- Sentiment consistency: frozen ERNIE classifier; predicted label must match source label; confidence >= `0.70`
- ERNIE checkpoint: `nghuyong/ernie-3.0-base-zh`
- ERNIE hidden size: 768
- CNN kernels: `[2, 3, 4]`
- Filters per CNN branch: 128
- Attention projection: 128
- CNN output: 384
- Attention output: 768
- Fusion output: 1152
- Dropout: 0.30
- Sequence length: 256
- Batch size: 32
- Epochs: 20
- Learning rate: 5e-6
- Optimizer: Adam
- Weight decay: 0.02
- Seed: 42

## Expected data format

Place the official/adopted SMP2020 partitions as:

```text
data/raw/train.csv
data/raw/val.csv
data/raw/test.csv
```

Each file must contain:

```csv
text,label
今天真的很开心,happy
太让人生气了,angry
```

Valid labels:

```text
happy, angry, sad, fear, surprise, neutral
```

The repository does **not** redistribute the benchmark dataset.

## Installation

Python 3.8 is recommended to match the manuscript environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPT augmentation:

```bash
export OPENAI_API_KEY="your_key_here"
```

## Pipeline

### 1. Normalize fixed partitions

```bash
python scripts/prepare_data.py \
  --train data/raw/train.csv \
  --val data/raw/val.csv \
  --test data/raw/test.csv \
  --out-dir data/processed
```

### 2. Train the frozen ERNIE sentiment-consistency filter

```bash
python scripts/train_sentiment_filter.py \
  --config config.yaml \
  --train data/processed/train.csv \
  --val data/processed/val.csv \
  --output-dir checkpoints/sentiment_filter
```

### 3. Generate GPT candidates

```bash
python scripts/generate_augmentation.py \
  --config config.yaml \
  --train data/processed/train.csv \
  --output data/augmentation/candidates.csv
```

### 4. Filter candidates

```bash
python scripts/filter_augmentation.py \
  --config config.yaml \
  --train data/processed/train.csv \
  --candidates data/augmentation/candidates.csv \
  --sentiment-checkpoint checkpoints/sentiment_filter/best \
  --output data/augmentation/filtered.csv
```

### 5. Build augmented train set

```bash
python scripts/build_augmented_train.py \
  --config config.yaml \
  --train data/processed/train.csv \
  --filtered data/augmentation/filtered.csv \
  --output data/processed/train_augmented.csv
```

The script enforces the manuscript's retained-sample caps if enough valid generated samples are available. It does not fabricate missing examples.

### 6. Train GAI-MSA

```bash
python scripts/train.py \
  --config config.yaml \
  --variant gai_msa \
  --train data/processed/train_augmented.csv \
  --val data/processed/val.csv \
  --output-dir checkpoints/gai_msa
```

### 7. Evaluate

```bash
python scripts/evaluate.py \
  --config config.yaml \
  --variant gai_msa \
  --checkpoint checkpoints/gai_msa/best \
  --test data/processed/test.csv \
  --output-dir results/gai_msa
```

Outputs:

- `metrics.json`
- `classification_report.csv`
- `confusion_matrix.csv`
- `predictions.csv`

### 8. Ablation study

```bash
python scripts/run_ablation.py \
  --config config.yaml \
  --train-original data/processed/train.csv \
  --train-augmented data/processed/train_augmented.csv \
  --val data/processed/val.csv \
  --test data/processed/test.csv \
  --output-dir results/ablation
```

Ablation variants:

- `ernie`
- `ernie_cnn`
- `ernie_attention`
- `ernie_cnn_attention`
- `gai_msa`

### 9. Baselines

Transformer baselines:

```bash
python scripts/train_baseline.py --config config.yaml --model bert \
  --train data/processed/train.csv --val data/processed/val.csv \
  --test data/processed/test.csv --output-dir results/baselines/bert
```

Supported: `bert`, `ernie`, `bert_bigru`.

Classic baselines:

```bash
python scripts/train_classic_baseline.py --config config.yaml --model textcnn \
  --train data/processed/train.csv --val data/processed/val.csv \
  --test data/processed/test.csv --output-dir results/baselines/textcnn
```

Supported: `word2vec_lstm`, `textcnn`, `textrcnn`, `dpcnn`.

The manuscript does not fully specify every hyperparameter of the comparison baselines. Their defaults are documented in `config.yaml` and should be replaced if the original baseline settings are available.

### 10. Prediction

```bash
python scripts/predict.py \
  --config config.yaml \
  --checkpoint checkpoints/gai_msa/best \
  --text "今天真的太开心了！"
```

## Zenodo release

For the final archival release, include code, configs, environment, exact generated/filtered augmentation artifacts where licensing permits, and run logs. Do not redistribute third-party benchmark data if its license forbids it.
