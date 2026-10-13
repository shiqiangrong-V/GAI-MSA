#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_data.py \
  --train data/raw/train.csv \
  --val data/raw/val.csv \
  --test data/raw/test.csv \
  --out-dir data/processed

python scripts/train_sentiment_filter.py \
  --config config.yaml \
  --train data/processed/train.csv \
  --val data/processed/val.csv \
  --output-dir checkpoints/sentiment_filter

python scripts/generate_augmentation.py \
  --config config.yaml \
  --train data/processed/train.csv \
  --output data/augmentation/candidates.csv

python scripts/filter_augmentation.py \
  --config config.yaml \
  --train data/processed/train.csv \
  --candidates data/augmentation/candidates.csv \
  --sentiment-checkpoint checkpoints/sentiment_filter/best \
  --output data/augmentation/filtered.csv

python scripts/build_augmented_train.py \
  --config config.yaml \
  --train data/processed/train.csv \
  --filtered data/augmentation/filtered.csv \
  --output data/processed/train_augmented.csv

python scripts/train.py \
  --config config.yaml \
  --variant gai_msa \
  --train data/processed/train_augmented.csv \
  --val data/processed/val.csv \
  --output-dir checkpoints/gai_msa

python scripts/evaluate.py \
  --config config.yaml \
  --variant gai_msa \
  --checkpoint checkpoints/gai_msa/best \
  --test data/processed/test.csv \
  --output-dir results/gai_msa
