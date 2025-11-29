# ts-transformer-spending
This project applies a Time-Series Transformer (TST) architecture using PyTorch + Hugging Face to forecast national health expenditure using public datasets (e.g., OECD/World Bank GHED).

### Features

Time Series Transformer implemented from scratch (PyTorch)

Modular and readable code structure

Config-driven experiments (configs/*.yaml)

Clean Jupyter notebooks for exploration → preprocessing → training → evaluation

### Repository Structure
CLI scripts for training and evaluation
```text
ts-transformer-spending/
│
├── data/
│   ├── raw/                # original CSVs
│   ├── processed/          # after preprocessing
│   └── README.md           # data dictionary
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_train_TST.ipynb       # simple, small, readable
│   └── 04_eval_forecast.ipynb
│
├── src/
│   ├── data/
│   │   └── preprocess.py
│   ├── models/
│   │   ├── transformer.py       # the Time Series Transformer
│   │   └── dataset.py           # PyTorch Dataset + Dataloader
│   ├── train.py                 # CLI training entry
│   ├── evaluate.py
│   └── utils.py
│
├── configs/
│   ├── default.yaml
│   └── transformer_small.yaml
│
├── requirements.txt
├── README.md
└── LICENSE
```

### Model Overview: Time Series Transformer (TST)

This project uses a Transformer Encoder architecture adapted for forecasting:

Sliding-window input sequences

Positional encodings

Multi-head self-attention across time

Feed-forward layers

Dropout + layer normalization

Linear projection head

Supports:

variable input lengths

multi-feature time series

autoregressive or direct multi-step forecasting





