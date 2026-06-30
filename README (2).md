# High-Frequency Mid-Price Movement Forecasting (LOB)

A deep-learning pipeline that predicts short-horizon **mid-price movements** from high-frequency **Limit Order Book (LOB)** data for Amazon (AMZN) on NASDAQ. The task is framed as a three-class problem — predicting whether the mid-price will move **Up**, stay **Stationary**, or move **Down** after a fixed horizon of `k = 10` events.

> **TL;DR** — A 3-model ensemble (MLP + CNN-LSTM + LOB-BERT) reaches **0.46 Macro F1**, lifting performance from a **0.36** linear baseline and matching published FI-2010 LOB benchmarks on a harder single-day, high-liquidity setting.

---

## Overview

Forecasting short-term price direction from the order book is a core problem in market-microstructure research and high-frequency trading. Unlike daily-bar prediction, LOB forecasting is dominated by **class imbalance**, **non-stationarity**, and **noise**, which makes out-of-sample generalisation — not raw model capacity — the central challenge.

This project builds an end-to-end pipeline covering microstructure feature engineering, leakage-safe feature selection, a family of deep-learning models, and an ensemble strategy, benchmarked against a linear baseline and the FI-2010 literature.

**Hypotheses tested**
1. Domain-specific feature engineering beats raw price inputs.
2. Smaller, regularised models generalise better than larger ones on noisy intraday data.
3. Ensembling architecturally diverse models improves over any single model.

---

## Dataset

| Property | Value |
|---|---|
| Instrument | Amazon (AMZN), NASDAQ |
| Session | Single trading day (22 Sep 2015), 04:00–20:00 EST |
| Events | ~559,700 |
| Depth | Top 10 price & volume levels (bid + ask) |
| Fields | Per-level price/volume, mid-price, bid-ask spread |
| Price units | 1/10,000 of a dollar (median mid-price ≈ $536.40, median spread ≈ $0.041) |

**Labelling.** A label is assigned by comparing the mid-price `k = 10` events ahead against a threshold `θ = 0.0001` raw units:

```
l_t = Up         if m_{t+k} - m_t >  θ
l_t = Down       if m_{t+k} - m_t < -θ
l_t = Stationary otherwise
```

Resulting class distribution: **~54% Stationary, ~23% Down, ~23% Up**.

**Splitting.** Strict temporal **60 / 20 / 20** train/validation/test split to prevent look-ahead bias. The test set covers the final 20% of the session, during which volatility declines toward market close — creating a mild regime shift that penalises models that overfit mid-session dynamics.

---

## Methodology

### 1. Feature engineering (147 candidate features)
All features are computed using only information available at time `t`.

- **Depth & imbalance** — buy/sell pressure across LOB levels.
- **Order Flow Imbalance (OFI)** — incremental changes in queued bid/ask volumes.
- **Microprice** — volume-weighted best-level price that accounts for queue asymmetry.
- **Price-distance features** — bid/ask prices expressed relative to the mid-price, making them stationary across price regimes.
- **Rolling statistics** (windows 5 and 10) — short-term momentum of imbalance, microprice deviation, OFI, and spread volatility.

### 2. Leakage-safe feature selection (train data only)
1. **Constant removal** — zero-variance features dropped (e.g. hour-of-day, constant on a single day).
2. **Correlation filtering** — pairwise Pearson `> 0.95` removed (42 redundant features eliminated).
3. **Mutual information** — top 60 features by MI with training labels retained. Microprice ranks highest (MI = 0.404), followed by multi-level depth imbalance.

### 3. Normalisation
`StandardScaler` (Z-score), **fit on the training set only** and applied to validation/test — more robust than min-max to outliers from large institutional orders.

---

## Models

| Model | Role | Notes |
|---|---|---|
| **Logistic Regression** | Linear baseline | `class_weight='balanced'`, `C=0.1` |
| **RBFNN** | Non-temporal NN | k-means RBF centres + multinomial output |
| **MLP** | Flat-feature NN | hidden (128, 64), dropout 0.4, Weighted CE |
| **CNN-LSTM** | Local + sequential | Conv (BatchNorm) → LSTM, processes `T=10` snapshots |
| **LOB-BERT** | Global attention | BERT-style encoder, learnable CLS token, `T=10` snapshots |
| **Ensemble** | Final model | Equal-weight softmax averaging of MLP + CNN-LSTM + LOB-BERT |

A consistent finding across families: **smaller, well-regularised configurations (dropout 0.4) generalise better**, and **Weighted Cross-Entropy is more stable than Focal Loss** on this imbalanced data.

---

## Results

**Primary metric: Macro F1** (treats all three classes equally; not inflated by the majority Stationary class).

| Model | Test Accuracy | Test Macro F1 |
|---|---|---|
| Logistic Regression (baseline) | 0.40 | 0.36 |
| RBFNN (100 centres) | 0.47 | 0.39 |
| MLP (dropout 0.4) | 0.52 | 0.43 |
| CNN-LSTM (dropout 0.4) | 0.51 | 0.45 |
| LOB-BERT (Tiny) | 0.51 | 0.45 |
| Ensemble (CNN-LSTM + BERT) | 0.52 | 0.4572 |
| **Ensemble (MLP + CNN-LSTM + BERT)** | **0.54** | **0.4641** |

**Per-class breakdown (best ensemble)**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Down | 0.32 | 0.27 | 0.29 |
| Stationary | 0.70 | 0.65 | 0.68 |
| Up | 0.37 | 0.49 | 0.42 |
| **Macro avg** | **0.47** | **0.47** | **0.46** |

**Context.** Ntakaris et al. (2018) report ≈46% F1 on FI-2010 (10 days, less liquid exchange); Prata et al. (2024) show SOTA models dropping to 48–61% F1 on real LOBSTER data. The 0.46 here is achieved on a single day of highly liquid NASDAQ data, making it competitive given the harder setting.

---

## Key Findings

- **Microprice and depth imbalance** are the most informative features (MI = 0.404 and 0.308).
- **Capacity ≠ generalisation** — smaller models with dropout 0.4 consistently beat larger ones on noisy LOB data.
- **Weighted CE > Focal Loss** for training stability under imbalance.
- **Diversity drives ensemble gains** — equal weighting beats up-weighting the strongest single model; complementary error patterns matter more than individual strength.
- **The Down class is hardest** (F1 = 0.29) — sell-side pressure is rapidly absorbed by the bid queue and often masked by iceberg/split orders on a large-cap name.

---

## Project Structure

```
.
├── data/                # raw LOB snapshots (not tracked)
├── src/
│   ├── features.py      # microstructure feature engineering
│   ├── selection.py     # constant / correlation / MI selection
│   ├── datasets.py      # labelling, temporal split, scaling
│   ├── models/          # MLP, CNN-LSTM, LOB-BERT, RBFNN, baseline
│   ├── train.py         # training loop (AdamW, early stopping)
│   └── ensemble.py      # softmax averaging + weight search
├── notebooks/           # EDA, MI scores, correlation heatmaps
├── results/             # metrics, figures
├── requirements.txt
└── README.md
```

> Adjust the structure above to match your actual repository layout.

---

## Setup & Usage

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Build features and labels
python src/features.py --data data/amzn_lob.csv --out data/features.parquet

# 3. Train a single model
python src/train.py --model cnn_lstm --config configs/cnn_lstm.yaml

# 4. Train all models and evaluate the ensemble
python src/ensemble.py --models mlp cnn_lstm lob_bert
```

> Commands are templates — update the entry points and config paths to your code.

---

## Limitations & Future Work

- **Single trading day** — generalisation to other regimes/volatility is untested; multi-day training would enable anchored cross-validation.
- **No execution realism** — transaction costs, market impact, and latency are not modelled, all of which erode signal profitability in live trading.
- **Down-class reliability** — F1 = 0.29 implies a high false-positive rate for sell signals; costly in execution.
- **Next steps** — multi-day data, adaptive (spread-based) labelling thresholds, and a dual-output architecture predicting both **direction and timing**.

---

## References

- Ntakaris et al. (2018). *Benchmark dataset for mid-price forecasting of limit order book data.*
- Prata et al. (2024). *LOB-based deep learning models for stock price trend prediction (LOBCAST).*
- Stoikov (2018). *The micro-price: a high-frequency estimator of future prices.*
- Lin et al. (2017). *Focal Loss for Dense Object Detection.*

---

*Coursework: Financial Markets Applications of Deep & Reinforcement Learning — University of Edinburgh.*
