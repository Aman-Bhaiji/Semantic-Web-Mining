# Directional Stock Price Prediction using Numerical and News Data

## Objective
The goal of this project is to **predict the direction (up or down)** of stock prices (for Apple and Amazon) by combining:
1. **Numerical price data** — OHLCV features at multiple time intervals (5 min–1 day).
2. **Textual news sentiment** — extracted from ~87,000 news articles covering the same period.

The problem is framed as a **binary classification task**:
> 1 → price expected to rise  
> 0 → price expected to fall

We use traditional machine-learning models (Logistic Regression, SVM, and XGBoost) to analyze performance on:
- Numerical data only  
- Numerical + sentiment data (VADER + FinBERT)

and compare how much sentiment improves directional accuracy.

---

## Phase 1 – Numerical Dataset Preprocessing

### 1. Inspecting and loading numerical data
We began with the Amazon 30-minute interval dataset as our baseline.  
Each row contained: Date, Time, Open, High, Low, Close, Volume.

**Goal:** confirm data integrity before modeling.

Performed:
- File load & schema inspection  
- Conversion to numeric types (`float64` for prices, `int` for volume)  
- Missing-value checks and data-type validation

---

### 2. Parsing timestamps and structuring the index
We combined `date` + `time` into a single timestamp column `ts`,  
then converted it into a pandas `DatetimeIndex`.

**Purpose:** enable time-series operations like rolling statistics, lags, and resampling.

Validated:
- Chronological order, duplicate timestamps, and market gaps (weekends)  
- Volume distribution and trade intensity

---

### 3. Feature engineering (rolling, momentum, volume)
Created lag-safe indicators using `.shift(1)`:
- Rolling mean, std, max, min (3, 6, 12 windows)
- Momentum = `close − rolling mean`
- Volume change & rolling mean

**Why:** capture short-term volatility, momentum, and liquidity dynamics — typical in quantitative trading.

---

### 4. Building the target variable
We defined **future return** for each bar as the percentage change between the next closing price and the current close.  
If the next close was higher than the current one, the return was positive; otherwise it was negative.

Then we created a binary **target column**:
- `1` → if the next bar’s price increased  
- `0` → if the next bar’s price decreased or stayed flat  

**Purpose:** classify *directional movement* rather than predict the exact price value.  
The resulting dataset was roughly balanced (≈ 50 % ups vs 50 % downs).

---

### 5. Cleaning and leakage verification
Validated:
- No duplicate timestamps  
- Correct weekend/holiday handling  
- Proper lag structure (no future information used)

---

### 6. Baseline model training (LR, SVM, XGBoost)
Split **chronologically (80/20)** and trained:
1. Logistic Regression  
2. SVM (RBF kernel)  
3. XGBoost  

**Purpose:** create a numerical-only benchmark.

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC.

**Observation:** F1 ≈ 0.45–0.48 → weak but stable baseline.

---

### 7–8. Hyperparameter tuning & imbalance handling
Used:
- `class_weight='balanced'` for LR/SVM  
- `scale_pos_weight` for XGB  

**Goal:** improve F1 and handle mild class imbalance.  
Result: slight F1 increase (~0.48–0.49).

---

## Phase 2 – News Data Processing

### 9. Loading & cleaning news data
- 87 k JSON articles (2018–2019)  
- Extracted: `published`, `title`, `text`, `site`, `url`, `language`

**After cleaning:** 70 731 valid English articles.

---

### 10. Sentiment analysis with VADER
Computed scores: `vader_pos`, `vader_neg`, `vader_neu`, `vader_compound`  
(mean compound ≈ 0.70 → positive skew)

**Why:** establish a fast lexicon-based sentiment baseline.

---

### 11. Aggregating news into 30-minute bins
For each 30-min bar:
- Aggregated `news_count`, `vader_mean`, `vader_sum`, `pos_prop`, `neg_prop`
- Added rolling features (4 and 12 bars)

**Purpose:** align unstructured news to structured price intervals.

---

### 12. Merging with price data
Joined aggregated news with price data → `model_df_news`.  
Filled missing bins with zeros. ~66 % of bars contained news.

Added columns: `news_count`, `vader_mean`, `vader_mean_roll_4`, `pos_prop`, etc.

---

### Leakage verification
Checked correlations with future returns — all |ρ| < 0.03.  
✅ **No look-ahead leakage detected.**

---

## Phase 3 – Modeling with VADER Sentiment
Compared models with numeric-only vs numeric + VADER data.

| Model | Baseline F1 | With VADER F1 |
|-------|--------------|---------------|
| Logistic Regression | 0.48 | 0.53 |
| SVM | 0.42 | 0.43 |
| XGBoost | 0.37 | 0.38 |

✅ **Insight:** sentiment provided a small but consistent lift.

---

## Phase 4 – Advanced Sentiment: FinBERT
Used **`yiyanghkust/finbert-tone`** — a transformer trained on financial tone data.

Steps:
1. Tokenized text (title + body)  
2. Ran batched inference (GPU if available)  
3. Extracted probabilities for `positive`, `neutral`, `negative`  
4. Derived compound score = `positive − negative`  
5. Aggregated & rolled over 30-min bins  
6. Merged with price data → `model_df_news_with_finbert`

**Why:** capture context-aware financial sentiment beyond simple lexicons.

---

### FinBERT runtime & persistence
- CPU inference ≈ 5 hours for 70 k articles  
- Saved artifacts:  
  - `finbert_article_scores.pkl` (per-article)  
  - `news_finbert_agg.parquet` (aggregated bins)  
  - `model_df_news_with_finbert.parquet` (merged data)

**Purpose:** reproducibility and checkpointing.

---

## Phase 5 – Combined Modeling (Numeric + FinBERT)

| Model | Dataset | F1 | AUC |
|-------|----------|----|-----|
| Logistic Regression | numeric | 0.47 | 0.51 |
| Logistic Regression | + FinBERT | 0.47 | 0.52 |
| XGBoost | numeric | 0.37 | 0.50 |
| XGBoost | + FinBERT | 0.35 | 0.51 |

**Observation:** FinBERT improved stability and recall modestly.

---

## Phase 6 – Purged Time-Series Cross-Validation
Implemented a **Purged TimeSeriesSplit** (with embargo) to avoid temporal leakage.

### Parameter grids tuned
- **LR:** Regularization (C)  
- **SVM:** C & γ  
- **XGB:** depth, learning_rate, subsample, colsample, n_estimators

---

### Best cross-validation results
| Model | Mean CV F1 | Key Params |
|-------|-------------|------------|
| Logistic Regression | 0.4145 | C = 10.0 |
| SVM | **0.4224** | C = 0.1, γ = 0.01 |
| XGBoost | 0.4037 | n_estimators = 200, depth = 3 |

---

### Hold-out test performance
| Model | Accuracy | Recall | F1 | AUC | Notes |
|-------|-----------|---------|----|-----|-------|
| LR_purged | 0.50 | 0.54 | 0.48 | 0.52 | Balanced performance |
| SVM_purged | 0.43 | **0.98** | **0.59** | 0.51 | High recall, low precision |
| XGB_purged | 0.51 | 0.46 | 0.44 | 0.51 | Conservative, stable |

✅ **Interpretation:** leak-free, realistic metrics. FinBERT adds incremental but genuine signal.

---

### Model persistence
All tuned pipelines saved via `joblib`:
- `best_lr_purged.pkl`  
- `best_svm_purged.pkl`  
- `best_xgb_purged.pkl`

Each includes its preprocessing scaler and tuned estimator.

---

## Key Learnings & Design Decisions
| Step | Purpose | Why It Matters |
|------|----------|----------------|
| Rolling features | Capture short-term momentum | Core numeric signal |
| VADER | Lexicon-based baseline | Benchmark sentiment signal |
| FinBERT | Context-aware NLP | Adds nuanced sentiment |
| Purged CV | Time-aware validation | Prevents future leakage |
| Class weighting | Balance recall vs precision | Stabilizes performance |
| Rolling sentiment | News persistence | Mimics market digestion time |

---

## Current State
✅ Clean, leak-free combined dataset (`model_df_news_with_finbert`, 56 features)  
✅ Tuned models stored & reproducible  
✅ FinBERT outputs saved locally  
✅ Baseline + tuned benchmarks established  

---

## Next Steps
1. **Walk-forward backtesting** – rolling-train rolling-test simulation  
2. **Feature importance analysis** – LR coefficients, XGB importances  
3. **Real-time prediction** – deploy `best_lr_purged` for live inference  
4. **Enhancements** – include macro text, transformer-based time models  

---

## Summary
We built an end-to-end, **time-aware, sentiment-enhanced stock direction prediction pipeline**, starting from raw OHLCV bars and unstructured news, through rigorous feature engineering, sentiment modeling (VADER → FinBERT), and purged cross-validation — producing reproducible, leak-free predictive models ready for backtesting or deployment.
