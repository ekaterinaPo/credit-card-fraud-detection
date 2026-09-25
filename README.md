# Credit Card Fraud Detection — Imbalanced Classification

**Kate Pogrebnyakova | Data Analyst / Data Scientist**

## Business question

A bank processes 284,807 credit card transactions over 2 days. 492 are fraudulent — just 0.17% of all transactions. How do you build a model that catches fraud without blocking too many legitimate customers?

This is a precision-recall tradeoff, not an accuracy problem. A missed fraud (false negative) costs the bank the full fraud loss; a false alarm (false positive) costs customer friction and potential churn. The right model isn't the most accurate one — it's the one that minimizes total business cost.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle / ULB Machine Learning Group) — 284,807 European cardholder transactions over 2 days (Sept 2013), 492 fraud cases (0.173%). Features V1–V28 are PCA-transformed for privacy; `Time` and `Amount` are raw.

## Approach

1. **EDA** — confirmed extreme class imbalance, examined amount/time distributions by class, identified the most discriminative PCA features
2. **Preprocessing** — scaled only `Time`/`Amount` (V-features are already PCA-scaled), stratified train/test split to preserve the fraud rate in both sets
3. **Imbalance handling** — compared three approaches: SMOTE (Logistic Regression), `class_weight='balanced'` (Random Forest), `scale_pos_weight` (XGBoost) — SMOTE applied to the training set only, never the test set
4. **Modeling** — trained and compared 3 classifiers, evaluated on Precision-Recall AUC (not accuracy — a model predicting "no fraud" for every transaction scores 99.83% accuracy while catching zero fraud)
5. **Threshold optimization** — swept classification thresholds against explicit business cost assumptions (€200 per missed fraud, €10 per false alarm) to find the cost-minimizing threshold, instead of defaulting to 0.5

## Key findings

| Model | PR-AUC | Recall | Precision | Missed Fraud | False Alarms |
|---|---|---|---|---|---|
| Logistic Regression | 0.7249 | 91.8% | 5.8% | 8 | 1,458 |
| Random Forest | 0.8653 | 75.5% | 96.1% | 24 | 3 |
| **XGBoost** | **0.8810** | 83.7% | 87.2% | 16 | 12 |

- XGBoost gives the best balance of catching fraud (83.7% recall) without excessive false alarms — Logistic Regression catches more fraud but fires 1,458 false alerts, which would overwhelm a real review team.
- **The default 0.5 threshold is not optimal.** Sweeping thresholds against real cost assumptions shifted the optimal point to ~0.35 — catching more fraud at lower total business cost than the statistical default.
- Top predictive features: V14, V17, V12, V10, V4 (PCA components).

## Why this matters — the accounting angle

In reconciliation work, catching every discrepancy matters more than minimizing review time — the cost of a missed error far exceeds the cost of double-checking a clean transaction. Fraud detection has the same cost asymmetry: this project treats the classification threshold as a business decision, not a statistical default, which is where most technical-only approaches stop short.

## Tech stack

Python, pandas, scikit-learn, XGBoost, imbalanced-learn (SMOTE), matplotlib, seaborn

## Project structure

```
04-Credit-Card-Fraud-Detection/
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   └── 02_modelling.ipynb
├── data/raw/                   — dataset not committed (see below)
└── requirements.txt
```

## How to run

```
pip install -r requirements.txt
```
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) to `data/raw/` (not committed — 144MB), then run the notebooks in order.
