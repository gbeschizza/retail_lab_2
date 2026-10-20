# Retail Repeat Purchase Prediction

Predicting whether a retail order leads to a repeat purchase, using exploratory analysis, feature engineering, multilingual LLM sentiment scoring of customer feedback, and a CatBoost classifier.

## Summary

| | |
|---|---|
| **Task** | Binary classification (`RepeatPurchase`: 1 = repeat, 0 = no repeat) |
| **Data** | 8,047 orders across 15 European countries (56% repeat / 44% no repeat) |
| **Model** | CatBoost with categorical features handled natively |
| **Result** | ROC-AUC ≈ 0.86, accuracy ≈ 0.80 on a held-out 20% test set |
| **Top drivers** | `Profit` and LLM-derived feedback sentiment (SHAP) |

## Approach

1. **Data wrangling and EDA**
   - Split the combined `Discount, Profit` column into two numeric features.
   - Extracted an `Order Prefix` (AZ / BN) from `Order ID` and found a moderate association with the target (Cramér's V ≈ 0.26) and a strong one with `Country` (Cramér's V ≈ 0.52).
   - Derived month, half-month and weekday from `Order Date`; none showed a meaningful association with the target and they were dropped.
   - Dropped `Category` (redundant with `Sub-Category`), and `Segment` / `Ship Mode` (no association with the target).
   - Tested associations with chi-square and Cramér's V (categorical), point-biserial correlation and Mann-Whitney U (numeric vs. target).

2. **Sentiment from free text**
   - `FeedbackText` contains reviews in several languages, with typos.
   - VADER was tried first as a baseline but handles non-English text poorly.
   - Claude was used to score each review from -1 to 1. A manual validation on a sample of 30 reviews gave a Spearman correlation of ≈ 0.67 (MAE ≈ 0.16).
   - Final correlation of the LLM sentiment with the target: point-biserial r ≈ 0.46, the strongest single feature.

3. **Modeling**
   - Stratified 80/20 train/test split.
   - CatBoost baseline (500 iterations, depth 6, learning rate 0.05, early stopping).
   - Random search over depth, learning rate, L2 regularization and iterations gave no improvement over the baseline (ROC-AUC ≈ 0.86).
   - Feature importance and SHAP values used for interpretation.

## Results

| Metric | Class 0 | Class 1 |
|---|---|---|
| Precision | 0.83 | 0.78 |
| Recall | 0.68 | 0.89 |
| F1 | 0.75 | 0.83 |

Overall accuracy 0.80, ROC-AUC 0.86.

Feature importance (CatBoost):

| Feature | Importance |
|---|---|
| Profit | 40.3 |
| llm_sentiment | 39.5 |
| Sub-Category | 8.5 |
| Discount | 6.6 |
| Country | 3.6 |
| Order Prefix | 1.5 |

Note that `Profit`, `Discount`, `Sub-Category` and `Country` are correlated or associated with each other, so importance is shared among them and the ranking should be read with that in mind.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── Laboratorio_II_G_Beschizza.ipynb
├── data/
│   ├── retail_lab_2.csv             # original data (if shareable)
│   └── df_with_sentiment.pkl        # precomputed LLM sentiment scores
└── figures/
```

## How to run

```bash
pip install -r requirements.txt
```

1. Place the dataset in `data/` and update the file path in the first cells of the notebook (the original uses Google Drive paths).
2. The LLM sentiment cells are commented out because they call the Anthropic API (about 8,000 requests). To skip them, load the precomputed scores from `data/df_with_sentiment.pkl` and continue from the modeling section.
3. To regenerate the scores, set `ANTHROPIC_API_KEY` as an environment variable and uncomment the relevant cells.

## Limitations

- The test set was also used as the evaluation set for early stopping and for comparing tuned models, so the reported metrics may be slightly optimistic. A separate validation split would be a cleaner setup.
- The dataset appears to be synthetic or course-provided; results may not generalize to real customer data.
- LLM sentiment was validated on only 30 manually labeled reviews.

## Tech stack

Python, pandas, NumPy, SciPy, scikit-learn, CatBoost, SHAP, seaborn, matplotlib, GeoPandas, VADER, Anthropic API.
