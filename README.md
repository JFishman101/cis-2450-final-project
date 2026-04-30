# Predicting Song Genre, Danceability, and Energy from Lyrics + Audio Features

**CIS 2450 — Big Data Analytics, Spring 2026**
**Team:** David Jorge-Bates & Jonah Fishman
**Professor:** Zachary G. Ives

## What this project does

Given a song's lyrics, Spotify audio features, and Deezer metadata, predict:
- **Genre** (10-way classification: Rock, Pop, Electronic, Folk, Country, Hip-Hop, R&B, Blues, Jazz, Classical)
- **Danceability** (regression, 0–1)
- **Energy** (regression, 0–1)

The corpus is ~434K songs from Kaggle's 550K+ Spotify Songs dataset, inner-joined to the Deezer API by artist + title. Genre is heavily imbalanced (Rock 37% → Classical 1.9%, a 19.4× ratio), so we use macro F1 as the primary metric.

## Headline results

| Task | Best model | Test score |
|---|---|---|
| Genre classification | Logistic Regression (multinomial, L2, C=1.0) | 89.69% accuracy / **87.12% macro F1** |
| Energy regression | Ridge (alpha=1.0) | **R² = 0.823**, RMSE = 0.103 |
| Danceability regression | Ridge (alpha=1.0) | **R² = 0.571**, RMSE = 0.113 |

Linear models won this problem — Random Forest tuned hit 82.5% accuracy, AdaBoost 57%. The 521-feature space we engineered is linearly separable enough that L2-regularized boundaries beat tree splits.

## Repository contents

```
.
├── README.md                 ← this file
├── extract_data.ipynb        ← ETL pipeline (Please see warning inside notebook)
├── genre_prediction.ipynb    ← main analysis notebook (Sections 1–9)
├── dashboard.py              ← interactive Plotly Dash dashboard
├── data/
│   └── song_data.db          ← DuckDB file (~1 GB), table `songs`, 433,618 rows
└── models/                   ← fitted preprocessors + trained models (joblib)
    ├── scaler.joblib
    ├── tfidf.joblib
    ├── svd_scan.joblib
    ├── kmeans.joblib
    ├── ohe_cluster.joblib
    ├── preprocessing_meta.joblib
    ├── lr_baseline.joblib
    ├── lr_tuned.joblib
    ├── rf_model.joblib
    ├── rf_tuned.joblib
    ├── dt_model.joblib
    ├── ab_model.joblib
    ├── ridge_danceability.joblib
    ├── ridge_energy.joblib
    ├── rfr_danceability.joblib
    └── rfr_energy.joblib
```

### Notebook and dashboard roles

The repo has three main entry points and they are run in different ways:

- **`extract_data.ipynb`** — the ETL pipeline. **Please read the warning inside the notebook before running.** It documents how `data/song_data.db` was originally built from Kaggle's 550K+ Spotify Songs CSV plus enrichment from the Deezer API. The Deezer API loop in Step 2c takes several days to complete; the database is already provided in `data/`, so this notebook is included only as a record of the data pipeline.
- **`genre_prediction.ipynb`** — the analysis notebook. Open this in VS Code and run cells top-to-bottom. This is what produces the trained models in `models/`.
- **`dashboard.py`** — the interactive dashboard. Run with `python dashboard.py` from the project root. It loads the artifacts produced by the analysis notebook and serves an interactive web UI at http://127.0.0.1:8050.

## How to run

### 1. Set up the environment

Python 3.12 in a virtual environment. From the project root:

```bash
python -m venv venv
source venv/bin/activate          # on Windows/PowerShell: venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Required packages: `duckdb polars pandas numpy pyarrow scipy scikit-learn>=1.6 imbalanced-learn matplotlib seaborn ipykernel joblib psutil dash dash-bootstrap-components plotly`. See `requirements.txt` for pinned versions.

> **scikit-learn version note:** the saved models in `models/` were trained on scikit-learn 1.8.0. Loading them with a different version still works but will emit harmless `InconsistentVersionWarning`s. Installing from `requirements.txt` matches the training version exactly.

### 2. Run the notebook

Open `genre_prediction.ipynb` in VS Code (with the Jupyter extension installed) and select the project's `venv` as the kernel — it will appear in the kernel picker as "CIS 2450 (venv)" or "Python (venv)" depending on your setup. Run cells top-to-bottom. Sections 6 and 7 are compute-heavy:
- Section 6.6 (TF-IDF fit): 2–6 min
- Section 6.7 (TruncatedSVD scan): 2–4 min
- Section 6.11 (SMOTE): 1–3 min
- Section 7.2 (Random Forest fit): 4–8 min
- Section 7.5 (RandomizedSearchCV): 15–40 min

The notebook saves all fitted preprocessors and trained models into `models/` along the way, so the dashboard can load them directly.

### 3. Run the dashboard

```bash
python dashboard.py
```

Then open http://127.0.0.1:8050 in a browser. The first launch builds a `dashboard_cache.joblib` file (~3–7 min) by scoring every classifier and regressor on the held-out test set; subsequent launches load this cache and start in seconds.

**CLI flags:**
- `--rebuild-cache` — force rebuild after retraining a model
- `--port 8051` — use a non-default port
- `--host 0.0.0.0` — bind to all interfaces (default 127.0.0.1)
- `--debug` — enable Dash debug mode

To stop the server, press **Ctrl+C** in the terminal.

## What the dashboard shows

Four tabs:

1. **Project Overview & EDA** — class balance, ANOVA effect sizes (η²) per audio feature, audio feature boxplots by genre, Pearson correlation matrix, genre × decade heatmap (demonstrates a SQL JOIN against a derived `eras` table), top niche tags, per-tag genre composition.
2. **Model Comparison** — six classifiers compared side-by-side (accuracy, macro F1, weighted F1), per-model confusion matrices, Ridge vs RF Regressor on both regression targets, per-genre RMSE breakdown.
3. **What Predicts Each Genre** — RF feature importance top 25, importance aggregated by feature block (audio / LSA lyrics / cluster / niche tag), per-genre LR coefficient inspection, ANOVA-vs-RF rank-agreement scatter.
4. **Try It Live** — paste lyrics, set the 10 audio sliders, optionally select niche tags from the top-200 list, click predict. Shows top-3 genre confidence with progress bars, the prediction from every loaded classifier, and Ridge regression predictions for danceability and energy (with the corresponding target column dropped from input to prevent trivial leakage).

## Pipeline summary

The 521-dimensional feature matrix `X_final` is laid out as:

| Slice | Source | Dimensions |
|---|---|---|
| `[0:11]` | RobustScaler-transformed numeric features (10 Spotify audio + lyric word count) | 11 |
| `[11:311]` | TruncatedSVD top 300 components on TF-IDF lyrics (max_features=10K, sublinear_tf) | 300 |
| `[311:321]` | K-Means cluster ID one-hot (k=10, fit on the 10 scaled audio features) | 10 |
| `[321:521]` | Top-200 niche tags one-hot (covers 88.7% of corpus) | 200 |

Pipeline rules followed throughout (rubric Section 5a):
- Train-test split (stratified 80/20 on genre) is the **first** transformation; nothing fits on test data.
- RobustScaler, TF-IDF, TruncatedSVD, K-Means, and SMOTE all `.fit()` on training only and `.transform()` both splits.
- Loudness winsorization at the 1st/99th percentile uses training-only thresholds.
- Hyperparameter tuning uses 3-fold CV on the training set; the test set is held out until final evaluation.

## Course topics demonstrated

Polars, SQL/DuckDB, Joins, Relational database, Supervised Learning, Text representations (TF-IDF), Dimensionality Reduction (TruncatedSVD), Hyperparameter Tuning (RandomizedSearchCV), Hypothesis Testing (ANOVA + chi-squared with effect-size analysis), Unsupervised Learning (K-Means cluster IDs fed back into the supervised pipeline as features).

## Limitations

- The 10-way genre label is itself a coarse aggregation; some confusions (Pop ↔ Electronic, R&B ↔ Pop) reflect genuine boundary-blur in the source labels.
- Spotify's audio features are themselves model outputs — we are predicting model outputs from model outputs. Raw audio embeddings (CLAP, MERT) would be a stronger setup but were outside scope.
- SMOTE interpolates linearly across all features including the 200 binary tag columns, producing fractional tag values that are mathematically valid for L2-LR but semantically odd. SMOTENC would be more principled.

## Reproducibility

`RANDOM_SEED = 42` throughout. Train/test split, SMOTE, K-Means, RandomizedSearchCV, and all model fits use this seed. The full pipeline reproduces from a fresh kernel by running the notebook top-to-bottom.

## Credits

- Upstream ETL pipeline (Kaggle ⨝ Deezer record linking, schema cleaning) — David Jorge-Bates
- Modeling notebook + dashboard — Jonah Fishman
- Both team members collaborated on the dashboard structure and final presentation.
