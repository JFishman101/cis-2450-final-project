"""
CIS 2450 Final Project — Dashboard
Predicting Song Genre, Danceability, and Energy from Lyrics + Audio Features

Team:    David Jorge-Bates & Jonah Fishman
Course:  CIS 2450 — Big Data Analytics, Spring 2026
Run:     python dashboard.py   (then open http://127.0.0.1:8050)

Expects, relative to this file:
    data/song_data.db          # DuckDB file with `songs` table
    models/*.joblib            # Preprocessors + trained models from the notebook

First run builds dashboard_cache.joblib (~3-7 min on the full 433K corpus).
Subsequent runs load the cache and start in seconds.
Pass --rebuild-cache to force a fresh build.
"""

# ============================================================================
# Imports
# ============================================================================
import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

import dash
from dash import Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration — paths, constants, colors
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "song_data.db"
MODELS_DIR = BASE_DIR / "models"
CACHE_PATH = BASE_DIR / "dashboard_cache.joblib"

SONGS_TABLE = "songs"
RANDOM_SEED = 42
TEST_SIZE = 0.20
EDA_SAMPLE_SIZE = 50_000

# Feature column order — must match notebook Section 6 exactly.
NUMERIC_AUDIO_FEATURES = [
    "danceability", "energy", "valence", "tempo", "loudness",
    "acousticness", "speechiness", "instrumentalness", "liveness", "popularity",
]
NUMERIC_FEATURES = NUMERIC_AUDIO_FEATURES + ["lyric_word_count"]

# Columns we pull from the DB for modeling.
DB_COLUMNS = NUMERIC_AUDIO_FEATURES + [
    "_id", "song_name", "artists_list", "lyrics", "genre", "niche_genres",
    "release_date",
]

# Per-class genre colors used everywhere a chart shows genre.
GENRE_COLORS = {
    "Rock":       "#E74C3C",
    "Pop":        "#9B59B6",
    "Electronic": "#3498DB",
    "Folk":       "#27AE60",
    "Country":    "#F39C12",
    "Hip-Hop":    "#1ABC9C",
    "R&B":        "#E67E22",
    "Blues":      "#34495E",
    "Jazz":       "#16A085",
    "Classical":  "#7F8C8D",
}

# Buckets for the 521-dim feature matrix (used by feature-importance plots).
BUCKET_COLORS = {
    "Numeric audio (11)":   "#1f77b4",
    "LSA lyrics (300)":     "#ff7f0e",
    "K-Means cluster (10)": "#2ca02c",
    "Niche tag (200)":      "#d62728",
}

# Default slider values for Try-It-Live (training medians, ish).
DEFAULT_AUDIO_VALUES = {
    "danceability":     (0.0, 1.0,   0.55, 0.01),
    "energy":           (0.0, 1.0,   0.65, 0.01),
    "valence":          (0.0, 1.0,   0.50, 0.01),
    "tempo":            (40,  220,   120,  1),
    "loudness":         (-30, 0,     -7.0, 0.1),
    "acousticness":     (0.0, 1.0,   0.20, 0.01),
    "speechiness":      (0.0, 1.0,   0.05, 0.01),
    "instrumentalness": (0.0, 1.0,   0.0,  0.01),
    "liveness":         (0.0, 1.0,   0.13, 0.01),
    "popularity":       (0,   100,   30,   1),
}

# Friendly labels for the audio-feature dropdowns.
FEATURE_LABEL = {
    "danceability":     "Danceability",
    "energy":           "Energy",
    "valence":          "Valence (positivity)",
    "tempo":            "Tempo (BPM)",
    "loudness":         "Loudness (dB)",
    "acousticness":     "Acousticness",
    "speechiness":      "Speechiness",
    "instrumentalness": "Instrumentalness",
    "liveness":         "Liveness",
    "popularity":       "Popularity (0-100)",
    "lyric_word_count": "Lyric word count",
}

# Classifier files in display order. Multiple candidates allow rename tolerance.
CLASSIFIER_FILES = [
    ("LR (baseline)",  ["lr_baseline.joblib"]),
    ("LR (tuned)",     ["lr_tuned.joblib"]),
    ("RF (baseline)",  ["rf_model.joblib", "rf_baseline.joblib"]),
    ("RF (tuned)",     ["rf_tuned.joblib"]),
    ("Decision Tree",  ["dt_model.joblib", "dt.joblib"]),
    ("AdaBoost",       ["ab_model.joblib", "ab.joblib"]),
]

# Regressors: (target, model_label) -> filename candidates
REGRESSOR_FILES = {
    ("danceability", "Ridge"):        ["ridge_danceability.joblib", "ridge_dance.joblib"],
    ("danceability", "RF Regressor"): ["rfr_danceability.joblib",   "rfr_dance.joblib"],
    ("energy",       "Ridge"):        ["ridge_energy.joblib"],
    ("energy",       "RF Regressor"): ["rfr_energy.joblib"],
}

PREPROCESSOR_FILES = {
    "scaler":      "scaler.joblib",
    "tfidf":       "tfidf.joblib",
    "svd_scan":    "svd_scan.joblib",
    "kmeans":      "kmeans.joblib",
    "ohe_cluster": "ohe_cluster.joblib",
    "meta":        "preprocessing_meta.joblib",
}


# ============================================================================
# Loading utilities
# ============================================================================
def _find_first(candidates):
    """Return the first existing path from a list of candidate filenames in models/."""
    for name in candidates:
        p = MODELS_DIR / name
        if p.exists():
            return p
    return None


def load_preprocessors():
    """Load the 5 fitted preprocessors + the metadata dict.

    Returns a dict keyed: scaler, tfidf, svd_scan, kmeans, ohe_cluster, meta.
    Raises with a helpful message if anything is missing.
    """
    missing = []
    out = {}
    for key, fname in PREPROCESSOR_FILES.items():
        path = MODELS_DIR / fname
        if not path.exists():
            missing.append(str(path))
            continue
        out[key] = joblib.load(path)

    if missing:
        raise FileNotFoundError(
            "Missing required preprocessor file(s):\n  "
            + "\n  ".join(missing)
            + f"\nRun the notebook through Section 7.0 to regenerate, "
              f"or check that {MODELS_DIR} contains them."
        )
    return out


def load_classifiers():
    """Load every classifier we can find. Skips missing models with a printed warning.

    Returns: list of (display_name, model) tuples in CLASSIFIER_FILES order.
    """
    loaded = []
    for display, candidates in CLASSIFIER_FILES:
        path = _find_first(candidates)
        if path is None:
            print(f"  [skip] {display}: none of {candidates} found in {MODELS_DIR}")
            continue
        loaded.append((display, joblib.load(path)))
        print(f"  [ok]   {display}: {path.name}")
    if not loaded:
        raise FileNotFoundError(
            "No classifier models found in " + str(MODELS_DIR)
        )
    return loaded


def load_regressors():
    """Load every regressor we can find. Returns dict keyed (target, label) -> model."""
    loaded = {}
    for key, candidates in REGRESSOR_FILES.items():
        path = _find_first(candidates)
        if path is None:
            print(f"  [skip] {key}: none of {candidates} found")
            continue
        loaded[key] = joblib.load(path)
        print(f"  [ok]   {key}: {path.name}")
    return loaded


def pick_primary_classifier(classifiers):
    """Choose which classifier drives the live-predict tab.

    Notebook Section 9 found LR (baseline) was the best test model
    (87.12% macro F1, marginally above LR tuned). Prefer in that order;
    fall back to whatever else loaded.
    """
    by_name = {name: m for name, m in classifiers}
    for preferred in ["LR (baseline)", "LR (tuned)", "RF (tuned)", "RF (baseline)"]:
        if preferred in by_name:
            return preferred, by_name[preferred]
    name, m = classifiers[0]
    return name, m


# ============================================================================
# Feature pipeline — must match notebook Section 6 exactly
# ============================================================================
def _make_word_count(lyrics_series):
    """Replicate `pl.col('lyrics').str.split(' ').list.len()` from notebook 6.1."""
    return lyrics_series.fillna("").astype(str).str.split(" ").str.len()


def parse_niche_tags(raw):
    """Parse a niche_genres column value (JSON string) into a list. Notebook 5.7."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or raw == "":
        return []
    if isinstance(raw, list):
        return raw
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def build_features_for_dataframe(df, preprocessors):
    """Apply the full preprocessing pipeline to a pandas DataFrame.

    Pipeline (notebook Sections 6.4 - 6.10):
      1. Clip loudness to [low, high] from training stats
      2. RobustScaler.transform on 11 numeric features
      3. TF-IDF.transform on lyrics
      4. SVD.transform; take first n_svd cols
      5. KMeans.predict on the 10 scaled audio features (drop lyric_word_count)
      6. OneHotEncoder.transform on cluster ids
      7. Manual top-200 niche-tag one-hot encode
      8. Concatenate -> dense float32 (n_rows, 521)

    Expects df to have columns: niche_tags (already parsed list) OR niche_genres,
    lyrics, plus all 10 audio features.
    """
    meta = preprocessors["meta"]
    scaler = preprocessors["scaler"]
    tfidf = preprocessors["tfidf"]
    svd = preprocessors["svd_scan"]
    kmeans = preprocessors["kmeans"]
    ohe = preprocessors["ohe_cluster"]

    df = df.copy()
    df["lyrics"] = df["lyrics"].fillna("").astype(str)

    # 1. Loudness winsorization using saved training thresholds
    loud_low = float(meta.get("loudness_low", df["loudness"].min()))
    loud_high = float(meta.get("loudness_high", df["loudness"].max()))
    df["loudness"] = df["loudness"].clip(lower=loud_low, upper=loud_high)

    # Engineer lyric_word_count
    df["lyric_word_count"] = _make_word_count(df["lyrics"])

    # 2. Scale 11 numeric features (in stable order)
    numeric_block = scaler.transform(df[NUMERIC_FEATURES].astype(np.float64))  # (n,11)

    # 3-4. TF-IDF -> LSA
    n_svd = int(meta.get("n_svd_components", 300))
    tfidf_mat = tfidf.transform(df["lyrics"].tolist())
    lsa_block = svd.transform(tfidf_mat)[:, :n_svd]  # (n, 300)

    # 5. K-Means cluster on the 10 scaled audio features only (drop lyric_word_count)
    audio_only_scaled = numeric_block[:, :len(NUMERIC_AUDIO_FEATURES)]
    cluster_ids = kmeans.predict(audio_only_scaled)

    # 6. One-hot encode cluster ids
    cluster_block = ohe.transform(cluster_ids.reshape(-1, 1))
    if hasattr(cluster_block, "toarray"):
        cluster_block = cluster_block.toarray()  # (n, 10)

    # 7. Niche tags -> top-200 one-hot
    top_tags = list(meta["top_tags"])
    tag_to_idx = {t: i for i, t in enumerate(top_tags)}
    n_tags = len(top_tags)

    if "niche_tags" in df.columns:
        tag_lists = df["niche_tags"].tolist()
    elif "niche_genres" in df.columns:
        tag_lists = [parse_niche_tags(v) for v in df["niche_genres"].tolist()]
    else:
        tag_lists = [[] for _ in range(len(df))]

    tag_block = np.zeros((len(df), n_tags), dtype=np.float32)
    for i, tags in enumerate(tag_lists):
        if not tags:
            continue
        for t in tags:
            j = tag_to_idx.get(t)
            if j is not None:
                tag_block[i, j] = 1.0

    # 8. Concatenate
    X = np.hstack([
        numeric_block.astype(np.float32),
        lsa_block.astype(np.float32),
        cluster_block.astype(np.float32),
        tag_block,
    ])
    return X


def build_features_live(lyrics, audio_dict, selected_tags, preprocessors):
    """Single-row variant for the Try-It-Live tab. Returns (1, 521) float32."""
    row = {**audio_dict, "lyrics": lyrics or ""}
    df = pd.DataFrame([row])
    if selected_tags:
        df["niche_tags"] = [list(selected_tags)]
    else:
        df["niche_tags"] = [[]]
    return build_features_for_dataframe(df, preprocessors)


# ============================================================================
# Cache builder — runs once on first launch, cached to disk
# ============================================================================
def _effect_label(eta2):
    if eta2 >= 0.14:
        return "large"
    if eta2 >= 0.06:
        return "medium"
    if eta2 >= 0.01:
        return "small"
    return "negligible"


def _feature_bucket(name):
    if name.startswith("lsa_"):
        return "LSA lyrics (300)"
    if name.startswith("cluster_"):
        return "K-Means cluster (10)"
    if name.startswith("tag_"):
        return "Niche tag (200)"
    return "Numeric audio (11)"


def build_cache(preprocessors, classifiers, regressors):
    """Compute everything the dashboard needs once, return a serializable dict.

    Heavy steps run here so the dashboard itself only ever reads from the dict.
    """
    print("\n" + "=" * 68)
    print("Building dashboard cache (one-time, ~3-7 min on full corpus)")
    print("=" * 68)

    cache = {}
    meta = preprocessors["meta"]

    # ------------------------------------------------------------------
    # Step 1/8: Load full songs table from DuckDB
    # ------------------------------------------------------------------
    print("\n[1/8] Loading songs from DuckDB...")
    t0 = time.time()
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database file not found: {DB_PATH}\n"
            "Place song_data.db inside data/ next to this dashboard.py."
        )
    cols_csv = ", ".join(DB_COLUMNS)
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        df = conn.execute(
            f"SELECT {cols_csv} FROM {SONGS_TABLE}"
        ).fetch_df()
    print(f"      Loaded {len(df):,} rows in {time.time() - t0:.1f}s")

    cache["n_songs_total"] = int(len(df))

    # ------------------------------------------------------------------
    # Step 2/8: EDA aggregates
    # ------------------------------------------------------------------
    print("[2/8] Computing EDA aggregates...")
    t0 = time.time()

    # Genre counts
    genre_counts = (
        df["genre"].value_counts()
        .rename_axis("genre").reset_index(name="count")
    )
    genre_counts["pct"] = (genre_counts["count"] / len(df) * 100).round(2)
    cache["genre_counts"] = genre_counts
    cache["genre_order"] = genre_counts["genre"].tolist()

    # Audio feature sample (50K rows for plotting)
    sample = df.sample(n=min(EDA_SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED)
    cache["audio_sample"] = sample[
        ["genre"] + NUMERIC_AUDIO_FEATURES
    ].reset_index(drop=True)

    # Audio feature summary stats
    cache["audio_summary"] = (
        df[NUMERIC_AUDIO_FEATURES].describe().round(3)
    )

    # ANOVA per audio feature with eta-squared
    anova_rows = []
    genres_in_order = cache["genre_order"]
    for feat in NUMERIC_AUDIO_FEATURES:
        groups = [
            df.loc[df["genre"] == g, feat].dropna().to_numpy()
            for g in genres_in_order
        ]
        groups = [g for g in groups if len(g) > 1]
        f_stat, p_value = f_oneway(*groups)
        all_vals = np.concatenate(groups)
        grand_mean = all_vals.mean()
        ss_total = float(((all_vals - grand_mean) ** 2).sum())
        ss_between = float(sum(
            len(g) * (g.mean() - grand_mean) ** 2
            for g in groups
        ))
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
        anova_rows.append({
            "feature":     feat,
            "F_statistic": float(f_stat),
            "p_value":     float(p_value),
            "eta_squared": eta_sq,
            "effect_size": _effect_label(eta_sq),
        })
    anova_df = pd.DataFrame(anova_rows).sort_values("eta_squared", ascending=False)
    cache["anova"] = anova_df.reset_index(drop=True)

    # Correlation matrix
    cache["corr_matrix"] = df[NUMERIC_AUDIO_FEATURES].corr().round(3)
    print(f"      EDA aggregates done in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 3/8: Niche tag stats
    # ------------------------------------------------------------------
    print("[3/8] Parsing niche tags + per-genre composition...")
    t0 = time.time()
    parsed_tags = df["niche_genres"].apply(parse_niche_tags)
    tag_counts = {}
    tag_genre_counts = {}
    for tags, g in zip(parsed_tags, df["genre"]):
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
            if t not in tag_genre_counts:
                tag_genre_counts[t] = {}
            tag_genre_counts[t][g] = tag_genre_counts[t].get(g, 0) + 1

    tag_counts_df = (
        pd.DataFrame(
            [{"tag": k, "count": v} for k, v in tag_counts.items()]
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    cache["tag_counts"] = tag_counts_df.head(50).reset_index(drop=True)
    cache["tag_genre_composition"] = tag_genre_counts
    cache["top_50_tags"] = tag_counts_df.head(50)["tag"].tolist()
    cache["top_200_tags"] = list(meta.get("top_tags", []))
    cache["n_zero_tag_songs"] = int((parsed_tags.apply(len) == 0).sum())
    cache["n_unique_tags"] = int(len(tag_counts))
    print(f"      Parsed {cache['n_unique_tags']:,} unique tags in "
          f"{time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 4/8: Genre x Era SQL join
    # ------------------------------------------------------------------
    print("[4/8] Computing genre x era SQL join...")
    t0 = time.time()
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        conn.execute("""
            CREATE TEMP TABLE eras AS
            SELECT decade, era_label FROM (VALUES
                (1900, '1900s'), (1910, '1910s'), (1920, '1920s'),
                (1930, '1930s'), (1940, '1940s'), (1950, '1950s'),
                (1960, '1960s'), (1970, '1970s'), (1980, '1980s'),
                (1990, '1990s'), (2000, '2000s'), (2010, '2010s'),
                (2020, '2020s')
            ) AS t(decade, era_label)
        """)
        join_df = conn.execute(f"""
            WITH parsed AS (
                SELECT s.*,
                       TRY_CAST(SUBSTR(s.release_date, 1, 4) AS INTEGER) AS year_int
                FROM {SONGS_TABLE} s
            )
            SELECT
                p.genre,
                e.era_label,
                COUNT(*) AS n_songs
            FROM parsed p
            JOIN eras e
              ON CAST(FLOOR(p.year_int / 10.0) * 10 AS INTEGER) = e.decade
            WHERE p.year_int BETWEEN 1900 AND 2026
            GROUP BY p.genre, e.era_label
            ORDER BY p.genre, e.era_label
        """).fetch_df()
    cache["genre_era"] = join_df
    print(f"      Genre x era join done in {time.time() - t0:.1f}s "
          f"({len(join_df)} cells)")

    # ------------------------------------------------------------------
    # Step 5/8: Reproduce train/test split (same seed) + build X_test_final
    # ------------------------------------------------------------------
    print("[5/8] Rebuilding test set features (~1-2 min)...")
    t0 = time.time()

    # Add niche_tags column to df for the feature builder
    df["niche_tags"] = parsed_tags
    cols_for_split = (
        NUMERIC_AUDIO_FEATURES
        + ["lyrics", "niche_tags", "genre", "_id"]
    )
    data = df[cols_for_split].reset_index(drop=True)

    train_data, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        stratify=data["genre"],
        random_state=RANDOM_SEED,
    )
    test_data = test_data.reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    # Free df early — we only need the splits past this point
    del df, parsed_tags, data
    gc.collect()

    print(f"      Building X_test ({len(test_data):,} rows)...")
    X_test = build_features_for_dataframe(test_data, preprocessors)
    y_test_genre = test_data["genre"].to_numpy()
    y_test_dance = test_data["danceability"].to_numpy()
    y_test_energy = test_data["energy"].to_numpy()

    cache["n_train"] = int(len(train_data))
    cache["n_test"] = int(len(test_data))
    cache["X_test_shape"] = X_test.shape

    print(f"      Test features built: {X_test.shape} in "
          f"{time.time() - t0:.1f}s")

    # We do NOT save X_test in the cache (would balloon to ~300 MB).
    # Instead we save predictions + metrics computed below.

    # ------------------------------------------------------------------
    # Step 6/8: Score every classifier on the test set
    # ------------------------------------------------------------------
    print("[6/8] Scoring classifiers on test set...")
    classifier_metrics = []
    test_predictions = {}
    confusion_matrices = {}

    for name, clf in classifiers:
        t0 = time.time()
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test_genre, y_pred)
        f1m = f1_score(y_test_genre, y_pred, average="macro")
        f1w = f1_score(y_test_genre, y_pred, average="weighted")
        cm = confusion_matrix(
            y_test_genre, y_pred, labels=cache["genre_order"]
        )

        classifier_metrics.append({
            "model":       name,
            "accuracy":    float(acc),
            "macro_f1":    float(f1m),
            "weighted_f1": float(f1w),
            "score_time_s": time.time() - t0,
        })
        test_predictions[name] = y_pred
        confusion_matrices[name] = cm
        print(f"      {name:<18s}  acc={acc*100:5.2f}%  "
              f"macroF1={f1m*100:5.2f}%  ({time.time()-t0:4.1f}s)")

    cache["classifier_metrics"] = pd.DataFrame(classifier_metrics)
    cache["confusion_matrices"] = confusion_matrices
    cache["y_test_genre"] = y_test_genre

    # ------------------------------------------------------------------
    # Step 7/8: Score regressors (drop target column to prevent leakage)
    # ------------------------------------------------------------------
    print("[7/8] Scoring regressors on test set...")
    regression_metrics = []
    per_genre_rmse = {}  # target -> DataFrame
    DANCE_IDX = NUMERIC_FEATURES.index("danceability")
    ENERGY_IDX = NUMERIC_FEATURES.index("energy")
    target_col_idx = {"danceability": DANCE_IDX, "energy": ENERGY_IDX}
    y_test_targets = {"danceability": y_test_dance, "energy": y_test_energy}

    for (target, label), reg in regressors.items():
        t0 = time.time()
        col_idx = target_col_idx[target]
        keep = [i for i in range(X_test.shape[1]) if i != col_idx]
        X_test_reg = X_test[:, keep]

        y_pred = reg.predict(X_test_reg)
        y_true = y_test_targets[target]

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        regression_metrics.append({
            "target": target,
            "model":  label,
            "RMSE":   rmse,
            "MAE":    mae,
            "R2":     r2,
        })
        print(f"      {target:<14s} {label:<14s}  "
              f"R2={r2:.3f}  RMSE={rmse:.4f}  ({time.time()-t0:.1f}s)")

        # Per-genre RMSE for this (target, model)
        rows = []
        for g in cache["genre_order"]:
            mask = (y_test_genre == g)
            if mask.sum() == 0:
                continue
            rmse_g = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
            rows.append({
                "genre":  g,
                "n_test": int(mask.sum()),
                "rmse":   round(rmse_g, 4),
            })
        per_genre_rmse[(target, label)] = pd.DataFrame(rows)

    cache["regression_metrics"] = pd.DataFrame(regression_metrics)
    cache["per_genre_rmse"] = per_genre_rmse

    # ------------------------------------------------------------------
    # Step 8/8: Feature importance + LR coefficients (for Tab 3)
    # ------------------------------------------------------------------
    print("[8/8] Extracting feature importance + LR coefficients...")
    feature_names = list(meta.get("feature_names", []))
    if not feature_names:
        # Fallback: rebuild from constants
        n_svd = int(meta.get("n_svd_components", 300))
        n_clusters = int(meta.get("n_clusters", 10))
        top_tags = list(meta.get("top_tags", []))
        feature_names = (
            NUMERIC_FEATURES
            + [f"lsa_{i+1}" for i in range(n_svd)]
            + [f"cluster_{i}" for i in range(n_clusters)]
            + [f"tag_{t}" for t in top_tags]
        )
    cache["feature_names"] = feature_names

    classifiers_dict = {n: m for n, m in classifiers}

    # RF feature importance (prefer tuned)
    rf_for_imp = classifiers_dict.get(
        "RF (tuned)", classifiers_dict.get("RF (baseline)")
    )
    if rf_for_imp is not None and hasattr(rf_for_imp, "feature_importances_"):
        importances = rf_for_imp.feature_importances_
        if len(importances) == len(feature_names):
            fi_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances,
            }).sort_values("importance", ascending=False).reset_index(drop=True)
            fi_df["bucket"] = fi_df["feature"].apply(_feature_bucket)
            cache["rf_feature_importance"] = fi_df

            bucket_agg = (
                fi_df.groupby("bucket")
                .agg(total_importance=("importance", "sum"),
                     n_features=("feature", "count"),
                     avg_per_feature=("importance", "mean"))
                .sort_values("total_importance", ascending=False)
                .reset_index()
            )
            cache["bucket_importance"] = bucket_agg
            print(f"      RF importances: {len(importances)} features")
        else:
            print(f"      [warn] RF importance length mismatch: "
                  f"{len(importances)} vs {len(feature_names)} feature names")

    # LR coefficients per genre (prefer baseline since it was the best test model)
    lr_for_coef = classifiers_dict.get(
        "LR (baseline)", classifiers_dict.get("LR (tuned)")
    )
    if lr_for_coef is not None and hasattr(lr_for_coef, "coef_"):
        coef = lr_for_coef.coef_  # shape (n_classes, n_features)
        if coef.shape[1] == len(feature_names):
            cache["lr_coef"] = coef
            cache["lr_classes"] = list(lr_for_coef.classes_)
            print(f"      LR coefficients: {coef.shape}")
        else:
            print(f"      [warn] LR coef shape mismatch: "
                  f"{coef.shape} vs {len(feature_names)} feature names")

    # ANOVA-vs-RF rank comparison (audio features only)
    if "rf_feature_importance" in cache:
        fi_df = cache["rf_feature_importance"]
        audio_fi = fi_df[fi_df["feature"].isin(NUMERIC_AUDIO_FEATURES)].copy()
        audio_fi["rf_rank"] = audio_fi["importance"].rank(ascending=False).astype(int)
        anova_rank = anova_df.copy()
        anova_rank["anova_rank"] = (
            anova_rank["eta_squared"].rank(ascending=False).astype(int)
        )
        rank_compare = audio_fi.merge(
            anova_rank[["feature", "anova_rank", "eta_squared", "effect_size"]],
            on="feature",
        ).sort_values("rf_rank").reset_index(drop=True)
        cache["anova_rf_compare"] = rank_compare

    print("\nCache built. Saving to disk...")
    return cache


def load_or_build_cache(rebuild=False):
    """Load cache from disk if it exists and is fresh, else build it."""
    if not rebuild and CACHE_PATH.exists():
        print(f"Loading cache from {CACHE_PATH}...")
        t0 = time.time()
        cache_data = joblib.load(CACHE_PATH)
        print(f"  Loaded in {time.time() - t0:.1f}s")
        return cache_data["cache"], cache_data["preprocessors"], \
               cache_data["classifiers"], cache_data["regressors"]

    # Fresh build
    print("Loading preprocessors from", MODELS_DIR, "...")
    preprocessors = load_preprocessors()
    print("  ok\n")

    print("Loading classifiers...")
    classifiers = load_classifiers()
    print()

    print("Loading regressors...")
    regressors = load_regressors()
    print()

    cache = build_cache(preprocessors, classifiers, regressors)

    print(f"\nSaving cache to {CACHE_PATH}...")
    t0 = time.time()
    joblib.dump({
        "cache": cache,
        "preprocessors": preprocessors,
        "classifiers": classifiers,
        "regressors": regressors,
    }, CACHE_PATH, compress=3)
    print(f"  Saved in {time.time() - t0:.1f}s\n")

    return cache, preprocessors, classifiers, regressors


# ============================================================================
# Plotly figure helpers
# ============================================================================
PLOTLY_TEMPLATE = "plotly_white"


def _genre_color_list(genres):
    return [GENRE_COLORS.get(g, "#888888") for g in genres]


def fig_class_balance(genre_counts):
    """Bar chart of song counts per genre, colored by genre."""
    fig = go.Figure()
    fig.add_bar(
        x=genre_counts["genre"],
        y=genre_counts["count"],
        marker_color=_genre_color_list(genre_counts["genre"]),
        text=[f"{c:,}<br>({p:.1f}%)"
              for c, p in zip(genre_counts["count"], genre_counts["pct"])],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Songs: %{y:,}<extra></extra>",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Songs per Genre — 19.4x Imbalance Drives Macro-F1 as Primary Metric",
        xaxis_title="",
        yaxis_title="Number of songs",
        height=440,
        margin=dict(t=60, l=50, r=20, b=50),
        showlegend=False,
    )
    return fig


def fig_anova(anova_df):
    """Bar chart of eta-squared per feature, colored by effect size."""
    color_map = {
        "large": "#E74C3C", "medium": "#F39C12",
        "small": "#F1C40F", "negligible": "#BDC3C7",
    }
    colors = [color_map[lbl] for lbl in anova_df["effect_size"]]
    fig = go.Figure()
    fig.add_bar(
        x=anova_df["feature"],
        y=anova_df["eta_squared"],
        marker_color=colors,
        text=[f"η²={v:.3f}<br>{lbl}"
              for v, lbl in zip(anova_df["eta_squared"], anova_df["effect_size"])],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>η² = %{y:.3f}<extra></extra>",
    )
    fig.add_hline(y=0.14, line_dash="dash", line_color="#E74C3C",
                  annotation_text="Large (0.14)", annotation_position="right")
    fig.add_hline(y=0.06, line_dash="dash", line_color="#F39C12",
                  annotation_text="Medium (0.06)", annotation_position="right")
    fig.add_hline(y=0.01, line_dash="dash", line_color="#F1C40F",
                  annotation_text="Small (0.01)", annotation_position="right")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="ANOVA Effect Sizes (η²) — How Much Variance Genre Explains",
        xaxis_title="",
        yaxis_title="η² (proportion of variance)",
        height=460,
        margin=dict(t=60, l=60, r=80, b=60),
    )
    return fig


def fig_audio_boxplot(audio_sample, feature, genre_order):
    """Boxplot of one audio feature across genres, ordered by median."""
    medians = (
        audio_sample.groupby("genre")[feature].median()
        .sort_values(ascending=False)
    )
    ordered = [g for g in medians.index if g in genre_order]

    fig = go.Figure()
    for g in ordered:
        vals = audio_sample.loc[audio_sample["genre"] == g, feature].dropna()
        fig.add_box(
            y=vals,
            name=g,
            marker_color=GENRE_COLORS.get(g, "#888"),
            boxpoints=False,
            line=dict(width=1.4),
            hovertemplate=f"<b>{g}</b><br>" + feature + ": %{y:.3f}<extra></extra>",
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"{FEATURE_LABEL.get(feature, feature)} by Genre — Sorted by Median",
        yaxis_title=FEATURE_LABEL.get(feature, feature),
        xaxis_title="",
        height=480,
        showlegend=False,
        margin=dict(t=60, l=60, r=20, b=60),
    )
    return fig


def fig_correlation_heatmap(corr_matrix):
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu",
        zmid=0, zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(title="Pearson r"),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Pearson Correlation — energy/loudness/acousticness Cluster Visible",
        height=520,
        margin=dict(t=60, l=80, r=40, b=80),
    )
    return fig


def fig_top_tags(tag_counts, top_n=20):
    top = tag_counts.head(top_n).iloc[::-1]
    fig = go.Figure()
    fig.add_bar(
        x=top["count"],
        y=top["tag"],
        orientation="h",
        marker_color="#5DADE2",
        hovertemplate="<b>%{y}</b><br>%{x:,} songs<extra></extra>",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Top {top_n} Niche Genre Tags by Song Count",
        xaxis_title="Number of songs",
        yaxis_title="",
        height=520,
        margin=dict(t=60, l=180, r=40, b=50),
    )
    return fig


def fig_tag_genre_composition(tag_genre_counts, tag, genre_order):
    """Horizontal bar: % of a chosen tag's songs that fall under each genre."""
    if tag not in tag_genre_counts:
        return go.Figure().update_layout(
            title=f"No data for tag '{tag}'", template=PLOTLY_TEMPLATE,
        )
    counts = tag_genre_counts[tag]
    total = sum(counts.values())
    rows = [
        {"genre": g, "pct": (counts.get(g, 0) / total * 100) if total else 0,
         "n": counts.get(g, 0)}
        for g in genre_order
    ]
    df = pd.DataFrame(rows).sort_values("pct", ascending=True)
    fig = go.Figure()
    fig.add_bar(
        x=df["pct"],
        y=df["genre"],
        orientation="h",
        marker_color=[GENRE_COLORS.get(g, "#888") for g in df["genre"]],
        text=[f"{p:.1f}% (n={n:,})" for p, n in zip(df["pct"], df["n"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:.1f}% of '" + tag
                      + "' songs<extra></extra>",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Genre Composition of '{tag}' (n = {total:,} songs)",
        xaxis_title="% of tagged songs in this genre",
        yaxis_title="",
        height=460,
        margin=dict(t=60, l=110, r=80, b=50),
    )
    return fig


def fig_genre_era_heatmap(genre_era):
    """Heatmap showing songs per genre x decade. Demonstrates the SQL join."""
    eras = ["1900s", "1910s", "1920s", "1930s", "1940s", "1950s",
            "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
    pivot = (
        genre_era.pivot(index="genre", columns="era_label", values="n_songs")
        .reindex(columns=eras)
        .fillna(0).astype(int)
    )
    # Order genres by total counts descending
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="YlOrRd",
        text=pivot.values,
        texttemplate="%{text:,}",
        textfont=dict(size=9),
        hovertemplate="<b>%{y} - %{x}</b><br>%{z:,} songs<extra></extra>",
        colorbar=dict(title="Songs"),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Songs per Genre per Decade (SQL JOIN: songs ⨝ eras)",
        xaxis_title="Decade",
        yaxis_title="Genre",
        height=440,
        margin=dict(t=60, l=110, r=20, b=50),
    )
    return fig


def fig_classifier_comparison(metrics_df):
    """Grouped bar: accuracy + macro F1 + weighted F1 per model."""
    df = metrics_df.copy()
    df["accuracy"] = df["accuracy"] * 100
    df["macro_f1"] = df["macro_f1"] * 100
    df["weighted_f1"] = df["weighted_f1"] * 100
    df = df.sort_values("macro_f1", ascending=False)

    fig = go.Figure()
    fig.add_bar(
        name="Accuracy", x=df["model"], y=df["accuracy"],
        marker_color="#3498DB",
        text=[f"{v:.1f}%" for v in df["accuracy"]], textposition="outside",
    )
    fig.add_bar(
        name="Macro F1", x=df["model"], y=df["macro_f1"],
        marker_color="#E74C3C",
        text=[f"{v:.1f}%" for v in df["macro_f1"]], textposition="outside",
    )
    fig.add_bar(
        name="Weighted F1", x=df["model"], y=df["weighted_f1"],
        marker_color="#F39C12",
        text=[f"{v:.1f}%" for v in df["weighted_f1"]], textposition="outside",
    )
    # Reference lines: always-Rock and random
    fig.add_hline(y=37.1, line_dash="dot", line_color="#7F8C8D",
                  annotation_text="Always-Rock baseline (37.1%)",
                  annotation_position="right")
    fig.add_hline(y=10.0, line_dash="dot", line_color="#BDC3C7",
                  annotation_text="Random (10%)", annotation_position="right")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Classifier Test-Set Performance — All 6 Configurations",
        yaxis_title="Score (%)",
        xaxis_title="",
        barmode="group",
        height=500,
        yaxis_range=[0, 105],
        margin=dict(t=60, l=60, r=180, b=60),
    )
    return fig


def fig_confusion_matrix(cm, genre_order, model_name):
    """Row-normalized (recall) confusion matrix heatmap."""
    cm_norm = cm / cm.sum(axis=1, keepdims=True) * 100
    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=genre_order,
        y=genre_order,
        colorscale="Blues",
        zmin=0, zmax=100,
        text=cm_norm.round(0).astype(int),
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="<b>True: %{y} → Pred: %{x}</b><br>%{z:.1f}%<extra></extra>",
        colorbar=dict(title="% of true class"),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Confusion Matrix (Row-Normalized = Recall) — {model_name}",
        xaxis_title="Predicted genre",
        yaxis_title="True genre",
        height=540,
        margin=dict(t=60, l=80, r=40, b=80),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def fig_regression_comparison(reg_metrics):
    """Grouped bar of R^2 per (target, model)."""
    fig = go.Figure()
    targets = sorted(reg_metrics["target"].unique())
    for label in ["Ridge", "RF Regressor"]:
        subset = reg_metrics[reg_metrics["model"] == label]
        if subset.empty:
            continue
        ordered = subset.set_index("target").reindex(targets).reset_index()
        fig.add_bar(
            name=label,
            x=ordered["target"],
            y=ordered["R2"],
            marker_color="#2ECC71" if label == "Ridge" else "#E67E22",
            text=[f"R²={v:.3f}<br>RMSE={r:.4f}"
                  for v, r in zip(ordered["R2"], ordered["RMSE"])],
            textposition="outside",
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Regression Performance — Ridge vs RF Regressor",
        yaxis_title="R² (test set)",
        xaxis_title="Target",
        barmode="group",
        height=440,
        yaxis_range=[0, 1.0],
        margin=dict(t=60, l=60, r=40, b=60),
    )
    return fig


def fig_per_genre_rmse(per_genre_df, target):
    df = per_genre_df.sort_values("rmse")
    fig = go.Figure()
    fig.add_bar(
        x=df["rmse"],
        y=df["genre"],
        orientation="h",
        marker_color=[GENRE_COLORS.get(g, "#888") for g in df["genre"]],
        text=[f"{v:.4f} (n={n:,})" for v, n in zip(df["rmse"], df["n_test"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>RMSE = %{x:.4f}<extra></extra>",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Per-Genre RMSE — {target} (RF Regressor)",
        xaxis_title="RMSE",
        yaxis_title="",
        height=440,
        margin=dict(t=60, l=110, r=120, b=50),
    )
    return fig


def fig_rf_feature_importance(fi_df, top_n=25):
    top = fi_df.head(top_n).iloc[::-1].copy()
    colors = [BUCKET_COLORS.get(b, "#999") for b in top["bucket"]]
    fig = go.Figure()
    fig.add_bar(
        x=top["importance"],
        y=top["feature"],
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>importance = %{x:.4f}<br>"
                      "(bucket: " + top["bucket"].astype(str) + ")<extra></extra>",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Top {top_n} Features by Random Forest Importance",
        xaxis_title="Gini importance (averaged across trees)",
        yaxis_title="",
        height=620,
        margin=dict(t=60, l=180, r=40, b=50),
    )
    return fig


def fig_bucket_importance(bucket_agg):
    df = bucket_agg.sort_values("total_importance", ascending=True)
    colors = [BUCKET_COLORS.get(b, "#999") for b in df["bucket"]]
    fig = go.Figure()
    fig.add_bar(
        x=df["total_importance"],
        y=df["bucket"],
        orientation="h",
        marker_color=colors,
        text=[f"{v*100:.1f}%  ({n} features)"
              for v, n in zip(df["total_importance"], df["n_features"])],
        textposition="outside",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Total RF Importance by Feature Block",
        xaxis_title="Sum of importances (out of 1.0)",
        yaxis_title="",
        height=320,
        margin=dict(t=60, l=180, r=140, b=50),
    )
    return fig


def fig_lr_coefficients(coef_matrix, classes, feature_names, target_genre, top_n=10):
    """Show top positive + negative LR coefficients for a single genre."""
    if target_genre not in classes:
        return go.Figure().update_layout(title=f"Genre '{target_genre}' not in model")
    class_idx = list(classes).index(target_genre)
    coefs = coef_matrix[class_idx]
    order = np.argsort(coefs)
    top_neg_idx = order[:top_n]
    top_pos_idx = order[-top_n:][::-1]

    items = []
    for i in top_pos_idx:
        items.append((feature_names[i], float(coefs[i]), "positive"))
    for i in top_neg_idx:
        items.append((feature_names[i], float(coefs[i]), "negative"))
    df = pd.DataFrame(items, columns=["feature", "coef", "sign"])
    df = df.sort_values("coef", ascending=True)
    colors = ["#27AE60" if s == "positive" else "#C0392B" for s in df["sign"]]

    fig = go.Figure()
    fig.add_bar(
        x=df["coef"],
        y=df["feature"],
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>coef = %{x:.3f}<extra></extra>",
    )
    fig.add_vline(x=0, line_color="#7F8C8D", line_width=1)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Top ±{top_n} LR Coefficients for '{target_genre}' "
              f"(green = pushes toward this genre, red = away)",
        xaxis_title="Logistic regression coefficient (L2-penalized)",
        yaxis_title="",
        height=560,
        margin=dict(t=60, l=180, r=40, b=50),
    )
    return fig


def fig_anova_rf_agreement(rank_compare):
    """Scatter: ANOVA rank vs RF rank for the 10 audio features."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rank_compare["anova_rank"],
        y=rank_compare["rf_rank"],
        mode="markers+text",
        text=rank_compare["feature"],
        textposition="top center",
        marker=dict(
            size=14,
            color=rank_compare["eta_squared"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="η²"),
            line=dict(color="#2C3E50", width=1),
        ),
        hovertemplate="<b>%{text}</b><br>"
                      "ANOVA rank: %{x}<br>RF rank: %{y}<extra></extra>",
    ))
    n = len(rank_compare)
    fig.add_trace(go.Scatter(
        x=[1, n], y=[1, n],
        mode="lines",
        line=dict(dash="dash", color="#7F8C8D"),
        name="Perfect agreement",
        hoverinfo="skip",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="ANOVA vs Random Forest: Audio-Feature Importance Rankings",
        xaxis_title="ANOVA rank by η² (1 = strongest)",
        yaxis_title="RF rank by importance (1 = strongest)",
        height=480,
        showlegend=False,
        margin=dict(t=60, l=80, r=40, b=60),
    )
    fig.update_xaxes(autorange="reversed")
    fig.update_yaxes(autorange="reversed")
    return fig


# ============================================================================
# Live predict
# ============================================================================
def live_predict(lyrics, audio_dict, selected_tags,
                 preprocessors, classifiers, regressors,
                 primary_classifier_name):
    """Run the full feature pipeline + predict on user inputs.

    Returns a dict with predicted genre, top-3 confidence, all-classifier votes,
    danceability prediction, energy prediction.
    """
    X = build_features_live(lyrics, audio_dict, selected_tags, preprocessors)

    # Primary classifier prediction + top-3 confidence
    primary_clf = dict(classifiers).get(primary_classifier_name)
    if primary_clf is None:
        # Fallback to first available
        primary_classifier_name, primary_clf = classifiers[0]

    primary_pred = primary_clf.predict(X)[0]
    top3 = []
    if hasattr(primary_clf, "predict_proba"):
        proba = primary_clf.predict_proba(X)[0]
        order = np.argsort(proba)[::-1]
        top3 = [
            (str(primary_clf.classes_[i]), float(proba[i]))
            for i in order[:3]
        ]
    else:
        top3 = [(str(primary_pred), 1.0)]

    # All-classifier votes
    votes = []
    for name, clf in classifiers:
        try:
            v = clf.predict(X)[0]
        except Exception as e:
            v = f"(error: {e.__class__.__name__})"
        votes.append({"model": name, "predicted": str(v)})

    # Regression — drop target column
    DANCE_IDX = NUMERIC_FEATURES.index("danceability")
    ENERGY_IDX = NUMERIC_FEATURES.index("energy")
    keep_dance = [i for i in range(X.shape[1]) if i != DANCE_IDX]
    keep_energy = [i for i in range(X.shape[1]) if i != ENERGY_IDX]

    dance_pred = None
    energy_pred = None
    ridge_dance = regressors.get(("danceability", "Ridge"))
    ridge_energy = regressors.get(("energy", "Ridge"))
    if ridge_dance is not None:
        dance_pred = float(ridge_dance.predict(X[:, keep_dance])[0])
    if ridge_energy is not None:
        energy_pred = float(ridge_energy.predict(X[:, keep_energy])[0])

    return {
        "primary_classifier": primary_classifier_name,
        "predicted_genre": str(primary_pred),
        "top3": top3,
        "all_votes": votes,
        "dance_pred": dance_pred,
        "energy_pred": energy_pred,
    }


# ============================================================================
# App initialization (these are populated in main() before app.run)
# ============================================================================
CACHE = {}
PREPROCESSORS = {}
CLASSIFIERS = []
REGRESSORS = {}
PRIMARY_CLF_NAME = None

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME],
    title="CIS 2450 — Genre Prediction Dashboard",
    suppress_callback_exceptions=True,
)


# ============================================================================
# Layout
# ============================================================================
def _metric_card(label, value, sublabel=None, color="primary"):
    return dbc.Card(
        dbc.CardBody([
            html.Div(label, style={
                "fontSize": "0.85rem", "color": "#7F8C8D",
                "textTransform": "uppercase", "letterSpacing": "0.05em",
            }),
            html.Div(value, style={
                "fontSize": "1.9rem", "fontWeight": 700,
                "marginTop": "0.25rem", "lineHeight": 1.1,
                "color": "#2C3E50",
            }),
            html.Div(sublabel or "", style={
                "fontSize": "0.8rem", "color": "#95A5A6",
                "marginTop": "0.25rem",
            }) if sublabel else None,
        ]),
        className="shadow-sm",
        style={"border": "1px solid #E5E8EC"},
    )


def build_header():
    """Top banner + headline metrics."""
    metrics = CACHE.get("classifier_metrics")
    reg_metrics = CACHE.get("regression_metrics")

    if metrics is not None and not metrics.empty:
        best = metrics.sort_values("macro_f1", ascending=False).iloc[0]
        best_card_label = best["model"]
        best_card_val = f"{best['macro_f1']*100:.1f}%"
        best_sublabel = f"acc {best['accuracy']*100:.1f}% — held-out test"
    else:
        best_card_label = "(no models)"; best_card_val = "—"; best_sublabel = ""

    energy_r2 = "—"
    dance_r2 = "—"
    if reg_metrics is not None and not reg_metrics.empty:
        e = reg_metrics[(reg_metrics["target"] == "energy")
                        & (reg_metrics["model"] == "Ridge")]
        d = reg_metrics[(reg_metrics["target"] == "danceability")
                        & (reg_metrics["model"] == "Ridge")]
        if not e.empty:
            energy_r2 = f"{e.iloc[0]['R2']:.3f}"
        if not d.empty:
            dance_r2 = f"{d.iloc[0]['R2']:.3f}"

    n_total = CACHE.get("n_songs_total", 0)

    banner = dbc.Container([
        html.Div([
            html.Div([
                html.H2("Predicting Song Genre, Danceability, and Energy",
                        style={"fontWeight": 700, "marginBottom": "0.2rem",
                               "color": "#FFFFFF"}),
                html.Div(
                    "From lyrics, Spotify audio features, and Deezer metadata.  "
                    "CIS 2450 — Big Data Analytics — David Jorge-Bates & Jonah Fishman",
                    style={"opacity": 0.85, "fontSize": "0.95rem",
                           "color": "#FFFFFF"},
                ),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], fluid=True, style={
        "background": "linear-gradient(95deg, #2C3E50 0%, #4A6789 100%)",
        "padding": "1.4rem 2rem",
        "marginBottom": "1.2rem",
    })

    cards_row = dbc.Container([
        dbc.Row([
            dbc.Col(_metric_card("Best classifier (macro F1)",
                                 best_card_val,
                                 sublabel=f"{best_card_label} — {best_sublabel}"),
                    md=4),
            dbc.Col(_metric_card("Energy regression R²", energy_r2,
                                 sublabel="Ridge baseline"), md=3),
            dbc.Col(_metric_card("Danceability R²", dance_r2,
                                 sublabel="Ridge baseline"), md=3),
            dbc.Col(_metric_card("Songs in corpus", f"{n_total:,}",
                                 sublabel="train+test, after Spotify⨝Deezer"), md=2),
        ], className="g-3"),
    ], fluid=True)

    return html.Div([banner, cards_row])


def build_tabs():
    return dbc.Container([
        dbc.Tabs(
            id="dashboard-tabs",
            active_tab="tab-eda",
            children=[
                dbc.Tab(label="Project Overview & EDA",  tab_id="tab-eda"),
                dbc.Tab(label="Model Comparison",        tab_id="tab-models"),
                dbc.Tab(label="What Predicts Each Genre", tab_id="tab-explain"),
                dbc.Tab(label="Try It Live",             tab_id="tab-live"),
            ],
            className="mt-2",
        ),
        html.Div(id="tab-content", className="mt-3"),
    ], fluid=True, style={"paddingBottom": "3rem"})


app.layout = html.Div([
    build_header(),
    build_tabs(),
], style={"backgroundColor": "#F4F6F8", "minHeight": "100vh"})


# ============================================================================
# Tab callbacks
# ============================================================================
def _section_card(title, children, subtitle=None, height=None):
    body = []
    if subtitle:
        body.append(html.Div(subtitle, style={
            "color": "#7F8C8D", "fontSize": "0.92rem",
            "marginBottom": "0.8rem",
        }))
    body.extend(children)
    return dbc.Card([
        dbc.CardHeader(html.Strong(title)),
        dbc.CardBody(body, style={"padding": "1rem"}),
    ], className="shadow-sm mb-3", style={"border": "1px solid #E5E8EC"})


def render_tab_eda():
    audio_features = NUMERIC_AUDIO_FEATURES
    return html.Div([
        dbc.Row([
            dbc.Col(_section_card(
                "Why this project",
                [html.P([
                    "Genre, danceability, and energy from a song's lyrics + Spotify "
                    "features + Deezer metadata — useful for streaming services "
                    "that need to auto-tag uploaded music. The corpus is ~434K "
                    "songs after a Kaggle ⨝ Deezer inner join. Ten broad genres "
                    "with a 19.4× imbalance (Rock 37% → Classical 1.9%). "
                    "We use macro F1 as the primary metric to penalize "
                    "majority-class shortcuts."
                ], style={"marginBottom": 0}),
                ],
            ), md=12),
        ]),
        dbc.Row([
            dbc.Col(_section_card(
                "1. Class balance",
                [dcc.Graph(figure=fig_class_balance(CACHE["genre_counts"]),
                           config={"displayModeBar": False})],
                subtitle="Naive 'always Rock' baseline = 37.1%; random = 10%. "
                         "Models below must clear those.",
            ), md=12),
        ]),
        dbc.Row([
            dbc.Col(_section_card(
                "2. ANOVA effect sizes — which audio features carry signal",
                [dcc.Graph(figure=fig_anova(CACHE["anova"]),
                           config={"displayModeBar": False})],
                subtitle="η² = proportion of feature variance explained by genre. "
                         "Four large-effect features (η² ≥ 0.14) dominate model "
                         "importance later; popularity, tempo, liveness contribute "
                         "weak signal.",
            ), md=12),
        ]),
        dbc.Row([
            dbc.Col(_section_card(
                "3. Audio feature distribution by genre",
                [
                    html.Div([
                        html.Label("Audio feature:", style={"fontWeight": 600,
                                                            "marginRight": "0.6rem"}),
                        dcc.Dropdown(
                            id="eda-feature-dropdown",
                            options=[{"label": FEATURE_LABEL.get(f, f), "value": f}
                                     for f in audio_features],
                            value="danceability",
                            clearable=False,
                            style={"width": "260px", "display": "inline-block"},
                        ),
                    ], style={"display": "flex", "alignItems": "center",
                              "marginBottom": "0.6rem"}),
                    dcc.Graph(id="eda-boxplot",
                              config={"displayModeBar": False}),
                ],
                subtitle="Genres ordered by median value. Hip-Hop sits highest "
                         "on danceability and speechiness; Classical/Jazz on "
                         "acousticness; Rock/Electronic on energy.",
            ), md=12),
        ]),
        dbc.Row([
            dbc.Col(_section_card(
                "4. Feature correlation",
                [dcc.Graph(figure=fig_correlation_heatmap(CACHE["corr_matrix"]),
                           config={"displayModeBar": False})],
                subtitle="One tight cluster: energy ↔ loudness ↔ acousticness. "
                         "Handled by L2 in linear models; harmless for trees.",
            ), md=6),
            dbc.Col(_section_card(
                "5. Genre × decade (SQL JOIN demonstration)",
                [dcc.Graph(figure=fig_genre_era_heatmap(CACHE["genre_era"]),
                           config={"displayModeBar": False})],
                subtitle="songs ⨝ derived eras table on FLOOR(year/10)*10. "
                         "Hints at temporal drift in what each genre means.",
            ), md=6),
        ]),
        dbc.Row([
            dbc.Col(_section_card(
                "6. Top niche tags",
                [dcc.Graph(figure=fig_top_tags(CACHE["tag_counts"], top_n=20),
                           config={"displayModeBar": False})],
                subtitle=f"{CACHE.get('n_unique_tags', 0):,} unique tags total. "
                         f"Top 200 cover 88.7% of songs and become 200 binary "
                         f"features in the model.",
            ), md=6),
            dbc.Col(_section_card(
                "7. Genre composition of a chosen tag",
                [
                    html.Div([
                        html.Label("Niche tag:", style={"fontWeight": 600,
                                                        "marginRight": "0.6rem"}),
                        dcc.Dropdown(
                            id="tag-dropdown",
                            options=[{"label": t, "value": t}
                                     for t in CACHE.get("top_50_tags", [])],
                            value=(CACHE["top_50_tags"][0]
                                   if CACHE.get("top_50_tags") else None),
                            clearable=False,
                            style={"width": "240px", "display": "inline-block"},
                        ),
                    ], style={"display": "flex", "alignItems": "center",
                              "marginBottom": "0.6rem"}),
                    dcc.Graph(id="tag-genre-composition",
                              config={"displayModeBar": False}),
                ],
                subtitle="A 'metalcore'-tagged song is overwhelmingly Rock — "
                         "validates that niche tags add real signal.",
            ), md=6),
        ]),
    ])


def render_tab_models():
    classifier_options = (
        CACHE["classifier_metrics"]["model"].tolist()
        if "classifier_metrics" in CACHE else []
    )
    target_options = ["danceability", "energy"]
    initial_target = "energy"

    return html.Div([
        dbc.Row([
            dbc.Col(_section_card(
                "Headline finding",
                [html.P([
                    "Linear baselines won this problem. ",
                    html.Strong("Logistic Regression (multinomial, L2)"),
                    " reached 89.7% accuracy / 87.1% macro F1 on the held-out "
                    "test set. Random Forest and AdaBoost both lagged. "
                    "RandomizedSearchCV on LR moved test F1 ", html.Em("down"),
                    " a hair — the validation curve is on a flat plateau across "
                    "two orders of magnitude of C."
                ], style={"marginBottom": 0})],
            ), md=12),
        ]),
        dbc.Row([
            dbc.Col(_section_card(
                "1. Classifier comparison (held-out test set)",
                [dcc.Graph(
                    figure=fig_classifier_comparison(CACHE["classifier_metrics"])
                    if "classifier_metrics" in CACHE else go.Figure(),
                    config={"displayModeBar": False},
                )],
            ), md=12),
        ]),
        dbc.Row([
            dbc.Col(_section_card(
                "2. Confusion matrix per model",
                [
                    html.Div([
                        html.Label("Model:", style={"fontWeight": 600,
                                                    "marginRight": "0.6rem"}),
                        dcc.Dropdown(
                            id="cm-model-dropdown",
                            options=[{"label": m, "value": m}
                                     for m in classifier_options],
                            value=(classifier_options[0] if classifier_options
                                   else None),
                            clearable=False,
                            style={"width": "260px", "display": "inline-block"},
                        ),
                    ], style={"display": "flex", "alignItems": "center",
                              "marginBottom": "0.6rem"}),
                    dcc.Graph(id="confusion-matrix",
                              config={"displayModeBar": False}),
                ],
                subtitle="Diagonal cells = recall per genre. Off-diagonal cells "
                         "show which genre pairs each model confuses most. "
                         "Common confusions: Pop ↔ Electronic, R&B ↔ Pop.",
            ), md=12),
        ]),
        dbc.Row([
            dbc.Col(_section_card(
                "3. Regression — predicting danceability and energy",
                [dcc.Graph(
                    figure=fig_regression_comparison(CACHE["regression_metrics"])
                    if "regression_metrics" in CACHE else go.Figure(),
                    config={"displayModeBar": False},
                )],
                subtitle="Energy is highly predictable (R² ≈ 0.82) thanks to "
                         "tight correlations with loudness and acousticness. "
                         "Danceability sits at R² ≈ 0.57 — no single dominant "
                         "linear correlate. Ridge edges out RF Regressor on "
                         "both targets.",
            ), md=6),
            dbc.Col(_section_card(
                "4. Per-genre regression error",
                [
                    html.Div([
                        html.Label("Target:", style={"fontWeight": 600,
                                                     "marginRight": "0.6rem"}),
                        dcc.Dropdown(
                            id="rmse-target-dropdown",
                            options=[{"label": t, "value": t}
                                     for t in target_options],
                            value=initial_target,
                            clearable=False,
                            style={"width": "200px", "display": "inline-block"},
                        ),
                    ], style={"display": "flex", "alignItems": "center",
                              "marginBottom": "0.6rem"}),
                    dcc.Graph(id="per-genre-rmse",
                              config={"displayModeBar": False}),
                ],
                subtitle="Country and Rock are easiest (most training data); "
                         "Classical and Electronic hardest.",
            ), md=6),
        ]),
    ])


def render_tab_explain():
    has_fi = "rf_feature_importance" in CACHE
    has_lr = "lr_coef" in CACHE
    has_anova_rf = "anova_rf_compare" in CACHE
    lr_classes = CACHE.get("lr_classes", [])

    return html.Div([
        dbc.Row([
            dbc.Col(_section_card(
                "How the model decides",
                [html.P([
                    "Two complementary views of which features drive prediction: ",
                    html.Strong("Random Forest importances"), " (non-linear, "
                    "interaction-aware) and ", html.Strong("LR coefficients"),
                    " (linear, directional — they tell you which features push "
                    "toward each genre). Both tie back to the ANOVA effect "
                    "sizes from EDA.",
                ], style={"marginBottom": 0})],
            ), md=12),
        ]),
        dbc.Row([
            dbc.Col(_section_card(
                "1. Top features by Random Forest importance",
                [
                    dcc.Graph(
                        figure=(fig_rf_feature_importance(
                            CACHE["rf_feature_importance"], top_n=25)
                            if has_fi else go.Figure()),
                        config={"displayModeBar": False},
                    ),
                ],
                subtitle="Bars are colored by which feature block they come "
                         "from — audio (blue), LSA lyrics (orange), cluster "
                         "id (green), niche tag (red).",
            ), md=7),
            dbc.Col(_section_card(
                "2. Importance aggregated by feature block",
                [
                    dcc.Graph(
                        figure=(fig_bucket_importance(CACHE["bucket_importance"])
                                if "bucket_importance" in CACHE else go.Figure()),
                        config={"displayModeBar": False},
                    ),
                    html.Hr(),
                    html.Div([
                        html.Strong("Reading this: "),
                        "niche tags are the largest single block by sum of "
                        "importances despite carrying small per-feature weight. "
                        "The 11 numeric audio features punch well above their "
                        "weight on a per-feature basis.",
                    ], style={"fontSize": "0.9rem", "color": "#5D6D7E"}),
                ],
            ), md=5),
        ]),
        dbc.Row([
            dbc.Col(_section_card(
                "3. LR coefficients for a chosen genre",
                [
                    html.Div([
                        html.Label("Genre:", style={"fontWeight": 600,
                                                     "marginRight": "0.6rem"}),
                        dcc.Dropdown(
                            id="lr-genre-dropdown",
                            options=[{"label": g, "value": g} for g in lr_classes],
                            value=("Hip-Hop" if "Hip-Hop" in lr_classes
                                   else (lr_classes[0] if lr_classes else None)),
                            clearable=False,
                            style={"width": "200px", "display": "inline-block"},
                        ),
                    ] if has_lr else [
                        html.Em("LR model not loaded — coefficients unavailable.")
                    ], style={"display": "flex", "alignItems": "center",
                              "marginBottom": "0.6rem"}),
                    dcc.Graph(id="lr-coefficients",
                              config={"displayModeBar": False}),
                ],
                subtitle="Positive coefficients (green) push the prediction "
                         "toward this genre; negative (red) push away.",
            ), md=7),
            dbc.Col(_section_card(
                "4. ANOVA vs RF rank agreement (audio features)",
                [
                    dcc.Graph(
                        figure=(fig_anova_rf_agreement(CACHE["anova_rf_compare"])
                                if has_anova_rf else go.Figure()),
                        config={"displayModeBar": False},
                    ),
                    html.Hr(),
                    html.Div([
                        "Points near the diagonal = EDA's effect-size ranking "
                        "agrees with what the trained Random Forest learned. "
                        "Most features land near the diagonal — EDA called "
                        "the shots correctly.",
                    ], style={"fontSize": "0.9rem", "color": "#5D6D7E"}),
                ],
            ), md=5),
        ]),
    ])


def _make_slider_row(feat):
    lo, hi, default, step = DEFAULT_AUDIO_VALUES[feat]
    if isinstance(default, int):
        marks = {lo: str(lo), hi: str(hi)}
    else:
        marks = {lo: f"{lo:g}", hi: f"{hi:g}"}
    return html.Div([
        html.Div([
            html.Span(FEATURE_LABEL[feat], style={"fontWeight": 600}),
            html.Span(id={"type": "slider-value", "feature": feat},
                      style={"float": "right", "color": "#3498DB",
                             "fontVariantNumeric": "tabular-nums"}),
        ], style={"marginBottom": "0.2rem"}),
        dcc.Slider(
            id={"type": "audio-slider", "feature": feat},
            min=lo, max=hi, step=step, value=default,
            marks=marks,
            tooltip={"placement": "bottom", "always_visible": False},
        ),
    ], style={"marginBottom": "0.5rem"})


def render_tab_live():
    top_tags_options = CACHE.get("top_200_tags", [])[:200]
    return html.Div([
        dbc.Row([
            dbc.Col(_section_card(
                "Try it live",
                [html.P([
                    "Paste lyrics and tweak the audio sliders. The prediction "
                    "uses the same preprocessing pipeline as training: TF-IDF "
                    "→ truncated SVD → numeric scaling → cluster lookup → "
                    "concatenated 521-dim feature vector. ",
                    html.Strong("Note: "),
                    "for the danceability/energy regressors the corresponding "
                    "slider is dropped from the input (otherwise the model "
                    "would trivially predict the slider value), so those "
                    "predictions are based on lyrics + the OTHER audio "
                    "features.",
                ], style={"marginBottom": 0})],
            ), md=12),
        ]),
        dbc.Row([
            dbc.Col([
                _section_card(
                    "Lyrics",
                    [dcc.Textarea(
                        id="lyrics-input",
                        placeholder="Paste a song's lyrics here (one or two "
                                    "verses works fine)...",
                        style={"width": "100%", "height": "180px",
                               "fontSize": "0.9rem"},
                        value="",
                    )],
                ),
                _section_card(
                    "Niche tags (optional, multi-select)",
                    [dcc.Dropdown(
                        id="niche-tag-multi",
                        options=[{"label": t, "value": t}
                                 for t in top_tags_options],
                        value=[],
                        multi=True,
                        placeholder="Select 0+ tags from the top-200 list...",
                    )],
                    subtitle="Empty is a valid input (~11% of training songs "
                             "have no top-200 tag).",
                ),
            ], md=6),
            dbc.Col(
                _section_card(
                    "Audio features",
                    [
                        html.Div([_make_slider_row(f)
                                  for f in NUMERIC_AUDIO_FEATURES]),
                    ],
                    subtitle="Defaults are roughly the corpus medians.",
                ),
                md=6,
            ),
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Button(
                        [html.I(className="fas fa-rocket me-2"),
                         "Predict genre, danceability, energy"],
                        id="predict-btn",
                        color="primary",
                        size="lg",
                        style={"fontWeight": 600},
                    ),
                ], style={"textAlign": "center", "margin": "1rem 0"}),
                html.Div(id="live-predict-output"),
            ], md=12),
        ]),
    ])


@app.callback(
    Output("tab-content", "children"),
    Input("dashboard-tabs", "active_tab"),
)
def switch_tab(active_tab):
    if active_tab == "tab-eda":
        return render_tab_eda()
    if active_tab == "tab-models":
        return render_tab_models()
    if active_tab == "tab-explain":
        return render_tab_explain()
    if active_tab == "tab-live":
        return render_tab_live()
    return html.Div()


# ============================================================================
# Chart-level callbacks (dropdown-driven figures inside tabs)
# ============================================================================
@app.callback(
    Output("eda-boxplot", "figure"),
    Input("eda-feature-dropdown", "value"),
)
def update_eda_boxplot(feature):
    if not feature:
        return go.Figure()
    return fig_audio_boxplot(
        CACHE["audio_sample"], feature, CACHE["genre_order"]
    )


@app.callback(
    Output("tag-genre-composition", "figure"),
    Input("tag-dropdown", "value"),
)
def update_tag_composition(tag):
    if not tag:
        return go.Figure()
    return fig_tag_genre_composition(
        CACHE["tag_genre_composition"], tag, CACHE["genre_order"]
    )


@app.callback(
    Output("confusion-matrix", "figure"),
    Input("cm-model-dropdown", "value"),
)
def update_confusion_matrix(model_name):
    if not model_name or model_name not in CACHE.get("confusion_matrices", {}):
        return go.Figure()
    cm = CACHE["confusion_matrices"][model_name]
    return fig_confusion_matrix(cm, CACHE["genre_order"], model_name)


@app.callback(
    Output("per-genre-rmse", "figure"),
    Input("rmse-target-dropdown", "value"),
)
def update_per_genre_rmse(target):
    if not target:
        return go.Figure()
    df = CACHE.get("per_genre_rmse", {}).get((target, "RF Regressor"))
    if df is None:
        # Fallback to Ridge if RFR not loaded
        df = CACHE.get("per_genre_rmse", {}).get((target, "Ridge"))
    if df is None or df.empty:
        return go.Figure().update_layout(
            title=f"No regressor results for {target}", template=PLOTLY_TEMPLATE,
        )
    return fig_per_genre_rmse(df, target)


@app.callback(
    Output("lr-coefficients", "figure"),
    Input("lr-genre-dropdown", "value"),
)
def update_lr_coefficients(target_genre):
    if not target_genre or "lr_coef" not in CACHE:
        return go.Figure()
    return fig_lr_coefficients(
        CACHE["lr_coef"],
        CACHE["lr_classes"],
        CACHE["feature_names"],
        target_genre,
        top_n=10,
    )


# Slider value display (live updates next to each slider label)
@app.callback(
    Output({"type": "slider-value", "feature": dash.MATCH}, "children"),
    Input({"type": "audio-slider", "feature": dash.MATCH}, "value"),
)
def display_slider_value(v):
    if v is None:
        return ""
    return f"{v:.2f}" if isinstance(v, float) else str(v)


# ============================================================================
# Live predict callback
# ============================================================================
@app.callback(
    Output("live-predict-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("lyrics-input", "value"),
    State({"type": "audio-slider", "feature": dash.ALL}, "value"),
    State({"type": "audio-slider", "feature": dash.ALL}, "id"),
    State("niche-tag-multi", "value"),
    prevent_initial_call=True,
)
def run_prediction(n_clicks, lyrics, slider_values, slider_ids, selected_tags):
    if not n_clicks:
        return no_update

    lyrics = (lyrics or "").strip()
    audio_dict = {}
    for sid, val in zip(slider_ids or [], slider_values or []):
        feat = sid.get("feature")
        if feat is not None:
            audio_dict[feat] = float(val)

    # Run predict
    try:
        result = live_predict(
            lyrics, audio_dict, selected_tags or [],
            PREPROCESSORS, CLASSIFIERS, REGRESSORS, PRIMARY_CLF_NAME,
        )
    except Exception as e:
        return dbc.Alert(
            [html.Strong("Prediction failed: "), str(e)],
            color="danger",
        )

    # Build genre-prediction card
    pred_genre = result["predicted_genre"]
    badge_color = GENRE_COLORS.get(pred_genre, "#3498DB")
    top3_rows = []
    for g, p in result["top3"]:
        top3_rows.append(html.Div([
            html.Div([
                html.Span(g, style={"fontWeight": 600,
                                    "color": GENRE_COLORS.get(g, "#2C3E50")}),
                html.Span(f"{p*100:.1f}%",
                          style={"float": "right",
                                 "fontVariantNumeric": "tabular-nums"}),
            ], style={"marginBottom": "0.15rem"}),
            dbc.Progress(
                value=p * 100,
                color=("primary" if g == pred_genre else "secondary"),
                style={"height": "0.6rem", "marginBottom": "0.6rem"},
            ),
        ]))

    # All-classifier votes
    votes_rows = [
        html.Tr([
            html.Td(v["model"]),
            html.Td(v["predicted"], style={
                "fontWeight": 600,
                "color": GENRE_COLORS.get(v["predicted"], "#2C3E50"),
            }),
        ])
        for v in result["all_votes"]
    ]
    votes_table = dbc.Table([
        html.Thead(html.Tr([html.Th("Model"), html.Th("Predicted")])),
        html.Tbody(votes_rows),
    ], bordered=False, size="sm", style={"marginBottom": 0})

    genre_card = _section_card(
        f"Predicted genre — {pred_genre}",
        [
            html.Div(
                f"Primary model: {result['primary_classifier']}",
                style={"color": "#7F8C8D", "fontSize": "0.88rem",
                       "marginBottom": "0.8rem"},
            ),
            html.Div(top3_rows),
            html.Hr(),
            html.Strong("All-classifier votes (sanity check):"),
            votes_table,
        ],
    )

    # Regression card
    dance_block = []
    energy_block = []
    if result.get("dance_pred") is not None:
        dp = result["dance_pred"]
        slider_dance = audio_dict.get("danceability", 0)
        dance_block = [
            html.Div("Predicted danceability",
                     style={"fontSize": "0.85rem", "color": "#7F8C8D",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.04em"}),
            html.Div(f"{dp:.3f}",
                     style={"fontSize": "2rem", "fontWeight": 700,
                            "color": "#2C3E50"}),
            html.Div(f"slider was {slider_dance:.2f}, "
                     f"gap = {abs(dp-slider_dance):.3f}",
                     style={"fontSize": "0.8rem", "color": "#95A5A6"}),
        ]
    if result.get("energy_pred") is not None:
        ep = result["energy_pred"]
        slider_energy = audio_dict.get("energy", 0)
        energy_block = [
            html.Div("Predicted energy",
                     style={"fontSize": "0.85rem", "color": "#7F8C8D",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.04em"}),
            html.Div(f"{ep:.3f}",
                     style={"fontSize": "2rem", "fontWeight": 700,
                            "color": "#2C3E50"}),
            html.Div(f"slider was {slider_energy:.2f}, "
                     f"gap = {abs(ep-slider_energy):.3f}",
                     style={"fontSize": "0.8rem", "color": "#95A5A6"}),
        ]

    regression_card = _section_card(
        "Regression — Ridge predictions",
        [
            dbc.Row([
                dbc.Col(html.Div(dance_block,
                                 style={"padding": "1rem",
                                        "borderRight": "1px solid #ECF0F1"}),
                        md=6),
                dbc.Col(html.Div(energy_block,
                                 style={"padding": "1rem"}),
                        md=6),
            ]) if (dance_block or energy_block)
                else html.Em("No regression models loaded."),
            html.Hr(),
            html.Div([
                html.Strong("Why doesn't this just match the slider? "),
                "When predicting danceability, the danceability column is "
                "dropped from the input — same for energy. The gap shows how "
                "well the model recovers the target from lyrics + the other "
                "audio features alone.",
            ], style={"fontSize": "0.85rem", "color": "#5D6D7E"}),
        ],
    )

    return [genre_card, regression_card]


# ============================================================================
# Main entry
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuild of the EDA + scoring cache.")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("=" * 68)
    print("CIS 2450 Final Project — Dashboard starting up")
    print("=" * 68)
    print(f"Working directory: {BASE_DIR}")
    print(f"Database: {DB_PATH}  ({'OK' if DB_PATH.exists() else 'MISSING'})")
    print(f"Models:   {MODELS_DIR}  ({'OK' if MODELS_DIR.exists() else 'MISSING'})")
    print()

    global CACHE, PREPROCESSORS, CLASSIFIERS, REGRESSORS, PRIMARY_CLF_NAME
    CACHE, PREPROCESSORS, CLASSIFIERS, REGRESSORS = load_or_build_cache(
        rebuild=args.rebuild_cache
    )
    PRIMARY_CLF_NAME, _ = pick_primary_classifier(CLASSIFIERS)
    print(f"Primary live-predict classifier: {PRIMARY_CLF_NAME}")

    # Layout has to be re-built now that CACHE is populated
    app.layout = html.Div([
        build_header(),
        build_tabs(),
    ], style={"backgroundColor": "#F4F6F8", "minHeight": "100vh"})

    print()
    print("=" * 68)
    print(f"Dashboard ready at http://{args.host}:{args.port}")
    print("=" * 68)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
