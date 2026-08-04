# escalation_features.py
# ---------------------------------------------------------------------------
# Pure-Python (pandas / numpy only — NO Spark, NO MLflow) feature-engineering
# logic for the Scapia ticket-escalation pipeline.
#
# This module is the SINGLE SOURCE OF TRUTH for the ticket-grain feature maths
# ported from `greylabs_lte_v3.ipynb`. It is imported by:
#   * notebooks/03_gold_features.py   — runs `build_ticket_feature_record`
#     inside a Spark `groupBy(ticket_id).applyInPandas(...)` so the EXACT same
#     per-ticket logic scales out, one ticket's calls per pandas group.
#   * notebooks/05_train_register.py  — reuses the encoders (GL ordinal map,
#     K-Fold target encoding, median imputation) and bakes the frozen versions
#     into the registered pyfunc.
#   * tests/test_features.py          — unit-tests this logic with NO Spark.
#
# Keeping the maths here (not duplicated inline in each notebook) means the unit
# tests exercise the real production code path, not a copy of it.
#
# --- Two deliberate reconciliations vs. the reference notebook / spec ---------
# 1. Sentiment top bucket. The silver schema labels the positive bucket of
#    `agent_sentiment` / `customer_sentiment` as "positive" (raw Good AND
#    Positive both fold into it), while `empathy` / `effective_communication`
#    use "good". `LADDER_RANK` / `GL_ORDINAL_MAP` therefore treat "good" and
#    "positive" as the SAME rank (2) so worst-of and ordinal encoding are
#    consistent across both vocabularies. (Matches the spec's
#    "bad=0, neutral=1, good/positive=2".)
# 2. Speech ratios. The spec's explicit per-feature formula is
#    "mean(Agent Talk Duration / Total Call Duration) across calls"
#    (mean-of-per-call-ratios). The reference notebook computed a
#    ratio-of-means (mean(agent)/mean(total)). We follow the SPEC formula here
#    (mean-of-ratios) — see `_mean_of_ratios`.
# ---------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column-name contract (silver -> gold). Kept here so notebooks and tests never
# hardcode strings independently.
# ---------------------------------------------------------------------------
ID_COL = "ticket_id"
DATETIME_COL = "call_datetime"  # MUST be a UTC instant (02_silver_clean sets this; IST kept as call_datetime_ist)
CALL_ID_COL = "call_id"  # true per-call unique key — deterministic secondary sort + silver dedup key
TARGET_COL = "real_time_alert"  # cleaned: 'yes' / 'no' / 'inconclusive' / None
LABEL_COL = "is_escalated"

# Model feature groups (the frozen contract shared with the pyfunc).
NUMERIC_FEATURES = [
    "num_calls_considered",
    "time_diff_hours",
    "calls_per_day",
    "agent_speech_ratio",
    "customer_speech_ratio",
    "customer_sentiment_trend",
    "total_call_duration",
    "total_nonspeech_duration",
    "agent_talk_duration",
    "customer_talk_duration",
    "weighted_average",
    "neg_sentiment_ratio",  # NEW feature (agreed Aug 3 meeting)
]
CATEGORICAL_GL_FEATURES = [
    "gl_empathy",
    "gl_agent_sentiment",
    "gl_customer_sentiment",
    "gl_repeat_calls",
]
CATEGORICAL_HIGH_CARD_FEATURES = [
    "scapia_category",
    "scapia_sub_category",
    "qrc_category",
    "qrc_sub_category",
]
# Frozen model-input column order (numeric -> GL ordinals -> target-encoded).
FEATURE_NAMES = (
    NUMERIC_FEATURES + CATEGORICAL_GL_FEATURES + CATEGORICAL_HIGH_CARD_FEATURES
)

# Metadata carried in the gold table but NEVER fed to the model.
META_COLS = [
    ID_COL,
    LABEL_COL,
    "first_call_datetime",
    "last_call_datetime",
    "total_calls_in_ticket",
]

# ---------------------------------------------------------------------------
# Ordinal ladder. bad (0) is always the "worst"; good == positive == best (2).
# ---------------------------------------------------------------------------
LADDER_RANK = {"bad": 0, "neutral": 1, "good": 2, "positive": 2}
# Ordinal encoding baked into the model. Unseen / null -> -1 (XGBoost splits it
# out) — mirrors the reference's OrdinalEncoder(unknown_value=-1).
GL_ORDINAL_MAP = {"bad": 0, "neutral": 1, "good": 2, "positive": 2}
GL_UNKNOWN_ORDINAL = -1

# Values that mean "missing" anywhere in the raw Greylabs export.
NULL_SENTINELS = {"-", "na", "n/a", "nan", "none", "null", "inconclusive", ""}


# ---------------------------------------------------------------------------
# Small scalar helpers (also used by silver-clean logic + tests)
# ---------------------------------------------------------------------------
def to_null(value):
    """Normalize a raw cell to a lowercased string, or None if it is a sentinel."""
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
    except TypeError:
        pass
    s = str(value).strip().lower()
    if s in NULL_SENTINELS:
        return None
    return s


def normalize_qrc_subcategory(value):
    """QRC sub-category casing fix: strip -> lower -> title.

    'Flight Related' and 'flight related' both map to 'Flight Related'.
    """
    s = to_null(value)
    if s is None:
        return None
    return s.title()


def parse_repeat_call_flag(value):
    """Parse the raw Greylabs `Repeat Calls` free text into a boolean flag.

    True  when the text contains 'repeat call: yes'
    False when the text contains 'repeat call: no'
    None  otherwise.
    """
    if value is None:
        return None
    s = str(value).strip().lower()
    if "repeat call: yes" in s:
        return True
    if "repeat call: no" in s:
        return False
    return None


def _rank(value):
    """Ordinal rank of a sentiment/quality string, or None if unrankable."""
    s = to_null(value)
    if s is None:
        return None
    return LADDER_RANK.get(s)


def worst_of(values) -> str | None:
    """Return the worst (most negative) rankable value observed.

    Worst rank wins: bad (0) < neutral (1) < good/positive (2). Nulls and
    unrankable values are ignored. Returns None if nothing is rankable.
    """
    best_rank = None
    best_val = None
    for v in values:
        s = to_null(v)
        if s is None:
            continue
        r = LADDER_RANK.get(s)
        if r is None:
            continue
        if best_rank is None or r < best_rank:
            best_rank, best_val = r, s
    return best_val


def latest_non_null(values):
    """Return the last non-null (post-sentinel-cleaning) value, else None.

    The original (display) casing is preserved for high-cardinality category
    columns — only sentinels are dropped.
    """
    result = None
    for v in values:
        if v is None:
            continue
        try:
            if isinstance(v, float) and np.isnan(v):
                continue
        except TypeError:
            pass
        if to_null(v) is None:
            continue
        result = v
    return result


def customer_sentiment_trend(sentiments) -> float:
    """rank(last) - rank(first) over rankable customer-sentiment values.

    Negative = worsening, positive = improving. 0.0 when fewer than two
    rankable values are present (spec + reference behaviour).
    """
    ranks = [_rank(v) for v in sentiments]
    ranks = [r for r in ranks if r is not None]
    if len(ranks) < 2:
        return 0.0
    return float(ranks[-1] - ranks[0])


def neg_sentiment_ratio(customer_sentiments, num_calls_considered) -> float:
    """Proportion of calls whose customer sentiment is 'bad'.

    count(customer_sentiment == 'bad') / num_calls_considered.
    num_calls_considered is always >= 1 in the pre-escalation slice, so there
    is no divide-by-zero.
    """
    if num_calls_considered <= 0:
        return 0.0
    n_bad = sum(1 for v in customer_sentiments if to_null(v) == "bad")
    return float(n_bad) / float(num_calls_considered)


def _mean_of_ratios(numerators, denominators) -> float:
    """mean over calls of (numerator / denominator), skipping calls where the
    denominator is missing or <= 0. Returns NaN if no call qualifies."""
    num = pd.to_numeric(pd.Series(list(numerators)), errors="coerce").to_numpy(dtype="float64")
    den = pd.to_numeric(pd.Series(list(denominators)), errors="coerce").to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(den > 0, num / den, np.nan)
    if np.all(np.isnan(ratios)):
        return float("nan")
    return float(np.nanmean(ratios))


def _mean(series) -> float:
    """Null-safe mean of a numeric-coercible iterable; NaN when all-null."""
    s = pd.to_numeric(pd.Series(list(series)), errors="coerce")
    if s.dropna().empty:
        return float("nan")
    return float(s.mean(skipna=True))


# ---------------------------------------------------------------------------
# Ticket-grain feature record — the core port of the reference aggregation.
# Operates on ONE ticket's calls (a pandas DataFrame). Returns a flat dict of
# gold-table columns, or None when the ticket must be excluded.
# ---------------------------------------------------------------------------
def build_ticket_feature_record(calls: pd.DataFrame, training: bool = True) -> dict | None:
    """Aggregate one ticket's calls into a single gold feature row.

    Rules (per spec):
      * Sort calls by (`call_datetime`, `call_id`) — the DETERMINISTIC order. The
        `call_id` secondary key breaks tied timestamps reproducibly, so the
        pre-escalation boundary does NOT depend on arbitrary Spark input order
        (item 4). `call_datetime` must be a UTC instant (item 7).
      * Pre-escalation slice: keep only calls STRICTLY BEFORE the first
        `real_time_alert == 'yes'` call. If a ticket never escalates, all calls
        are used.
      * Exclude tickets whose FIRST call already escalated (pre-slice empty).
      * Label `is_escalated` = 1 if ANY call escalated, else 0.
      * `training=True`: also drop tickets with no valid (yes/no) alert on any
        call (unlabelled — cannot be a confirmed negative). At inference we
        score whatever calls exist.

    Returns the feature dict, or None if the ticket is excluded.
    """
    if calls is None or len(calls) == 0:
        return None

    # Deterministic ordering. Sort by call_datetime, then by call_id as a stable
    # tie-breaker so two calls sharing a timestamp always order the same way
    # regardless of the (arbitrary) order Spark handed the group to us. If a
    # call_id column is somehow absent we fall back to a stable sort on the
    # timestamp only (mergesort), but call_id is a required silver column.
    sort_keys = [DATETIME_COL]
    if CALL_ID_COL in calls.columns:
        sort_keys.append(CALL_ID_COL)
    g = calls.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    total_calls_in_ticket = len(g)

    flags = [to_null(f) for f in g[TARGET_COL].tolist()]
    has_valid = any(f in ("yes", "no") for f in flags)
    if training and not has_valid:
        return None

    # Cut point: index of the first escalated call.
    cut = next((i for i, f in enumerate(flags) if f == "yes"), None)
    if cut is not None:
        pre = g.iloc[:cut]
        label = 1
    else:
        pre = g
        label = 0

    if len(pre) == 0:
        # First call already escalated -> no pre-escalation window.
        return None

    n = len(pre)
    first_dt = pre[DATETIME_COL].iloc[0]
    last_dt = pre[DATETIME_COL].iloc[-1]
    diff_hours = (last_dt - first_dt).total_seconds() / 3600.0

    # Non-speech duration can be negative in the raw feed — clamp to >= 0.
    nonspeech_clamped = pd.to_numeric(
        pre["total_nonspeech_duration"], errors="coerce"
    ).clip(lower=0)

    # gl_repeat_calls: a repeat call (flag True) is the "bad" signal.
    repeat_as_rank = [
        ("bad" if f is True else "good" if f is False else None)
        for f in pre["repeat_call_flag"].tolist()
    ]

    record = {
        ID_COL: pre[ID_COL].iloc[0],
        LABEL_COL: label,
        "first_call_datetime": first_dt,
        "last_call_datetime": last_dt,
        "total_calls_in_ticket": total_calls_in_ticket,
        # ---- numeric model features ----
        "num_calls_considered": n,
        "time_diff_hours": diff_hours,
        "calls_per_day": n / max(diff_hours / 24.0, 1.0 / 24.0),
        "agent_speech_ratio": _mean_of_ratios(
            pre["agent_talk_duration"], pre["total_call_duration"]
        ),
        "customer_speech_ratio": _mean_of_ratios(
            pre["customer_talk_duration"], pre["total_call_duration"]
        ),
        "customer_sentiment_trend": customer_sentiment_trend(
            pre["customer_sentiment"].tolist()
        ),
        "total_call_duration": _mean(pre["total_call_duration"]),
        "total_nonspeech_duration": _mean(nonspeech_clamped),
        "agent_talk_duration": _mean(pre["agent_talk_duration"]),
        "customer_talk_duration": _mean(pre["customer_talk_duration"]),
        "weighted_average": _mean(pre["weighted_average"]),
        "neg_sentiment_ratio": neg_sentiment_ratio(
            pre["customer_sentiment"].tolist(), n
        ),
        # ---- GL worst-of features ----
        "gl_empathy": worst_of(pre["empathy"].tolist()),
        "gl_agent_sentiment": worst_of(pre["agent_sentiment"].tolist()),
        "gl_customer_sentiment": worst_of(pre["customer_sentiment"].tolist()),
        "gl_repeat_calls": worst_of(repeat_as_rank),
        # ---- high-cardinality latest-non-null features ----
        "scapia_category": latest_non_null(pre["scapia_category"].tolist()),
        "scapia_sub_category": latest_non_null(pre["scapia_sub_category"].tolist()),
        "qrc_category": latest_non_null(pre["qrc_category"].tolist()),
        "qrc_sub_category": latest_non_null(pre["qrc_sub_category"].tolist()),
    }
    return record


# ---------------------------------------------------------------------------
# Encoders shared between training and the frozen pyfunc.
# ---------------------------------------------------------------------------
def encode_gl_ordinal(series: pd.Series) -> pd.Series:
    """Map a GL string column to its frozen ordinal (unseen/null -> -1)."""
    return (
        series.map(lambda v: GL_ORDINAL_MAP.get(to_null(v), GL_UNKNOWN_ORDINAL))
        .astype("float64")
    )


def compute_medians(X: pd.DataFrame, numeric_features) -> dict:
    """Frozen median per numeric feature, computed on the TRAIN split only."""
    medians = {}
    for c in numeric_features:
        col = pd.to_numeric(X[c], errors="coerce")
        med = col.median()
        medians[c] = float(med) if pd.notna(med) else 0.0
    return medians


def apply_medians(X: pd.DataFrame, medians: dict, numeric_features) -> pd.DataFrame:
    """Impute numeric features with frozen medians."""
    out = X.copy()
    for c in numeric_features:
        col = pd.to_numeric(out.get(c, np.nan), errors="coerce")
        out[c] = col.fillna(medians.get(c, 0.0)).astype("float64")
    return out


def compute_target_encode_maps(X, y, cols, smoothing: float = 10.0, global_mean: float | None = None):
    """FULL-data smoothed target-encoding maps — the SERVING encoder.

    Returns ``(encode_maps, global_mean)`` where ``encode_maps[col] = {category: smoothed_target_mean}``.
    Smoothing: ``(n*cat_mean + a*global_mean) / (n + a)``.

    These maps include EVERY row's label, so they are correct for (a) the frozen pyfunc at inference and
    (b) encoding a held-out fold whose labels are NOT in ``(X, y)``. They MUST NOT be used to encode the same
    ``(X, y)`` rows the booster trains on — that leaks each row's own label. Use ``kfold_oof_encode`` for
    training rows (item 1).
    """
    if global_mean is None:
        global_mean = float(y.mean())
    encode_maps = {}
    for col in cols:
        if col not in X.columns:
            continue
        stats = (
            pd.concat([X[col].rename("cat"), y.rename("t")], axis=1)
            .groupby("cat")["t"]
            .agg(n="count", s="sum")
        )
        stats["smoothed"] = (stats["s"] + smoothing * global_mean) / (stats["n"] + smoothing)
        encode_maps[col] = {str(k): float(v) for k, v in stats["smoothed"].items()}
    return encode_maps, float(global_mean)


def kfold_oof_encode(
    X: pd.DataFrame,
    y: pd.Series,
    cols,
    n_splits: int = 5,
    smoothing: float = 10.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """OUT-OF-FOLD smoothed target encoding for TRAINING rows (no leakage).

    Each row's high-cardinality encoding is computed from statistics on the OTHER folds ONLY, so a row's own
    label never enters its own feature value. This is the matrix the booster must TRAIN on (item 1). Fold
    smoothing uses the full-``y`` global mean (matches the reference), and unseen categories fall back to it.

    Returns a copy of ``X`` with ``cols`` replaced by their OOF encodings; all other columns are untouched.
    """
    from sklearn.model_selection import KFold

    global_mean = float(y.mean())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    out = X.copy()
    for col in cols:
        if col not in X.columns:
            continue
        oof = pd.Series(np.nan, index=X.index, dtype="float64")
        for tr_idx, val_idx in kf.split(X):
            fold_maps, _ = compute_target_encode_maps(
                X.iloc[tr_idx], y.iloc[tr_idx], [col], smoothing=smoothing, global_mean=global_mean
            )
            oof.iloc[val_idx] = (
                X.iloc[val_idx][col].astype("object").map(fold_maps[col]).fillna(global_mean)
            )
        out[col] = oof.fillna(global_mean)
    return out


def kfold_target_encode(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    cols,
    n_splits: int = 5,
    smoothing: float = 10.0,
    random_state: int = 42,
):
    """K-Fold target encoding with Bayesian smoothing (ported from reference).

    Train : OUT-OF-FOLD encoding (each row encoded from the OTHER folds only) — the leakage-free TRAINING
            matrix. Delegates to ``kfold_oof_encode``.
    Test  : encoded from FULL-training statistics (``compute_target_encode_maps``).

    Returns ``(X_tr_enc, X_te_enc, encode_maps, global_mean)``. ``encode_maps`` / ``global_mean`` are the
    FULL-training (serving) maps frozen into the pyfunc; ``X_tr_enc`` carries the OOF encodings used to TRAIN.
    """
    X_tr_enc = kfold_oof_encode(
        X_tr, y_tr, cols, n_splits=n_splits, smoothing=smoothing, random_state=random_state
    )
    encode_maps, global_mean = compute_target_encode_maps(X_tr, y_tr, cols, smoothing=smoothing)
    X_te_enc = X_te.copy()
    for col in cols:
        if col not in X_tr.columns:
            continue
        X_te_enc[col] = X_te[col].astype("object").map(encode_maps[col]).fillna(global_mean)
    return X_tr_enc, X_te_enc, encode_maps, global_mean


def assign_split_by_ticket(
    ticket_ids,
    test_frac: float = 0.20,
    val_frac_of_train: float = 0.15,
    seed: int = 42,
):
    """Deterministic, Spark-order-INDEPENDENT split assignment BY TICKET ID (item 3).

    Returns a pandas Series of ``{'train','val','test'}`` aligned to ``ticket_ids``. A ticket's split is a
    pure function of its id (stable md5 hash -> [0,1) bucket) and the fractions, so:
      * the SAME ticket always lands in the SAME split — identical in 04_hpo and 05_train_register, regardless
        of Spark row order or which notebook runs;
      * all rows of a ticket share one split (gold is one row/ticket, but this holds even if not);
      * re-running yields byte-identical assignments (no ``train_test_split`` row-position dependence).

    Test is the sealed lower ``test_frac`` of the hash space. The remaining space is split into val/train with
    a SECOND, independent hash salt so val selection is orthogonal to the test cut. ``seed`` salts both hashes.
    Note: this is a hash split, not a stratified one — for the thousands of tickets here the class balance is
    preserved closely, and determinism/leakage-safety is the priority the review requires.
    """
    import hashlib

    def _bucket(value, salt) -> float:
        h = hashlib.md5(f"{seed}:{salt}:{value}".encode("utf-8")).hexdigest()
        return int(h[:8], 16) / float(0xFFFFFFFF)

    ids = pd.Series(list(ticket_ids))
    labels = []
    for tid in ids:
        if _bucket(tid, "test") < test_frac:
            labels.append("test")
        elif _bucket(tid, "val") < val_frac_of_train:
            labels.append("val")
        else:
            labels.append("train")
    return pd.Series(labels, index=ids.index)


def apply_target_encode(series: pd.Series, encode_map: dict, global_mean: float) -> pd.Series:
    """Inference-time target encoding: map via frozen encode_map; unseen -> global_mean."""
    return series.astype("object").map(encode_map).fillna(global_mean).astype("float64")


def build_model_matrix(
    pdf: pd.DataFrame,
    medians: dict,
    encode_maps: dict,
    global_mean: float,
    feature_names=FEATURE_NAMES,
) -> pd.DataFrame:
    """FROZEN raw-gold -> model-matrix transform (used inside the pyfunc).

    numeric  : median-impute with frozen medians
    GL cats  : frozen ordinal map (unseen/null -> -1)
    high-card: frozen target-encoding maps (unseen -> global_mean)
    Assembled in the frozen `feature_names` order.
    """
    parts = {}
    for c in NUMERIC_FEATURES:
        col = pd.to_numeric(pdf.get(c, np.nan), errors="coerce")
        parts[c] = col.fillna(medians.get(c, 0.0)).astype("float64")
    for c in CATEGORICAL_GL_FEATURES:
        parts[c] = encode_gl_ordinal(pdf[c]) if c in pdf.columns else pd.Series(
            GL_UNKNOWN_ORDINAL, index=pdf.index, dtype="float64"
        )
    for c in CATEGORICAL_HIGH_CARD_FEATURES:
        if c in pdf.columns:
            parts[c] = apply_target_encode(pdf[c], encode_maps.get(c, {}), global_mean)
        else:
            parts[c] = pd.Series(global_mean, index=pdf.index, dtype="float64")
    X = pd.DataFrame(parts, index=pdf.index)
    return X.reindex(columns=list(feature_names), fill_value=0.0)


def f2_score(y_true, y_pred) -> float:
    """F-beta with beta=2 (recall-weighted). Pure numpy, no sklearn dependency."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    beta2 = 4.0
    denom = (1 + beta2) * tp + beta2 * fn + fp
    if denom == 0:
        return 0.0
    return float((1 + beta2) * tp / denom)


def best_f2_threshold(y_true, y_proba, lo=0.1, hi=0.9, step=0.05):
    """Grid-search the probability threshold maximizing F2 on a holdout.

    Returns (best_threshold, best_f2). Ties resolved toward the lower threshold
    (higher recall).
    """
    y_proba = np.asarray(y_proba, dtype="float64")
    best_thr, best_val = lo, -1.0
    thr = lo
    # Use a rounded range to avoid float drift on the grid.
    n_steps = int(round((hi - lo) / step))
    for i in range(n_steps + 1):
        thr = round(lo + i * step, 4)
        preds = (y_proba >= thr).astype(int)
        val = f2_score(y_true, preds)
        if val > best_val:
            best_val, best_thr = val, thr
    return best_thr, best_val
