# Databricks notebook source
# MAGIC %md
# MAGIC # Scapia — First-Order-Propensity — Training SPINE + FEATURE SELECTION
# MAGIC
# MAGIC Companion to `first_order_propensity_user_features.py` (the per-user **feature store** notebook).
# MAGIC This notebook owns the two **per-model** steps that deliberately do NOT live in the shared feature
# MAGIC table:
# MAGIC
# MAGIC 1. **Spine** — the label + eligibility population (`internal_user_id`, `feature_ts`, `output`), built
# MAGIC    from the LABEL / ELIGIBILITY half of the reference `v7` ETL query. The *feature* half of that query
# MAGIC    is intentionally absent here — those per-user features already live in the feature table.
# MAGIC 2. **Point-in-time training set** — a leak-free as-of join of the spine against the feature table via
# MAGIC    `FeatureEngineeringClient.create_training_set(..., timestamp_lookup_key='feature_ts')`.
# MAGIC 3. **Feature selection** — a staged, MLflow-logged pipeline run *on the PIT training set*: screen →
# MAGIC    redundancy → relevance → model-based (XGBoost gain + SHAP) → leakage guard.
# MAGIC 4. **Wire-back** — the selected subset fed back into a `FeatureLookup`, matching the hook the feature
# MAGIC    store notebook demonstrates.
# MAGIC
# MAGIC ## Design split kept intact
# MAGIC * **Label / eligibility** → this notebook (the spine).
# MAGIC * **Features** → the shared feature table, joined point-in-time.
# MAGIC * **Encoding** (bucketing / one-hot / imputation for the model) → the downstream *model pipeline*, NOT
# MAGIC   here. Feature selection operates on the **raw** stored values. Any imputation / one-hot that appears
# MAGIC   below is a **selector-only** computation (mutual-information, correlation, tree models need finite
# MAGIC   numerics) and is clearly marked as such — it is never written back and is not the production encoding.
# MAGIC
# MAGIC ## Out of scope (not in this notebook)
# MAGIC No backfill cell, no scheduled/incremental refresh job, no model training/registration, no serving.
# MAGIC This notebook stops at *"selected features logged to MLflow + demonstrated in the lookup"*.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Dependencies
# MAGIC `databricks-feature-engineering`, `xgboost`, `scikit-learn`, `shap` and `mlflow` all ship with the
# MAGIC Databricks ML Runtime. The install below just pins recent versions so `timestamp_lookup_key` on
# MAGIC `create_training_set` and `shap.TreeExplainer` are available. Safe to skip on a current ML Runtime.

# COMMAND ----------

# MAGIC %pip install -U databricks-feature-engineering xgboost scikit-learn shap mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 1 — Spine (label + eligibility)
# MAGIC
# MAGIC The spine is the LABEL / ELIGIBILITY half of the reference `v7` ETL query, and nothing else. It emits
# MAGIC exactly three columns — `internal_user_id`, `feature_ts`, `output` — so it can be handed straight to
# MAGIC `create_training_set`. The feature CTEs from `v7` (card / coin / search / app-open / lounge aggregates)
# MAGIC are **omitted**: those values are served point-in-time from the feature table in Section 2.

# COMMAND ----------

# ---------------------------------------------------------------------------
# CONFIG — edit here
# ---------------------------------------------------------------------------

# Feature table produced by first_order_propensity_user_features.py (verbatim).
CATALOG = "mlops_data_science"
SCHEMA = "features"
TABLE = "first_order_propensity_user_features"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"

# Entity key + point-in-time key — must match the feature table's PK / timeseries column exactly.
ENTITY_KEY = "internal_user_id"
FEATURE_TS = "feature_ts"
LABEL = "output"

# --- Source tables for the spine (read-only; label + eligibility ONLY) ------
ORDERS_TABLE = "rds_main.scapiadb.orders"                    # label + first-order anti-join
ONBOARDED_USERS_FACT = "simple.crud.onboarded_users_fact"    # carded universe + onboarding date
# `segment` is NOT in the feature table (audit finding #1 — omitted there for lack of an as-of column).
# It is sourced here from its original CRUD table for Section 2's feature set (see the caveat there).
USER_SEGMENT_MAPPING = "simple.crud.user_segment_mapping"

# IST offset. created_at in the RDS-sourced tables is UTC; the reference query anchors the cutoff in IST
# and subtracts this offset before comparing against the UTC values.
IST_OFFSET_MINUTES = 330

# Performance (label observation) window length in days after the cutoff — v7 uses 90.
PERFORMANCE_DAYS = 90

# Business-confirmed status filter — kept EXACTLY as v7 has it (NOT narrowed to COMPLETE-only).
QUALIFYING_STATUSES = ["COMPLETE", "CANCELLED"]
QUALIFYING_PRODUCT_CATEGORIES = [
    "FLIGHT", "BUS", "TRAIN", "HOTEL_STAY",
    "ECOMMERCE", "EXPERIENCE", "VISA", "HOLIDAY",
]

# --- Numeric feature superset to pull from the feature table for selection --
# Every name below is a REAL column produced by first_order_propensity_user_features.py.
# (onboarding_completion_date is stored there too but is a raw timestamp needing downstream
#  encoding, so it is not a selection candidate here.)
STORE_FEATURE_CANDIDATES = [
    "activation_days",
    "t_30_txn",
    "first_30_txn",
    "coins_bal_overall",
    "card_txn_lifetime",
    "flight_searches_7d", "flight_searches_15d", "flight_searches_30d", "days_since_last_flight_search",
    "hotel_searches_7d", "hotel_searches_15d", "hotel_searches_30d", "days_since_last_hotel_search",
    "bus_searches_7d", "bus_searches_15d", "bus_searches_30d", "days_since_last_bus_search",
    "train_searches_7d", "train_searches_15d", "train_searches_30d", "days_since_last_train_search",
    "app_opens_7d", "app_opens_30d", "app_opens_lifetime", "days_since_last_app_open",
    "lounge_used",
]

# Optional down-sampling ONLY for the single-node selection pass (Section 3). Leave None to use the
# full PIT training set; set a fraction (e.g. 0.25) if the eligible population is too large to fit
# in driver memory for pandas / XGBoost / SHAP.
SELECTION_SAMPLE_FRACTION = None

# Feature-selection thresholds (tune per dataset).
SCREEN_MISSING_THRESH = 0.98    # drop a feature missing in > 98% of rows
SCREEN_DOMINANT_THRESH = 0.999  # drop a feature whose single most-common value covers > 99.9% of rows
CORR_THRESH = 0.90              # prune one of any |spearman| > 0.90 pair (keep the stronger)
CV_SPLITS = 5                   # folds for model-based stability + leakage OOF encoding
MODEL_TOPK = 15                 # per-fold top-K (by gain) used for the stability vote
LEAKAGE_AUC_THRESH = 0.80       # audit any feature whose SOLO AUC exceeds this
RANDOM_STATE = 42

# ---------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1a. Parameters — cutoff / as-of date
# MAGIC The cutoff (a.k.a. as-of date) is `feature_ts`: the point-in-time at which the label window opens. It
# MAGIC **must** equal the `as_of_date` the feature table was materialized for, so the as-of join finds a row.
# MAGIC Leave the widget blank to auto-derive the `v7` reference cutoff (latest qualifying-order time in IST,
# MAGIC minus `PERFORMANCE_DAYS`). **For reproducible training runs always pass an explicit `as_of_date`** —
# MAGIC the auto default drifts as new orders arrive.

# COMMAND ----------

from datetime import datetime, timedelta

dbutils.widgets.text(
    "as_of_date",
    "",
    "Cutoff / as-of (YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS', IST). Blank = auto-derive v7 reference cutoff.",
)


def _sql_in_list(values):
    """Render a Python list as a SQL IN-list of single-quoted literals."""
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def resolve_cutoff_ts(widget_value: str) -> str:
    """Return the cutoff timestamp string (mirrors the feature store notebook's resolver).

    If the widget is set, use it verbatim (a bare date reads as IST midnight). Otherwise reproduce the
    v7 cutoff: latest qualifying-order time (in IST) minus PERFORMANCE_DAYS. Orders is touched ONLY for
    this default; the label / anti-join below re-read it explicitly.
    """
    widget_value = (widget_value or "").strip()
    if widget_value:
        return widget_value

    probe = spark.sql(
        f"""
        SELECT CAST(MAX(CAST(created_at AS timestamp)) + INTERVAL {IST_OFFSET_MINUTES} MINUTE AS timestamp) AS max_date
        FROM {ORDERS_TABLE}
        WHERE status IN ({_sql_in_list(QUALIFYING_STATUSES)})
          AND product_category IN ({_sql_in_list(QUALIFYING_PRODUCT_CATEGORIES)})
        """
    ).first()
    cutoff = probe["max_date"] - timedelta(days=PERFORMANCE_DAYS)
    return cutoff.strftime("%Y-%m-%d %H:%M:%S")


def _parse_ts(ts_str: str) -> datetime:
    """Parse 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' into a datetime."""
    return datetime.fromisoformat(ts_str.strip())


cutoff_ts = resolve_cutoff_ts(dbutils.widgets.get("as_of_date"))
# Performance window is (cutoff, cutoff + PERFORMANCE_DAYS] — reproduces v7 (where cutoff = max_date - 90d,
# so performance_end lands on the latest observed qualifying-order time).
performance_end_ts = (_parse_ts(cutoff_ts) + timedelta(days=PERFORMANCE_DAYS)).strftime("%Y-%m-%d %H:%M:%S")

print(f"Feature table   : {FEATURE_TABLE}")
print(f"cutoff/feature_ts : {cutoff_ts}")
print(f"performance_end   : {performance_end_ts}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1b. Spine query — label + first-order eligibility
# MAGIC * **cutoff derivation** → CTE `d` (from the resolved `cutoff_ts` / `performance_end_ts` above).
# MAGIC * **label** → CTE `performance_users`: made a first qualifying order in `(cutoff, performance_end]`.
# MAGIC * **first-order anti-join** → CTE `pre_cutoff_orderers` + `WHERE pco.internal_user_id IS NULL`: drop
# MAGIC   anyone who had already ordered as of the cutoff (this is *first*-order propensity).
# MAGIC * **eligibility window** → `WHERE ob.onboarding_completion_date < cutoff`: carded before the cutoff.
# MAGIC
# MAGIC The `v7` **engagement** gate (active in last 90d OR any recent search) is feature-derived, so it is not
# MAGIC in this SQL — it is applied in Section 2 *against the feature table*, exactly as the feature store
# MAGIC notebook recommends. Status filter kept as `IN ('COMPLETE','CANCELLED')` per the confirmed business rule.

# COMMAND ----------


def build_spine_sql(cutoff: str, performance_end: str, ist: int) -> str:
    """LABEL + ELIGIBILITY only. Emits (internal_user_id, feature_ts, output)."""
    statuses = _sql_in_list(QUALIFYING_STATUSES)
    categories = _sql_in_list(QUALIFYING_PRODUCT_CATEGORIES)
    return f"""
WITH d AS (
    SELECT timestamp '{cutoff}'          AS cutoff,
           timestamp '{performance_end}' AS performance_end
),

-- Carded universe + onboarding (first card-issue) date.
ob AS (
    SELECT user_id AS internal_user_id,
           CAST(MIN(card_issue_time) AS timestamp) AS onboarding_completion_date
    FROM {ONBOARDED_USERS_FACT}
    WHERE card_issue_time IS NOT NULL
    GROUP BY 1
),

-- LABEL: a first qualifying order in (cutoff, performance_end].
performance_users AS (
    SELECT DISTINCT user_id AS internal_user_id
    FROM {ORDERS_TABLE}
    WHERE status IN ({statuses})
      AND product_category IN ({categories})
      AND user_id IS NOT NULL
      AND CAST(created_at AS timestamp) >  (SELECT cutoff FROM d)          - INTERVAL {ist} MINUTE
      AND CAST(created_at AS timestamp) <= (SELECT performance_end FROM d) - INTERVAL {ist} MINUTE
),

-- FIRST-ORDER ANTI-JOIN: users who already had a qualifying order as of the cutoff -> excluded.
pre_cutoff_orderers AS (
    SELECT DISTINCT user_id AS internal_user_id
    FROM {ORDERS_TABLE}
    WHERE status IN ({statuses})
      AND product_category IN ({categories})
      AND user_id IS NOT NULL
      AND CAST(created_at AS timestamp) <= (SELECT cutoff FROM d) - INTERVAL {ist} MINUTE
)

SELECT
    ob.internal_user_id,
    (SELECT cutoff FROM d)                                          AS feature_ts,  -- as-of key = cutoff
    CASE WHEN pf.internal_user_id IS NOT NULL THEN 1 ELSE 0 END     AS output       -- the 0/1 label
FROM ob
LEFT JOIN performance_users   pf  ON pf.internal_user_id  = ob.internal_user_id
LEFT JOIN pre_cutoff_orderers pco ON pco.internal_user_id = ob.internal_user_id
WHERE ob.onboarding_completion_date < (SELECT cutoff FROM d)  -- eligibility window: carded before cutoff
  AND pco.internal_user_id IS NULL                            -- first-order anti-join: never ordered as of cutoff
"""


spine_df = spark.sql(build_spine_sql(cutoff_ts, performance_end_ts, IST_OFFSET_MINUTES))
print(f"Spine columns : {spine_df.columns}")
# .cache() so the count below and the create_training_set pass don't recompute the spine twice.
spine_df = spine_df.cache()
_n_spine = spine_df.count()
_n_pos = spine_df.filter(f"{LABEL} = 1").count()
print(f"Spine rows    : {_n_spine:,}  (positives: {_n_pos:,} = {_n_pos / max(_n_spine, 1):.3%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 2 — Point-in-time training set
# MAGIC
# MAGIC `create_training_set` performs the **as-of** join: for each spine row it looks up the feature-table row
# MAGIC whose `feature_ts` is the latest one `<= ` the spine's `feature_ts`. Since the spine's `feature_ts`
# MAGIC equals the cutoff the table was materialized for, the join is exact and structurally leak-free.
# MAGIC
# MAGIC ### `segment` — included WITH a leakage caveat (audit finding #1)
# MAGIC `segment` is deliberately **not** in the feature table: its source `simple.crud.user_segment_mapping`
# MAGIC is a CRUD current-state snapshot with **no effective-date / as-of column**, so it cannot be joined
# MAGIC point-in-time. We include it in this model's feature set anyway (confirmed business decision), joined
# MAGIC **current-state** below. That means a row's `segment` reflects the user's segment *now*, which may
# MAGIC encode a **post-cutoff reassignment** — i.e. potential target leakage. This is HIGH-severity finding #1.
# MAGIC It is kept for now pending an effective-dated `user_segment_mapping`; the leakage guard in Section 3
# MAGIC explicitly re-audits it, and Section 4 shows why it stays out of the point-in-time `FeatureLookup`.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()

# Point-in-time lookup of the FULL raw feature superset (we are about to SELECT among these).
training_set = fe.create_training_set(
    df=spine_df,
    feature_lookups=[
        FeatureLookup(
            table_name=FEATURE_TABLE,
            lookup_key=ENTITY_KEY,
            timestamp_lookup_key=FEATURE_TS,  # as-of join -> no future leakage from the feature side
            feature_names=STORE_FEATURE_CANDIDATES,
        )
    ],
    label=LABEL,
    exclude_columns=[FEATURE_TS],  # keep the join key out of the feature matrix
)

training_df = training_set.load_df()
print(f"PIT training columns ({len(training_df.columns)}): {training_df.columns}")

# COMMAND ----------

# --- Attach `segment` CURRENT-STATE (leakage caveat above) -------------------
# This is the ONE feature in this model's set the store cannot serve as-of. Joined as a plain
# current-state LEFT JOIN, reproducing v7's `coalesce(usm.segment_name, 'No Segment') AS segment`.
segment_df = spark.sql(
    f"""
    SELECT internal_user_id, segment_name
    FROM {USER_SEGMENT_MAPPING}
    """
)

analysis_df = (
    training_df.join(segment_df, on=ENTITY_KEY, how="left")
    .withColumnRenamed("segment_name", "segment")
    .fillna({"segment": "No Segment"})
)

# --- v7 engagement gate, applied AGAINST THE FEATURE TABLE (post PIT join) ---
# v7's final WHERE kept users active in the last 90d OR with any recent search. Those quantities are
# now feature-table columns, so the gate is evaluated here rather than in the spine SQL (per the feature
# store notebook's guidance). NULL recency == "never" (feature-store NULL contract), so `IS NOT NULL`
# reproduces v7's `<cte>.internal_user_id IS NOT NULL`, and `days_since_last_app_open < 90` reproduces
# v7's `coalesce(days_since_last_app_open, 9999) < 90` (NULL < 90 is not true, same as 9999 < 90).
analysis_df = analysis_df.filter(
    "(days_since_last_app_open < 90) "
    "OR days_since_last_flight_search IS NOT NULL "
    "OR days_since_last_hotel_search IS NOT NULL "
    "OR days_since_last_bus_search IS NOT NULL "
    "OR days_since_last_train_search IS NOT NULL"
)

analysis_df = analysis_df.cache()
_n_analysis = analysis_df.count()
print(f"Analysis (PIT training) set: {_n_analysis:,} rows, {len(analysis_df.columns)} columns")
print(f"Feature set for selection  : {STORE_FEATURE_CANDIDATES + ['segment']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 3 — Feature selection (on the PIT training set)
# MAGIC
# MAGIC A staged pipeline, logged to MLflow:
# MAGIC 1. **Screen** — drop near-constant / low-variance and high-missingness features.
# MAGIC 2. **Redundancy** — prune multicollinearity (Spearman clusters; keep the stronger representative; VIF reported).
# MAGIC 3. **Relevance** — univariate mutual information vs the label.
# MAGIC 4. **Model-based** — XGBoost gain + SHAP importance, kept stable across CV folds.
# MAGIC 5. **Leakage guard** — audit any feature with a suspiciously high SOLO AUC (explicitly re-audit `segment`).
# MAGIC 6. **Log** — final selected list + all diagnostics to MLflow, inside one `mlflow.start_run()`.
# MAGIC
# MAGIC Imputation / one-hot below is **selector-only** (MI, correlation and tree/SHAP models need finite
# MAGIC numerics); it is never persisted and is not the production encoding, which lives in the model pipeline.

# COMMAND ----------

# --- Materialize the selection frame on the driver (single-node selectors) ---
import numpy as np
import pandas as pd

if SELECTION_SAMPLE_FRACTION:
    _selection_sdf = analysis_df.sample(fraction=SELECTION_SAMPLE_FRACTION, seed=RANDOM_STATE)
    print(f"Sampling {SELECTION_SAMPLE_FRACTION:.0%} of the PIT training set for selection.")
else:
    _selection_sdf = analysis_df

pdf = _selection_sdf.toPandas()
y = pdf[LABEL].astype(int).values

NUMERIC_CANDIDATES = [c for c in STORE_FEATURE_CANDIDATES if c in pdf.columns]
CATEGORICAL_CANDIDATES = [c for c in ["segment"] if c in pdf.columns]
ALL_CANDIDATES = NUMERIC_CANDIDATES + CATEGORICAL_CANDIDATES

print(f"Selection frame  : {pdf.shape[0]:,} rows, base rate {y.mean():.3%}")
print(f"Candidate features ({len(ALL_CANDIDATES)}): {ALL_CANDIDATES}")

# COMMAND ----------

# --- Shared selector helpers -------------------------------------------------
import re
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _solo_auc_numeric(x, y_arr):
    """Univariate ranking AUC of a numeric feature (median-imputed, orientation-agnostic)."""
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)):
        return 0.5
    med = np.nanmedian(x)
    x = np.where(np.isnan(x), med, x)
    if len(np.unique(x)) < 2 or len(np.unique(y_arr)) < 2:
        return 0.5
    a = roc_auc_score(y_arr, x)
    return max(a, 1.0 - a)


def _oof_target_encode_auc(cat_series, y_arr, n_splits, random_state):
    """SOLO AUC of a categorical via out-of-fold target encoding (avoids trivial in-sample leakage)."""
    s = cat_series.astype("object").fillna("No Segment").reset_index(drop=True)
    y_s = pd.Series(np.asarray(y_arr)).reset_index(drop=True)
    if y_s.nunique() < 2:
        return 0.5
    oof = np.full(len(y_s), y_s.mean(), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, va in skf.split(s, y_s):
        means = y_s.iloc[tr].groupby(s.iloc[tr]).mean()
        oof[va] = s.iloc[va].map(means).fillna(y_s.iloc[tr].mean()).values
    a = roc_auc_score(y_s.values, oof)
    return max(a, 1.0 - a)


def _agg_importance(imp_by_col, origin):
    """Sum per-column importances back onto their originating feature (e.g. segment one-hot -> 'segment')."""
    agg = {}
    for col, val in imp_by_col.items():
        f = origin.get(col, col)
        agg[f] = agg.get(f, 0.0) + float(val)
    return agg


def _build_model_matrix(pdf_in, features, top_cats=8):
    """Diagnostic (selector-only) matrix: numeric raw (NaN kept for XGBoost) + one-hot top categories."""
    parts, origin = [], {}
    for c in features:
        s = pdf_in[c]
        if pd.api.types.is_numeric_dtype(s):
            parts.append(s.astype(float).rename(c))  # keep NaN — XGBoost handles missing natively
            origin[c] = c
        else:
            vals = s.astype("object").fillna("No Segment")
            keep_cats = vals.value_counts().head(top_cats).index
            vals = vals.where(vals.isin(keep_cats), "Other")
            dummies = pd.get_dummies(vals, prefix=c, dtype=float)
            dummies.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", str(dc)) for dc in dummies.columns]
            for dc in dummies.columns:
                origin[dc] = c
            parts.append(dummies)
    X = pd.concat(parts, axis=1)
    return X, origin

# COMMAND ----------

# --- Stage 1: SCREEN (near-constant / low-variance + high-missingness) -------
def screen_features(pdf_in, features, missing_thresh, dominant_thresh):
    n = len(pdf_in)
    rows, keep = [], []
    for c in features:
        s = pdf_in[c]
        miss = float(s.isna().mean())
        vc = s.value_counts(dropna=True)
        nuniq = int(s.nunique(dropna=True))
        dominant = float(vc.iloc[0] / n) if len(vc) and n else 1.0
        drop_missing = miss > missing_thresh
        drop_const = (nuniq <= 1) or (dominant > dominant_thresh)
        reasons = []
        if drop_missing:
            reasons.append(f"missing>{missing_thresh}")
        if drop_const:
            reasons.append("near_constant")
        dropped = bool(drop_missing or drop_const)
        rows.append({
            "feature": c, "missing_frac": round(miss, 4), "n_unique": nuniq,
            "dominant_frac": round(dominant, 4), "dropped": dropped, "reason": ";".join(reasons),
        })
        if not dropped:
            keep.append(c)
    return keep, pd.DataFrame(rows).sort_values(["dropped", "feature"]).reset_index(drop=True)


kept_screen, screen_df = screen_features(pdf, ALL_CANDIDATES, SCREEN_MISSING_THRESH, SCREEN_DOMINANT_THRESH)
print(f"Stage 1 SCREEN: kept {len(kept_screen)}/{len(ALL_CANDIDATES)} features")
print(screen_df.to_string(index=False))

# COMMAND ----------

# --- Stage 2: REDUNDANCY (Spearman clusters; VIF reported) -------------------
def prune_redundancy(pdf_in, features, y_arr, corr_thresh):
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(pdf_in[c])]
    non_numeric = [c for c in features if c not in numeric]
    if len(numeric) < 2:
        return features, pd.DataFrame(), []
    X = pdf_in[numeric].astype(float)
    X = X.fillna(X.median())
    corr = X.corr(method="spearman").abs()
    strength = {c: _solo_auc_numeric(X[c].values, y_arr) for c in numeric}
    ordered = sorted(numeric, key=lambda c: strength[c], reverse=True)  # keep the stronger of a pair
    kept, dropped = [], []
    for c in ordered:
        if any(corr.loc[c, k] > corr_thresh for k in kept):
            dropped.append(c)
        else:
            kept.append(c)
    return kept + non_numeric, corr, dropped


def compute_vif(pdf_in, features):
    """Variance Inflation Factor for the surviving numeric features (reported, not auto-pruned)."""
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(pdf_in[c])]
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except Exception as exc:  # statsmodels may be absent on some runtimes
        print(f"VIF skipped ({exc}); Spearman clustering already handled multicollinearity.")
        return pd.DataFrame()
    X = pdf_in[numeric].astype(float)
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 0]  # VIF is undefined for constant columns
    vif_rows = [
        {"feature": col, "vif": round(float(variance_inflation_factor(X.values, i)), 3)}
        for i, col in enumerate(X.columns)
    ]
    return pd.DataFrame(vif_rows).sort_values("vif", ascending=False).reset_index(drop=True)


kept_redundancy, corr_matrix, dropped_corr = prune_redundancy(pdf, kept_screen, y, CORR_THRESH)
vif_df = compute_vif(pdf, kept_redundancy)
print(f"Stage 2 REDUNDANCY: kept {len(kept_redundancy)} features; pruned {dropped_corr}")
if not vif_df.empty:
    print(vif_df.to_string(index=False))

# COMMAND ----------

# --- Stage 3: RELEVANCE (univariate mutual information) ----------------------
from sklearn.feature_selection import mutual_info_classif


def relevance_mi(pdf_in, features, y_arr, random_state):
    cols, mats, discrete = list(features), [], []
    for c in cols:
        s = pdf_in[c]
        if pd.api.types.is_numeric_dtype(s):
            mats.append(s.fillna(s.median()).astype(float).values)
            discrete.append(False)
        else:
            mats.append(s.astype("category").cat.codes.replace(-1, 0).astype(float).values)
            discrete.append(True)
    X = np.column_stack(mats)
    mi = mutual_info_classif(X, y_arr, discrete_features=discrete, random_state=random_state)
    return pd.DataFrame({"feature": cols, "mutual_info": np.round(mi, 6)}).sort_values(
        "mutual_info", ascending=False
    ).reset_index(drop=True)


mi_df = relevance_mi(pdf, kept_redundancy, y, RANDOM_STATE)
print("Stage 3 RELEVANCE (mutual information vs output):")
print(mi_df.to_string(index=False))

# COMMAND ----------

# --- Stage 4: MODEL-BASED (XGBoost gain + SHAP, stable across CV folds) ------
import xgboost as xgb


def _shap_importance(model, X_sample, origin):
    """Mean |SHAP| per feature (best-effort; returns None if shap unavailable)."""
    try:
        import shap
    except Exception as exc:
        print(f"SHAP unavailable ({exc}); falling back to gain-only for this fold.")
        return None
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):  # some shap versions return [neg, pos] for binary
        sv = sv[-1]
    mean_abs = np.abs(sv).mean(axis=0)
    return _agg_importance(dict(zip(X_sample.columns, mean_abs)), origin)


def model_based_selection(pdf_in, features, y_arr, n_splits, top_k, random_state):
    X, origin = _build_model_matrix(pdf_in, features)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    gains = {f: [] for f in features}
    shaps = {f: [] for f in features}
    gain_topk_counts = {f: 0 for f in features}   # per-fold appearances in the top-K by GAIN
    shap_topk_counts = {f: 0 for f in features}   # per-fold appearances in the top-K by mean |SHAP|
    shap_folds = 0                                # folds in which SHAP was actually computed
    for tr, va in skf.split(X, y_arr):
        model = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=random_state, n_jobs=-1,
        )
        model.fit(X.iloc[tr], y_arr[tr])
        # --- gain stability ---
        raw_gain = model.get_booster().get_score(importance_type="gain")
        fold_gain = _agg_importance({col: raw_gain.get(col, 0.0) for col in X.columns}, origin)
        for f in features:
            gains[f].append(fold_gain.get(f, 0.0))
        for f in sorted(features, key=lambda z: fold_gain.get(z, 0.0), reverse=True)[:top_k]:
            gain_topk_counts[f] += 1
        # --- SHAP stability (computed IN PARALLEL to gain, on the held-out fold) ---
        sample = X.iloc[va]
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=random_state)
        fold_shap = _shap_importance(model, sample, origin)
        if fold_shap is not None:
            shap_folds += 1
            for f in features:
                shaps[f].append(fold_shap.get(f, 0.0))
            for f in sorted(features, key=lambda z: fold_shap.get(z, 0.0), reverse=True)[:top_k]:
                shap_topk_counts[f] += 1
    imp_df = pd.DataFrame({
        "feature": features,
        "mean_gain": [round(float(np.mean(gains[f])), 4) if gains[f] else 0.0 for f in features],
        "gain_topk_folds": [gain_topk_counts[f] for f in features],
        "mean_abs_shap": [round(float(np.mean(shaps[f])), 6) if shaps[f] else np.nan for f in features],
        "shap_topk_folds": [shap_topk_counts[f] for f in features],
    })
    # A feature is "stable" under a metric if it lands in that metric's top-K in a MAJORITY of folds.
    majority_gain = (n_splits // 2) + 1
    gain_stable = {
        f: (gain_topk_counts[f] >= majority_gain) and bool(gains[f] and np.mean(gains[f]) > 0)
        for f in features
    }
    if shap_folds > 0:
        majority_shap = (shap_folds // 2) + 1
        shap_stable = {
            f: (shap_topk_counts[f] >= majority_shap) and bool(shaps[f] and np.mean(shaps[f]) > 0)
            for f in features
        }
        # COMBINED RULE = INTERSECTION of gain-stability AND SHAP-stability.
        # WHY intersection (not union / rank-average): gain measures how much a feature improves the tree
        # SPLITS; SHAP measures its marginal contribution to individual PREDICTIONS. These are independent
        # views, so requiring a feature to be stable under BOTH keeps only the ones the two signals AGREE
        # on across folds — the conservative choice that guards against metric-specific artifacts (a
        # feature XGBoost splits on often but that barely moves predictions, or vice-versa). This is the
        # mechanism by which SHAP actually DRIVES final_selected, not just decorates the report.
        rule = "gain_and_shap_intersection"
        selected = [f for f in features if gain_stable[f] and shap_stable[f]]
    else:
        # SHAP unavailable on this runtime -> documented fallback to gain-only stability.
        majority_shap = None
        shap_stable = {f: False for f in features}
        rule = "gain_only_fallback_shap_unavailable"
        selected = [f for f in features if gain_stable[f]]
    imp_df["gain_stable"] = [bool(gain_stable[f]) for f in features]
    imp_df["shap_stable"] = [bool(shap_stable[f]) for f in features]
    imp_df["selected"] = [f in set(selected) for f in features]
    imp_df = imp_df.sort_values(
        ["selected", "shap_topk_folds", "gain_topk_folds", "mean_gain"], ascending=False
    ).reset_index(drop=True)
    return imp_df, selected, majority_gain, majority_shap, shap_folds, rule


model_imp_df, model_selected, _maj_gain, _maj_shap, _shap_folds, _sel_rule = model_based_selection(
    pdf, kept_redundancy, y, CV_SPLITS, MODEL_TOPK, RANDOM_STATE
)
_shap_desc = (
    f"AND shap-stable >= {_maj_shap}/{_shap_folds} folds"
    if _shap_folds else "(SHAP unavailable this run -> gain-only fallback)"
)
print(
    f"Stage 4 MODEL-BASED [{_sel_rule}]: gain-stable >= {_maj_gain}/{CV_SPLITS} folds {_shap_desc} "
    f"-> {len(model_selected)} selected"
)
print(model_imp_df.to_string(index=False))

# COMMAND ----------

# --- Stage 5: LEAKAGE GUARD (solo AUC audit; explicit segment call-out) ------
def leakage_guard(pdf_in, features, y_arr, auc_thresh, n_splits, random_state):
    rows = []
    for c in features:
        s = pdf_in[c]
        if pd.api.types.is_numeric_dtype(s):
            auc = _solo_auc_numeric(s.values, y_arr)
        else:
            auc = _oof_target_encode_auc(s, y_arr, n_splits, random_state)
        rows.append({"feature": c, "solo_auc": round(float(auc), 4), "flag": bool(auc >= auc_thresh)})
    return pd.DataFrame(rows).sort_values("solo_auc", ascending=False).reset_index(drop=True)


leak_df = leakage_guard(pdf, kept_redundancy, y, LEAKAGE_AUC_THRESH, CV_SPLITS, RANDOM_STATE)
flagged = leak_df[leak_df["flag"]]["feature"].tolist()
print(f"Stage 5 LEAKAGE GUARD (solo AUC >= {LEAKAGE_AUC_THRESH} flagged): {flagged or 'none'}")
print(leak_df.to_string(index=False))

# Explicit re-audit of `segment` (finding #1): flag if present, regardless of threshold.
if "segment" in set(leak_df["feature"]):
    seg_auc = float(leak_df.loc[leak_df["feature"] == "segment", "solo_auc"].iloc[0])
    print(
        f"\n[FINDING #1] segment solo AUC = {seg_auc:.4f}. segment is joined CURRENT-STATE (no as-of source), "
        f"so a high solo AUC may reflect post-cutoff reassignment leaking the label rather than genuine "
        f"pre-cutoff signal. Retained WITH caveat pending an effective-dated user_segment_mapping; keep it "
        f"OUT of the point-in-time FeatureLookup (Section 4) until then."
    )

# COMMAND ----------

# --- Stage 6: FINAL selection + MLflow logging (whole run wrapped) -----------
import json
import os
import tempfile

import mlflow

# Final selection = model-based stable set. `segment` may be present WITH its finding-#1 caveat (kept per
# the confirmed business decision); the leakage guard's audit is logged alongside so the caveat travels
# with the artifact. This step logs only — no model is trained/registered here.
final_selected = list(model_selected)
final_store_features = [f for f in final_selected if f in STORE_FEATURE_CANDIDATES]
final_non_store = [f for f in final_selected if f not in STORE_FEATURE_CANDIDATES]


def _log_df_artifact(df, name, artifact_dir):
    path = os.path.join(artifact_dir, name)
    df.to_csv(path, index=False)
    mlflow.log_artifact(path)


with mlflow.start_run(run_name="fop_feature_selection") as run:
    mlflow.set_tags({
        "project": "first_order_propensity",
        "stage": "feature_selection",
        "feature_table": FEATURE_TABLE,
    })
    mlflow.log_params({
        "as_of_date": cutoff_ts,
        "performance_end": performance_end_ts,
        "performance_days": PERFORMANCE_DAYS,
        "n_candidates": len(ALL_CANDIDATES),
        "screen_missing_thresh": SCREEN_MISSING_THRESH,
        "screen_dominant_thresh": SCREEN_DOMINANT_THRESH,
        "corr_thresh": CORR_THRESH,
        "cv_splits": CV_SPLITS,
        "model_topk": MODEL_TOPK,
        "leakage_auc_thresh": LEAKAGE_AUC_THRESH,
        "n_selected": len(final_selected),
        # Model-based (Stage 4): the combined gain+SHAP rule that produced final_selected.
        "selection_rule": _sel_rule,
        "majority_gain_folds": _maj_gain,
        "majority_shap_folds": str(_maj_shap),  # None in the SHAP-unavailable fallback
        "shap_folds": _shap_folds,
    })
    mlflow.log_metrics({
        "n_rows": float(len(pdf)),
        "base_rate": float(y.mean()),
        "n_after_screen": float(len(kept_screen)),
        "n_after_redundancy": float(len(kept_redundancy)),
        "n_selected": float(len(final_selected)),
        "n_leakage_flagged": float(len(flagged)),
        # SHAP demonstrably participates: count of features stable under each metric across folds.
        "n_gain_stable": float(int(model_imp_df["gain_stable"].sum())),
        "n_shap_stable": float(int(model_imp_df["shap_stable"].sum())),
    })

    _artifact_dir = tempfile.mkdtemp()
    _log_df_artifact(screen_df, "01_screen.csv", _artifact_dir)
    if not corr_matrix.empty:
        _log_df_artifact(corr_matrix.reset_index().rename(columns={"index": "feature"}),
                         "02_spearman_corr.csv", _artifact_dir)
    if not vif_df.empty:
        _log_df_artifact(vif_df, "02_vif.csv", _artifact_dir)
    _log_df_artifact(mi_df, "03_mutual_info.csv", _artifact_dir)
    _log_df_artifact(model_imp_df, "04_model_importance.csv", _artifact_dir)
    _log_df_artifact(leak_df, "05_leakage_audit.csv", _artifact_dir)

    _selection_json = {
        "as_of_date": cutoff_ts,
        "selection_rule": _sel_rule,  # combined gain+SHAP rule behind final_selected (Stage 4)
        "final_selected": final_selected,
        "final_store_features": final_store_features,
        "final_non_store_features": final_non_store,  # e.g. ['segment'] — current-state, finding #1
        "gain_stable_features": model_imp_df[model_imp_df["gain_stable"]]["feature"].tolist(),
        "shap_stable_features": model_imp_df[model_imp_df["shap_stable"]]["feature"].tolist(),
        "dropped_screen": screen_df[screen_df["dropped"]]["feature"].tolist(),
        "dropped_redundancy": dropped_corr,
        "leakage_flagged": flagged,
    }
    _sel_path = os.path.join(_artifact_dir, "selected_features.json")
    with open(_sel_path, "w") as fh:
        json.dump(_selection_json, fh, indent=2)
    mlflow.log_artifact(_sel_path)

    print(f"MLflow run   : {run.info.run_id}")
    print(f"Selected ({len(final_selected)}): {final_selected}")
    print(f"  store-resident (PIT-lookupable): {final_store_features}")
    print(f"  non-store (current-state)      : {final_non_store}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 4 — Wire the selected features back into the lookup
# MAGIC
# MAGIC The selected **store-resident** subset feeds straight back into a `FeatureLookup`, matching the hook
# MAGIC the feature store notebook demonstrates (`feature_names=[...]` on the training set — never by trimming
# MAGIC the shared table). `segment`, if selected, stays OUT of this point-in-time lookup: it has no as-of
# MAGIC source (finding #1), so it must keep being joined current-state until an effective-dated
# MAGIC `user_segment_mapping` exists. This cell builds the lookup; it does not train or register anything.

# COMMAND ----------

from databricks.feature_engineering import FeatureLookup

# The SELECTED subset (Section 3) fed back in — exactly the hook the feature store notebook shows.
model_feature_lookups = [
    FeatureLookup(
        table_name=FEATURE_TABLE,
        lookup_key=ENTITY_KEY,
        timestamp_lookup_key=FEATURE_TS,
        feature_names=final_store_features,  # <-- the selected store-resident subset
    )
]

print("Final point-in-time FeatureLookup (selected subset):")
print(f"  table_name           : {FEATURE_TABLE}")
print(f"  lookup_key           : {ENTITY_KEY}")
print(f"  timestamp_lookup_key : {FEATURE_TS}")
print(f"  feature_names        : {final_store_features}")
if final_non_store:
    print(
        f"\nNOTE: {final_non_store} selected but NOT in the lookup — no as-of source (finding #1). "
        f"Keep joining current-state (see Section 2) until an effective-dated source exists."
    )

# A downstream model pipeline would rebuild the training set from the SELECTED subset like so:
# selected_training_set = fe.create_training_set(
#     df=spine_df, feature_lookups=model_feature_lookups, label=LABEL, exclude_columns=[FEATURE_TS],
# )
# selected_training_df = selected_training_set.load_df()   # then encode + train DOWNSTREAM (out of scope here)
