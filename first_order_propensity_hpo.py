# Databricks notebook source
# MAGIC %md
# MAGIC # Scapia — First-Order-Propensity — Hyperparameter Optimization (HPO)
# MAGIC
# MAGIC Companion to `first_order_propensity_spine_and_selection.py` and the shared
# MAGIC `first_order_propensity_user_features.py` feature-store notebook. This notebook runs the
# MAGIC **hyperparameter search** for the First-Order-Propensity XGBoost model on the **already-selected**
# MAGIC feature subset — it does NOT re-run feature selection.
# MAGIC
# MAGIC ## The one HPO mechanism: `MlflowSparkStudy` (distributed Optuna)
# MAGIC Trials are distributed across Spark **executors** via `MlflowSparkStudy`, and every trial is persisted
# MAGIC as an MLflow run via `MlflowStorage`. There is **no single-node Optuna fallback** — this is the sole,
# MAGIC deliberately-distributed HPO path (per the explicit build request).
# MAGIC
# MAGIC ## What this notebook does
# MAGIC 1. **Training set** — LOADS the selected feature list from the upstream feature-selection MLflow run
# MAGIC    (`selected_features.json` -> `final_store_features` + `final_non_store_features`), rebuilds the spine
# MAGIC    (label + eligibility) and the point-in-time `create_training_set` join on those selected store
# MAGIC    features, and joins `segment` current-state ONLY if selection retained it (finding-#1 leakage caveat
# MAGIC    preserved). Feature selection is NOT re-run and NOT guessed here. SQL/Spark only — no connector,
# MAGIC    no PAT, no egress.
# MAGIC 2. **Split** — stratified train / val / test. HPO uses train + val only; **test is held out and touched
# MAGIC    exactly once** at the end.
# MAGIC 3. **HPO** — distributed Optuna over the v7 XGBoost search space, early stopping (so `n_estimators` is
# MAGIC    NOT tuned), MedianPruner pruning with per-boosting-round intermediate values.
# MAGIC 4. **Champion fit + honest eval** — refit on the best params, evaluate ONCE on the held-out test at the
# MAGIC    true population base rate; report ROC-AUC, top-decile lift, F2; log params + metrics to MLflow.
# MAGIC
# MAGIC ## Two methodology fixes baked in (from the audit)
# MAGIC * **(a) No double imbalance correction.** Default: keep v7's 3:1 training-fold undersampling AND fix
# MAGIC   `scale_pos_weight = 1` (dropped from the search space). A one-line toggle instead tunes
# MAGIC   `scale_pos_weight` WITHOUT undersampling. Doing both at once double-corrects the prior and distorts
# MAGIC   predicted probabilities (audit Tier 1 #2).
# MAGIC * **(b) Optimize the BUSINESS metric on a POPULATION-rate validation fold.** The objective is
# MAGIC   top-decile lift / F2 / ROC-AUC (selectable) computed on a validation fold left at the TRUE prior —
# MAGIC   never accuracy, never on rebalanced data (audit Tier 1 #5). Only the training fold is undersampled.
# MAGIC
# MAGIC ## Compute requirement
# MAGIC `MlflowSparkStudy` only parallelizes when the cluster has **multiple executors** (a multi-node cluster).
# MAGIC On a single-node cluster the trials still run, but serially. Requires **MLflow 3.0**, which is
# MAGIC pre-installed on **Databricks Runtime 17.0 ML and above**; `optuna` and `xgboost` also ship with the ML
# MAGIC runtime. See the API-grounding cell below for the exact doc pages.
# MAGIC
# MAGIC ## Out of scope (noted as follow-ons in the closing cell)
# MAGIC No UC model registration, no probability calibration, no model serving, no feature-table
# MAGIC materialization, and no single-node Optuna fallback.

# COMMAND ----------

# MAGIC %md
# MAGIC ## API grounding — official doc pages this notebook was written against
# MAGIC The `MlflowSparkStudy` / `MlflowStorage` surface below was verified against these pages (retrieved
# MAGIC 2026-07-17). URLs are kept as comments in the "Run the distributed study" cell too.
# MAGIC
# MAGIC * Databricks — "Hyperparameter tuning with Optuna" (AWS):
# MAGIC   `https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/optuna`
# MAGIC * Azure Databricks — "Hyperparameter tuning with Optuna" (same content + full parameter tables):
# MAGIC   `https://learn.microsoft.com/en-us/azure/databricks/machine-learning/automl-hyperparam-tuning/optuna`
# MAGIC * MLflow source (constructor / optimize signatures + default optimization direction):
# MAGIC   `https://github.com/mlflow/mlflow/blob/master/mlflow/pyspark/optuna/study.py`
# MAGIC
# MAGIC Confirmed facts used below:
# MAGIC * Imports: `from mlflow.optuna.storage import MlflowStorage`,
# MAGIC   `from mlflow.pyspark.optuna.study import MlflowSparkStudy`.
# MAGIC * `MlflowStorage(experiment_id=..., name=..., batch_flush_interval=1.0, batch_size_threshold=100)`.
# MAGIC * `MlflowSparkStudy(study_name, storage, sampler=TPESampler(), pruner=MedianPruner(), mlflow_tracking_uri=None)`
# MAGIC   — `MedianPruner` is the documented default pruner.
# MAGIC * `MlflowSparkStudy.optimize(func, n_trials=None, timeout=None, n_jobs=-1, catch=(), callbacks=None)`
# MAGIC   — `n_jobs=-1` matches the number of Spark tasks (executor parallelism).
# MAGIC * **Direction:** `MlflowSparkStudy.__init__` exposes **no `direction` argument**; internally it calls
# MAGIC   `optuna.create_study(...)` with no direction, so the study **MINIMIZES** (Optuna's default). This
# MAGIC   notebook therefore returns the **negated** business metric from the objective (and reports negated
# MAGIC   per-round values), so minimizing the negative == maximizing the metric. See the objective cell.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Dependencies
# MAGIC MLflow 3.0, Optuna, XGBoost and scikit-learn all ship with Databricks Runtime 17.0 ML+. The install
# MAGIC below just pins recent versions so `mlflow.optuna` / `mlflow.pyspark.optuna` are importable and the
# MAGIC `create_training_set` point-in-time join is available. Safe to skip on a current ML Runtime.

# COMMAND ----------

# MAGIC %pip install -U "mlflow>=3.0" optuna databricks-feature-engineering xgboost scikit-learn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. CONFIG
# MAGIC Everything routinely changed lives in this block or in the widgets in the next cell. Names and source
# MAGIC tables are kept **verbatim** from the spine / feature-store notebooks so the point-in-time join lines up.

# COMMAND ----------

# ---------------------------------------------------------------------------
# CONFIG — edit here
# ---------------------------------------------------------------------------

# Feature table produced by first_order_propensity_user_features.py (verbatim).
CATALOG = "mlops_data_science"
SCHEMA = "features"
TABLE = "first_order_propensity_user_features"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"

# Entity key + point-in-time key + label — must match the feature table / spine exactly.
ENTITY_KEY = "internal_user_id"
FEATURE_TS = "feature_ts"
LABEL = "output"

# --- Source tables for the spine (read-only; label + eligibility ONLY) ------
ORDERS_TABLE = "rds_main.scapiadb.orders"                    # label + first-order anti-join
ONBOARDED_USERS_FACT = "simple.crud.onboarded_users_fact"    # carded universe + onboarding date
# `segment` is NOT in the feature table (finding #1 — omitted there for lack of an as-of column).
# It is sourced here from its original CRUD table for the model's feature set (see the caveat in Section 2).
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

# --- SELECTED feature subset --------------------------------------------------
# There is NO hardcoded/guessed feature list here. The selected subset is LOADED from the upstream
# feature-selection MLflow run (its `selected_features.json` artifact) — see the widget cell. The optional
# manual-candidate path is opt-in only and clearly labelled as candidates, not a selection.
#
# Only `segment` is supported as a non-store (current-state) feature in this pipeline. If the selection run
# lists other non-store features, they are reported and skipped (no encoder exists for them here).
SUPPORTED_NON_STORE_FEATURES = ["segment"]

# --- Reproducibility / split -------------------------------------------------
RANDOM_STATE = 42
TEST_FRACTION = 0.20   # held out, touched exactly once at the end
VAL_FRACTION = 0.25    # fraction of the (non-test) remainder used as the HPO validation fold (-> ~60/20/20)

# --- Imbalance handling (methodology fix (a): NO double correction) ---------
# v7 undersamples the majority (negatives) in TRAIN to 3:1 (neg:pos). Kept as the default.
TARGET_NEG_PER_POS = 3

# --- XGBoost / early stopping (methodology: do NOT tune n_estimators) -------
# Early stopping on the validation watchlist chooses the effective number of rounds. WATCH_METRIC is the
# per-round metric used for BOTH early stopping and the pruner's intermediate values. 'auc' (ROC-AUC) is a
# ranking metric aligned with the ranking-oriented business objective (v7's classifier declared 'aucpr';
# its selection was driven by an F2 CV scorer, so the eval_metric was not the selection signal).
MAX_BOOST_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 50
WATCH_METRIC = "auc"

# Bump this whenever the search space (param ranges / which params are tuned) changes. It is part of the
# study-name fingerprint (below), so a search-space change starts a FRESH study instead of resuming an
# incompatible one.
SEARCH_SPACE_VERSION = "v1"

# Optional driver-memory down-sampling of the pandas frame (mirrors the sibling notebook). None = full set.
SAMPLE_FRACTION = None

# Cap on distinct one-hot categories for `segment` (matches the sibling notebook's _build_model_matrix).
SEGMENT_TOP_CATS = 8

# ---------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1a. Widgets — runtime parameters + LOAD the selected feature list
# MAGIC `as_of_date` MUST equal the as-of date the feature table was materialized for (so the PIT join finds a
# MAGIC row). For reproducible runs always pass it explicitly; blank auto-derives the v7 reference cutoff (drifts
# MAGIC as new orders arrive). The imbalance-strategy widget is the one-line toggle for methodology fix (a).
# MAGIC
# MAGIC **Feature list is consumed, not guessed.** `features_source="selection_run"` (default) LOADS
# MAGIC `final_store_features` + `final_non_store_features` from the feature-selection run named by the required
# MAGIC `selection_run_id` widget; `segment` is included ONLY if selection retained it. If `selection_run_id`
# MAGIC is blank the cell **fails fast**. `features_source="manual_candidates"` is an explicit opt-in escape
# MAGIC hatch that treats `features_override` as CANDIDATES (not a selection) and prints a loud warning.

# COMMAND ----------

import json

import mlflow

dbutils.widgets.text(
    "as_of_date",
    "",
    "Cutoff / as-of (YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS', IST). Blank = auto-derive v7 reference cutoff.",
)
dbutils.widgets.text(
    "selection_run_id",
    "",
    "REQUIRED (features_source=selection_run): MLflow run_id of the feature-selection run to consume.",
)
dbutils.widgets.dropdown(
    "features_source",
    "selection_run",
    ["selection_run", "manual_candidates"],
    "Where the feature list comes from: the upstream selection run (default) OR a manual candidate list.",
)
dbutils.widgets.text(
    "features_override",
    "",
    "manual_candidates ONLY: comma-separated CANDIDATE features. Ignored when features_source=selection_run.",
)
dbutils.widgets.dropdown(
    "include_segment", "yes", ["yes", "no"],
    "manual_candidates ONLY: include `segment` (current-state; finding-#1 caveat). selection_run derives this.",
)
dbutils.widgets.text("n_trials", "64", "Number of Optuna trials.")
dbutils.widgets.text("n_jobs", "-1", "Parallel trials across Spark executors (-1 = match number of tasks).")
dbutils.widgets.dropdown(
    "imbalance_strategy",
    "undersample_fixed_spw1",
    ["undersample_fixed_spw1", "tune_spw_no_undersample"],
    "Fix (a): undersample 3:1 & fix scale_pos_weight=1 (default) OR tune scale_pos_weight & no undersampling.",
)
dbutils.widgets.dropdown(
    "hpo_objective_metric",
    "top_decile_lift",
    ["top_decile_lift", "roc_auc", "f2"],
    "Business metric to optimize on the POPULATION-rate validation fold (fix (b)).",
)

# Read the simple runtime knobs.
N_TRIALS = int(dbutils.widgets.get("n_trials"))
N_JOBS = int(dbutils.widgets.get("n_jobs"))
IMBALANCE_STRATEGY = dbutils.widgets.get("imbalance_strategy")
HPO_OBJECTIVE_METRIC = dbutils.widgets.get("hpo_objective_metric")
FEATURES_SOURCE = dbutils.widgets.get("features_source")
SELECTION_RUN_ID = dbutils.widgets.get("selection_run_id").strip()


def load_selected_features_from_run(run_id):
    """Load the SELECTED feature list from a feature-selection run's `selected_features.json` artifact.

    Returns (store_features, non_store_features) exactly as the selection notebook wrote them — no guessing.
    """
    local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="selected_features.json")
    with open(local) as fh:
        sel = json.load(fh)
    store = list(sel.get("final_store_features") or [])
    non_store = list(sel.get("final_non_store_features") or [])
    if not store and not non_store:
        raise ValueError(
            f"selected_features.json in run {run_id} has empty final_store_features AND "
            f"final_non_store_features — nothing selected to train on."
        )
    return store, non_store


if FEATURES_SOURCE == "selection_run":
    if not SELECTION_RUN_ID:
        raise ValueError(
            "features_source='selection_run' but `selection_run_id` widget is empty. Set it to the MLflow "
            "run_id of the feature-selection run (the one that logged selected_features.json). Refusing to "
            "guess a feature subset — FAILING FAST."
        )
    SELECTED_STORE_FEATURES, _final_non_store = load_selected_features_from_run(SELECTION_RUN_ID)
    # Only `segment` is supported as a non-store (current-state) feature in this pipeline.
    _unsupported_non_store = [f for f in _final_non_store if f not in SUPPORTED_NON_STORE_FEATURES]
    if _unsupported_non_store:
        print(f"WARNING: selection listed non-store features with no encoder here -> SKIPPED: {_unsupported_non_store}")
    INCLUDE_SEGMENT = "segment" in _final_non_store
    print(f"Loaded SELECTED features from run {SELECTION_RUN_ID}:")
    print(f"  final_store_features    : {SELECTED_STORE_FEATURES}")
    print(f"  final_non_store_features: {_final_non_store}  -> include_segment={INCLUDE_SEGMENT}")
else:
    # Opt-in escape hatch — these are CANDIDATES, not a selection.
    SELECTED_STORE_FEATURES = [c.strip() for c in dbutils.widgets.get("features_override").split(",") if c.strip()]
    if not SELECTED_STORE_FEATURES:
        raise ValueError(
            "features_source='manual_candidates' but `features_override` is empty. Provide a comma-separated "
            "candidate list, or switch features_source back to 'selection_run'. FAILING FAST."
        )
    INCLUDE_SEGMENT = dbutils.widgets.get("include_segment") == "yes"
    print("=" * 88)
    print("!!! WARNING: THESE ARE CANDIDATES, NOT A SELECTION !!!")
    print("!!! features_source='manual_candidates' — the feature list below was typed by hand and has NOT")
    print("!!! been through the feature-selection / leakage-guard pipeline. Use ONLY for illustration.")
    print("=" * 88)
    print(f"  candidate store features: {SELECTED_STORE_FEATURES}")
    print(f"  include_segment         : {INCLUDE_SEGMENT}")

if not SELECTED_STORE_FEATURES and not INCLUDE_SEGMENT:
    raise ValueError("Resolved feature set is empty (no store features and segment not included). FAILING FAST.")

# `segment` is the one selected feature with no as-of source (finding #1) — joined current-state below.
CATEGORICAL_FEATURES = ["segment"] if INCLUDE_SEGMENT else []

print(f"Feature table          : {FEATURE_TABLE}")
print(f"Selected store features: {SELECTED_STORE_FEATURES}")
print(f"Include segment        : {INCLUDE_SEGMENT}")
print(f"Imbalance strategy     : {IMBALANCE_STRATEGY}")
print(f"HPO objective metric   : {HPO_OBJECTIVE_METRIC}  (optimized on population-rate val fold)")
print(f"Trials / parallel jobs : {N_TRIALS} / {N_JOBS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 1 — Spine (label + eligibility)
# MAGIC Reused verbatim from `first_order_propensity_spine_and_selection.py`: the LABEL / ELIGIBILITY half of
# MAGIC the v7 ETL query, emitting exactly `internal_user_id`, `feature_ts`, `output`. Status filter kept as
# MAGIC `IN ('COMPLETE','CANCELLED')`; first-order anti-join drops anyone who had already ordered as of the cutoff.

# COMMAND ----------

from datetime import datetime, timedelta


def _sql_in_list(values):
    """Render a Python list as a SQL IN-list of single-quoted literals."""
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def resolve_cutoff_ts(widget_value: str) -> str:
    """Return the cutoff timestamp string (mirrors the spine notebook's resolver).

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


def build_spine_sql(cutoff: str, performance_end: str, ist: int) -> str:
    """LABEL + ELIGIBILITY only. Emits (internal_user_id, feature_ts, output). Reused verbatim."""
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


cutoff_ts = resolve_cutoff_ts(dbutils.widgets.get("as_of_date"))
performance_end_ts = (_parse_ts(cutoff_ts) + timedelta(days=PERFORMANCE_DAYS)).strftime("%Y-%m-%d %H:%M:%S")

spine_df = spark.sql(build_spine_sql(cutoff_ts, performance_end_ts, IST_OFFSET_MINUTES))
spine_df = spine_df.cache()  # count + create_training_set both consume it
_n_spine = spine_df.count()
_n_pos = spine_df.filter(f"{LABEL} = 1").count()
print(f"cutoff/feature_ts : {cutoff_ts}")
print(f"performance_end   : {performance_end_ts}")
print(f"Spine rows        : {_n_spine:,}  (positives: {_n_pos:,} = {_n_pos / max(_n_spine, 1):.3%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 2 — Point-in-time training set (SELECTED subset)
# MAGIC `create_training_set` does the as-of join on the **selected** store features only. Then `segment` is
# MAGIC attached **current-state** and the v7 engagement gate is applied against the feature columns.
# MAGIC
# MAGIC ### `segment` — included WITH the finding-#1 leakage caveat
# MAGIC `segment`'s source `simple.crud.user_segment_mapping` is a CRUD current-state snapshot with **no
# MAGIC effective-date / as-of column**, so it cannot be joined point-in-time. Joined current-state below, a
# MAGIC row's `segment` reflects the user's segment *now*, which may encode a **post-cutoff reassignment** —
# MAGIC i.e. potential target leakage (HIGH-severity finding #1). Kept per the confirmed business decision,
# MAGIC pending an effective-dated `user_segment_mapping`. Toggle it off with the `include_segment` widget.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()

# Point-in-time lookup of the SELECTED store-resident subset (feature selection already done upstream).
training_set = fe.create_training_set(
    df=spine_df,
    feature_lookups=[
        FeatureLookup(
            table_name=FEATURE_TABLE,
            lookup_key=ENTITY_KEY,
            timestamp_lookup_key=FEATURE_TS,  # as-of join -> no future leakage from the feature side
            feature_names=SELECTED_STORE_FEATURES,
        )
    ],
    label=LABEL,
    exclude_columns=[FEATURE_TS],  # keep the join key out of the feature matrix
)

training_df = training_set.load_df()
print(f"PIT training columns ({len(training_df.columns)}): {training_df.columns}")

# COMMAND ----------

# --- Attach `segment` CURRENT-STATE (finding-#1 caveat above) + v7 engagement gate ----------
analysis_df = training_df

if INCLUDE_SEGMENT:
    # Reproduces v7's `coalesce(usm.segment_name, 'No Segment') AS segment`, joined current-state.
    segment_df = spark.sql(f"SELECT internal_user_id, segment_name FROM {USER_SEGMENT_MAPPING}")
    analysis_df = (
        analysis_df.join(segment_df, on=ENTITY_KEY, how="left")
        .withColumnRenamed("segment_name", "segment")
        .fillna({"segment": "No Segment"})
    )

# v7's engagement gate, applied AGAINST THE FEATURE TABLE columns. Only reference recency columns that
# were actually selected (NULL recency == "never", so IS NOT NULL reproduces v7's *_search IS NOT NULL,
# and days_since_last_app_open < 90 reproduces v7's coalesce(...,9999) < 90).
_gate_terms = []
if "days_since_last_app_open" in analysis_df.columns:
    _gate_terms.append("days_since_last_app_open < 90")
for _rc in [
    "days_since_last_flight_search", "days_since_last_hotel_search",
    "days_since_last_bus_search", "days_since_last_train_search",
]:
    if _rc in analysis_df.columns:
        _gate_terms.append(f"{_rc} IS NOT NULL")
if _gate_terms:
    analysis_df = analysis_df.filter(" OR ".join(_gate_terms))
    print(f"Applied v7 engagement gate on selected recency columns: {_gate_terms}")
else:
    print("No recency/app-open columns in the selected subset -> engagement gate skipped (matches v7 only "
          "if those columns are selected; documented divergence otherwise).")

analysis_df = analysis_df.cache()
_n_analysis = analysis_df.count()
print(f"Analysis (PIT training) set: {_n_analysis:,} rows, {len(analysis_df.columns)} columns")
print(f"Feature set for HPO        : {SELECTED_STORE_FEATURES + CATEGORICAL_FEATURES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 3 — Model matrix + split
# MAGIC XGBoost consumes the **raw** stored values (native missing handling for NULL recency), with `segment`
# MAGIC one-hot-encoded — mirroring the sibling notebook's `_build_model_matrix`. v7's bucketing/one-hot of the
# MAGIC numerics is a separate downstream-encoding concern and is intentionally NOT reproduced here.
# MAGIC
# MAGIC Rows are split FIRST. The `segment` one-hot vocabulary is learned from the **TRAIN split only** and
# MAGIC applied unchanged to val/test (unseen categories fold into an explicit `Other` bucket), so there is no
# MAGIC train/test leakage and all splits share identical feature columns. No test label statistic is computed
# MAGIC here — the held-out test labels are untouched until the single final evaluation.

# COMMAND ----------

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if SAMPLE_FRACTION:
    _sdf = analysis_df.sample(fraction=SAMPLE_FRACTION, seed=RANDOM_STATE)
    print(f"Sampling {SAMPLE_FRACTION:.0%} of the PIT training set (driver-memory safety).")
else:
    _sdf = analysis_df

pdf = _sdf.toPandas()

_numeric_present = [c for c in SELECTED_STORE_FEATURES if c in pdf.columns]
_categorical_present = [c for c in CATEGORICAL_FEATURES if c in pdf.columns]
y_all = pdf[LABEL].astype(int)


def fit_segment_vocab(pdf_train, categorical_features, top_cats):
    """Learn the one-hot vocabulary from the TRAIN split ONLY. 'Other' is the catch-all for unseen/rare cats.

    Returns {feature: [categories..., 'Other']} — a FIXED category set applied unchanged to val/test.
    """
    vocab = {}
    for c in categorical_features:
        vals = pdf_train[c].astype("object").fillna("No Segment")
        cats = list(vals.value_counts().head(top_cats).index)
        if "Other" not in cats:
            cats.append("Other")
        vocab[c] = cats
    return vocab


def build_model_matrix(pdf_part, numeric_features, categorical_features, vocab):
    """Raw numeric (NaN kept — XGBoost handles missing natively) + one-hot using a FIXED train-fit vocab.

    Mirrors first_order_propensity_spine_and_selection.py::_build_model_matrix, but the category vocabulary
    is passed in (fit on TRAIN) rather than re-derived, so train/val/test get identical columns with no leak.
    """
    parts = []
    for c in numeric_features:
        parts.append(pdf_part[c].astype("float32").rename(c))
    for c in categorical_features:
        cats = vocab[c]
        vals = pdf_part[c].astype("object").fillna("No Segment")
        vals = vals.where(vals.isin(cats), "Other")             # unseen/rare -> explicit Other bucket
        vals = pd.Categorical(vals, categories=cats)            # FIXED categories -> stable, complete columns
        dummies = pd.get_dummies(vals, prefix=c, dtype="float32")
        dummies.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", str(dc)) for dc in dummies.columns]
        dummies.index = pdf_part.index
        parts.append(dummies)
    return pd.concat(parts, axis=1)


# COMMAND ----------

# --- SPLIT ROWS FIRST, then fit the vocab on TRAIN only. Test labels stay untouched until final eval. -------
# Split on row indices so the vocab is never exposed to val/test rows.
_idx = pdf.index.to_numpy()
_idx_trv, _idx_test = train_test_split(
    _idx, test_size=TEST_FRACTION, random_state=RANDOM_STATE, stratify=y_all.to_numpy(),
)
_idx_tr_raw, _idx_val = train_test_split(
    _idx_trv, test_size=VAL_FRACTION, random_state=RANDOM_STATE, stratify=y_all.loc[_idx_trv].to_numpy(),
)

# Vocabulary learned from the TRAIN split ONLY (no full-data fit -> no train/test leakage).
_vocab = fit_segment_vocab(pdf.loc[_idx_tr_raw], _categorical_present, SEGMENT_TOP_CATS)

X_tr_raw = build_model_matrix(pdf.loc[_idx_tr_raw], _numeric_present, _categorical_present, _vocab)
X_val = build_model_matrix(pdf.loc[_idx_val], _numeric_present, _categorical_present, _vocab)
X_test = build_model_matrix(pdf.loc[_idx_test], _numeric_present, _categorical_present, _vocab)
FEATURE_NAMES = list(X_tr_raw.columns)
# Defensive: identical columns across splits (the FIXED vocab already guarantees this).
X_val = X_val.reindex(columns=FEATURE_NAMES, fill_value=0.0)
X_test = X_test.reindex(columns=FEATURE_NAMES, fill_value=0.0)

y_tr_raw = y_all.loc[_idx_tr_raw]
y_val = y_all.loc[_idx_val]
y_test = y_all.loc[_idx_test]  # NOT inspected until the single final evaluation


def undersample_majority(X_in, y_in, neg_per_pos, seed):
    """v7's random majority (negatives) undersampling to neg_per_pos:1. Positives kept; test never touched."""
    pos_pos = np.where(y_in.values == 1)[0]
    neg_pos = np.where(y_in.values == 0)[0]
    n_target = len(pos_pos) * neg_per_pos
    if len(neg_pos) > n_target:
        rng = np.random.RandomState(seed)
        neg_keep = rng.choice(neg_pos, size=n_target, replace=False)
        keep = np.concatenate([pos_pos, neg_keep])
        return X_in.iloc[keep], y_in.iloc[keep]
    return X_in, y_in


# Methodology fix (a): the two mutually-exclusive imbalance strategies.
if IMBALANCE_STRATEGY == "undersample_fixed_spw1":
    # Default: undersample TRAIN to 3:1 AND fix scale_pos_weight=1 (dropped from the search space) — no
    # double correction. Val stays at the true prior.
    X_tr, y_tr = undersample_majority(X_tr_raw, y_tr_raw, TARGET_NEG_PER_POS, RANDOM_STATE)
    print(f"Undersampled TRAIN to {TARGET_NEG_PER_POS}:1 — dropped "
          f"{len(y_tr_raw) - len(y_tr):,} negatives. scale_pos_weight FIXED at 1 (not tuned).")
else:
    # tune_spw_no_undersample: keep the full prior in TRAIN and tune scale_pos_weight in the objective.
    X_tr, y_tr = X_tr_raw, y_tr_raw
    print("No undersampling — scale_pos_weight will be TUNED in the objective (population prior in TRAIN).")

# neg/pos ratio of the *current* training fold; used only when tuning scale_pos_weight.
SPW_RATIO = float((y_tr == 0).sum() / max(int((y_tr == 1).sum()), 1))

print(f"TRAIN: {X_tr.shape[0]:,} rows (rate {y_tr.mean():.3%})   [neg/pos ratio {SPW_RATIO:.1f}]")
print(f"VAL  : {X_val.shape[0]:,} rows (rate {y_val.mean():.3%})   [population prior — fix (b)]")
# TEST label rate deliberately NOT printed here (no peeking) — only its size; labels stay sealed until eval.
print(f"TEST : {X_test.shape[0]:,} rows   [held out; labels untouched until the single final evaluation]")

# Broadcast the compact arrays so each distributed trial reads the same data on its executor.
_HPO_DATA = spark.sparkContext.broadcast({
    "X_train": X_tr.to_numpy(dtype="float32"),
    "y_train": y_tr.to_numpy(dtype="int32"),
    "X_val": X_val.to_numpy(dtype="float32"),
    "y_val": y_val.to_numpy(dtype="int32"),
    "feature_names": FEATURE_NAMES,
    "spw_ratio": SPW_RATIO,
})

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 4 — Business-metric helpers + the Optuna objective
# MAGIC All metrics are evaluated on the **population-rate** validation fold (fix (b)). The objective returns
# MAGIC the **negated** business metric because `MlflowSparkStudy` minimizes (no `direction` arg; see the API
# MAGIC grounding cell). Per boosting round the pruning callback reports the negated watch metric so
# MAGIC `MedianPruner` — the documented default pruner — prunes consistently under minimization.

# COMMAND ----------


import xgboost as xgb  # module scope: the callback subclasses xgb.callback.TrainingCallback below


def _top_decile_lift(y_true, proba):
    """Booker rate in the top-10% by score / population base rate. == v7's decile-1 lift_vs_base."""
    import numpy as np

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    k = max(int(round(n * 0.10)), 1)
    top = np.argsort(-proba)[:k]  # highest scores first
    base = y_true.mean()
    if base <= 0:
        return 0.0
    return float(y_true[top].mean() / base)


def _max_f2(y_true, proba):
    """Max F2 over a threshold sweep on the population-rate fold (v7 optimized/reported F2)."""
    import numpy as np
    from sklearn.metrics import fbeta_score

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    best = 0.0
    for t in np.linspace(0.05, 0.95, 91):
        yp = (proba >= t).astype(int)
        if yp.sum() == 0:
            continue
        best = max(best, fbeta_score(y_true, yp, beta=2, zero_division=0))
    return float(best)


def best_f2_threshold(y_true, proba):
    """Threshold that maximizes F2 on the GIVEN fold. Call this on VALIDATION, then apply once to test
    (audit Tier 1 #3: never choose the threshold on the held-out test set). Returns (threshold, f2_at_thr)."""
    import numpy as np
    from sklearn.metrics import fbeta_score

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    best_t, best_f2 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        yp = (proba >= t).astype(int)
        if yp.sum() == 0:
            continue
        f2 = fbeta_score(y_true, yp, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2, best_t = f2, float(t)
    return best_t, float(best_f2)


def business_metric(y_true, proba, name):
    """Higher-is-better business metric on the population-rate fold. One of the fix-(b) choices."""
    from sklearn.metrics import roc_auc_score

    if name == "roc_auc":
        return float(roc_auc_score(y_true, proba))
    if name == "f2":
        return _max_f2(y_true, proba)
    return _top_decile_lift(y_true, proba)  # default: top_decile_lift


class OptunaXGBPruningCallback(xgb.callback.TrainingCallback):
    """Reports the validation watch metric to the Optuna trial each round and prunes via should_prune().

    Standard Optuna pruning pattern (mirrors optuna.integration.XGBoostPruningCallback). Because the study
    MINIMIZES, we report the NEGATED watch metric so a low ROC-AUC trajectory reads as "worse" (above the
    running median) and gets pruned. Pruning support is confirmed in the MLflow source; see the run cell.
    """

    def __init__(self, trial, eval_set_name, metric_name):
        self.trial = trial
        self.eval_set_name = eval_set_name
        self.metric_name = metric_name

    def after_iteration(self, model, epoch, evals_log):
        import optuna

        history = evals_log.get(self.eval_set_name, {}).get(self.metric_name)
        if not history:
            return False
        self.trial.report(-float(history[-1]), step=epoch)  # negate: study minimizes
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return False


def objective(trial):
    """One distributed trial: fit XGBoost with early stopping, return the negated business metric on VAL."""
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    data = _HPO_DATA.value

    # --- Search space (v7 param_distributions -> Optuna). n_estimators is NOT tuned (early stopping). ---
    # scipy semantics from v7: randint(a, b) is [a, b-1] inclusive; uniform(loc, scale) is [loc, loc+scale].
    params = {
        "objective": "binary:logistic",
        "eval_metric": WATCH_METRIC,                 # 'auc' -> maximized by early stopping; reported for pruning
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "max_depth": trial.suggest_int("max_depth", 3, 8),            # v7 randint(3, 9) -> 3..8
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 9),  # v7 randint(1, 10) -> 1..9
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),  # v7 loguniform(0.01, 0.2)
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),     # v7 uniform(0.7, 0.3) -> 0.7..1.0
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),  # v7 uniform(0.6, 0.4) -> 0.6..1.0
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),             # v7 uniform(0.0, 3.0) -> 0..3
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),     # v7 uniform(0.0, 2.0) -> 0..2
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 6.0),   # v7 uniform(1.0, 5.0) -> 1..6
    }

    # Methodology fix (a): tune scale_pos_weight ONLY when not undersampling; else FIX at 1 (no double-correct).
    if IMBALANCE_STRATEGY == "tune_spw_no_undersample":
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1.0, data["spw_ratio"], log=True)
    else:
        params["scale_pos_weight"] = 1.0

    dtrain = xgb.DMatrix(data["X_train"], label=data["y_train"], feature_names=data["feature_names"])
    dval = xgb.DMatrix(data["X_val"], label=data["y_val"], feature_names=data["feature_names"])

    pruning_cb = OptunaXGBPruningCallback(trial, "validation", WATCH_METRIC)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=MAX_BOOST_ROUNDS,
        evals=[(dtrain, "train"), (dval, "validation")],  # early stopping watches the LAST eval (validation)
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        callbacks=[pruning_cb],
        verbose_eval=False,
    )

    best_it = int(getattr(booster, "best_iteration", MAX_BOOST_ROUNDS - 1))
    val_proba = booster.predict(dval, iteration_range=(0, max(best_it + 1, 1)))

    score = business_metric(data["y_val"], val_proba, HPO_OBJECTIVE_METRIC)
    # Persisted with the trial (via MlflowStorage) for later inspection.
    trial.set_user_attr("best_iteration", best_it)
    trial.set_user_attr("val_roc_auc", float(roc_auc_score(data["y_val"], val_proba)))
    trial.set_user_attr("val_top_decile_lift", _top_decile_lift(data["y_val"], val_proba))
    trial.set_user_attr("val_f2_max", _max_f2(data["y_val"], val_proba))
    trial.set_user_attr("val_base_rate", float(np.mean(data["y_val"])))

    return -float(score)  # study MINIMIZES -> return negative to maximize the business metric


print("Objective + pruning callback defined.")
_spw_note = ", scale_pos_weight" if IMBALANCE_STRATEGY == "tune_spw_no_undersample" else ""
print("Search space: max_depth, min_child_weight, learning_rate, subsample, colsample_bytree, gamma, "
      f"reg_alpha, reg_lambda{_spw_note}  (n_estimators via early stopping)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 5 — Run the distributed study (`MlflowSparkStudy` + `MlflowStorage`)
# MAGIC The sole HPO mechanism. `MlflowStorage` persists every trial as an MLflow run in the notebook
# MAGIC experiment; `MlflowSparkStudy.optimize` fans trials out across executors (`n_jobs=-1` matches the
# MAGIC number of Spark tasks). `MedianPruner` and `TPESampler` are the documented defaults; passed explicitly
# MAGIC here (with a seed) for reproducibility.
# MAGIC
# MAGIC **Study name = config fingerprint.** `study_name` is a short hash of the immutable config that defines
# MAGIC trial comparability: cutoff / as-of date, the sorted selected feature list, whether `segment` is
# MAGIC included, the HPO objective metric, the imbalance strategy, and the search-space version. A study
# MAGIC resumes (re-uses prior trials) ONLY when that fingerprint matches; ANY change starts a FRESH study, so
# MAGIC a trial from an incompatible config can never become champion for this one.

# COMMAND ----------

# API grounded against (retrieved 2026-07-17):
#   https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/optuna
#   https://learn.microsoft.com/en-us/azure/databricks/machine-learning/automl-hyperparam-tuning/optuna
#   https://github.com/mlflow/mlflow/blob/master/mlflow/pyspark/optuna/study.py  (signatures + default direction)
#
# Pruning support is CONFIRMED in the MLflow source: MlflowStorage.set_trial_intermediate_value() exists
#   (so trial.report persists per-round values) and MlflowSparkStudy handles optuna.TrialPruned. MedianPruner
#   is the documented default. Pruning is a MANDATORY part of this HPO contract — there is no path that
#   silently disables it.
# TODO(verify-live): the docs show no end-to-end intermediate-value example, so confirm on the target cluster
#   that trials actually reach state PRUNED (e.g. inspect trial states after the run). If a runtime ever shows
#   pruning is NOT taking effect, FAIL FAST and fix the reporting path — do NOT disable pruning to proceed.
import mlflow
import optuna
from mlflow.optuna.storage import MlflowStorage
from mlflow.pyspark.optuna.study import MlflowSparkStudy
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Notebook experiment (per the official example). Call mlflow.set_experiment for a durable, non-notebook
# experiment instead.
experiment_id = mlflow.get_experiment_by_name(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
).experiment_id

mlflow_storage = MlflowStorage(experiment_id=experiment_id)

# Study name from an IMMUTABLE config fingerprint (see the markdown above). Resume ONLY on an exact match;
# any change to these inputs -> a new study, so incompatible trials never mix.
import hashlib

_fp_payload = "|".join([
    f"cutoff={cutoff_ts}",
    "features=" + ",".join(sorted(SELECTED_STORE_FEATURES)),
    f"segment={INCLUDE_SEGMENT}",
    f"objective={HPO_OBJECTIVE_METRIC}",
    f"imbalance={IMBALANCE_STRATEGY}",
    f"space={SEARCH_SPACE_VERSION}",
])
_fp = hashlib.sha1(_fp_payload.encode("utf-8")).hexdigest()[:12]
STUDY_NAME = f"fop_hpo_{_fp}"
print(f"Study fingerprint payload: {_fp_payload}")
print(f"Study name               : {STUDY_NAME}")

mlflow_study = MlflowSparkStudy(
    study_name=STUDY_NAME,
    storage=mlflow_storage,
    sampler=TPESampler(seed=RANDOM_STATE),  # documented default is TPESampler; seeded for reproducibility
    pruner=MedianPruner(),                  # documented default pruner
)

print(f"Launching MlflowSparkStudy '{STUDY_NAME}' — {N_TRIALS} trials, n_jobs={N_JOBS} "
      f"(experiment_id={experiment_id}). Trials distribute across executors on a multi-node cluster.")
mlflow_study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)
print("Study complete.")

# COMMAND ----------

# --- Retrieve best params/value (study MINIMIZES -> flip the sign back to the business metric) -----------
# `best_params` / `best_value` are inherited from optuna's Study and are supported here (the docs' example
# `study.best_params` just had a typo'd variable name). Use them directly — NOT get_resume_info(), which
# only holds results for RESUMED studies and would be unsound as a general accessor.
best_params = dict(mlflow_study.best_params)
best_value_min = float(mlflow_study.best_value)
best_business_metric = -best_value_min  # undo the minimization negation

print(f"Best {HPO_OBJECTIVE_METRIC} (val, population rate): {best_business_metric:.4f}")
print("Best params:")
for _k, _v in best_params.items():
    print(f"  {_k}: {_v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 6 — Champion fit + honest evaluation (test touched ONCE)
# MAGIC Refit XGBoost on the best params, then score the **held-out test at the true population base rate**.
# MAGIC The effective number of rounds is re-derived deterministically via early stopping on TRAIN→VAL (same
# MAGIC data/seed as the winning trial), then the champion is fit on TRAIN+VAL for that many rounds. Report
# MAGIC ROC-AUC and top-decile lift (both threshold-free). The **F2 threshold is chosen on VALIDATION** and
# MAGIC applied ONCE to test (audit Tier 1 #3 — never pick the threshold on the held-out set). Log best params,
# MAGIC the chosen threshold, and test metrics to MLflow.

# COMMAND ----------

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score

# Base params + the tuned params. scale_pos_weight per the imbalance strategy (fix (a)).
champ_params = {
    "objective": "binary:logistic",
    "eval_metric": WATCH_METRIC,
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
}
champ_params.update(best_params)
if IMBALANCE_STRATEGY != "tune_spw_no_undersample":
    champ_params["scale_pos_weight"] = 1.0  # fixed; not in the search space for the default strategy

# 1) Re-derive the effective round count on TRAIN -> VAL (reproduces the winning trial's early stopping).
_dtr = xgb.DMatrix(X_tr.to_numpy(dtype="float32"), label=y_tr.to_numpy(dtype="int32"), feature_names=FEATURE_NAMES)
_dvl = xgb.DMatrix(X_val.to_numpy(dtype="float32"), label=y_val.to_numpy(dtype="int32"), feature_names=FEATURE_NAMES)
_probe = xgb.train(
    champ_params, _dtr, num_boost_round=MAX_BOOST_ROUNDS,
    evals=[(_dvl, "validation")], early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose_eval=False,
)
best_num_round = int(getattr(_probe, "best_iteration", MAX_BOOST_ROUNDS - 1)) + 1
print(f"Champion rounds (from early stopping on train->val): {best_num_round}")

# F2 operating threshold is chosen on the VALIDATION fold (never on test — audit Tier 1 #3), using the
# probe model's validation predictions. This FIXED threshold is applied once to test below.
_val_proba_probe = _probe.predict(_dvl, iteration_range=(0, best_num_round))
f2_threshold, val_f2_at_threshold = best_f2_threshold(y_val.to_numpy(), _val_proba_probe)
print(f"F2 threshold chosen on VAL: {f2_threshold:.3f} (val F2 = {val_f2_at_threshold:.4f})")

# 2) Final champion on TRAIN+VAL, using the same imbalance strategy so the training prior is consistent.
X_fit = pd.concat([X_tr_raw, X_val], axis=0)
y_fit = pd.concat([y_tr_raw, y_val], axis=0)
if IMBALANCE_STRATEGY != "tune_spw_no_undersample":
    X_fit, y_fit = undersample_majority(X_fit, y_fit, TARGET_NEG_PER_POS, RANDOM_STATE)

_dfit = xgb.DMatrix(X_fit.to_numpy(dtype="float32"), label=y_fit.to_numpy(dtype="int32"), feature_names=FEATURE_NAMES)
champion = xgb.train(champ_params, _dfit, num_boost_round=best_num_round, verbose_eval=False)
print(f"Champion fit on TRAIN+VAL: {X_fit.shape[0]:,} rows (rate {y_fit.mean():.3%}).")

# COMMAND ----------

# --- Honest evaluation on the held-out TEST (population base rate) — touched ONCE ------------------------
_dtest = xgb.DMatrix(X_test.to_numpy(dtype="float32"), feature_names=FEATURE_NAMES)
test_proba = champion.predict(_dtest)
_y_test = y_test.to_numpy()

test_roc_auc = float(roc_auc_score(_y_test, test_proba))                 # threshold-free
test_pr_auc = float(average_precision_score(_y_test, test_proba))        # threshold-free
test_top_decile_lift = _top_decile_lift(_y_test, test_proba)             # threshold-free
# F2 at the VAL-chosen threshold, applied ONCE to test (no sweep on test labels).
test_f2_at_val_thr = float(fbeta_score(_y_test, (test_proba >= f2_threshold).astype(int), beta=2, zero_division=0))
test_base_rate = float(_y_test.mean())  # computed here, at the single final eval (not before)

print("=== Honest TEST metrics (population base rate) ===")
print(f"Base rate            : {test_base_rate:.3%}  ({int(_y_test.sum()):,} positives / {len(_y_test):,})")
print(f"ROC-AUC              : {test_roc_auc:.4f}")
print(f"PR-AUC               : {test_pr_auc:.4f}")
print(f"Top-decile lift      : {test_top_decile_lift:.3f}x")
print(f"F2 @ val threshold   : {test_f2_at_val_thr:.4f}  (threshold {f2_threshold:.3f} chosen on VAL)")

# Decile lift table (v7-style: Decile 1 = top 10%).
_eval = pd.DataFrame({"actual": _y_test, "proba": test_proba})
_eval["rank"] = _eval["proba"].rank(method="first", ascending=False)
_eval["decile"] = pd.cut(
    _eval["rank"], bins=np.linspace(0, len(_eval), 11), labels=list(range(1, 11)), include_lowest=True,
).astype(int)
_lift = _eval.groupby("decile").agg(users=("actual", "size"), bookers=("actual", "sum")).sort_index()
_lift["booker_rate"] = _lift["bookers"] / _lift["users"]
_lift["lift_vs_base"] = _lift["booker_rate"] / max(test_base_rate, 1e-12)
print("\n=== Decile lift TEST (Decile 1 = top 10%) ===")
print(_lift.to_string())

# COMMAND ----------

# --- Log best params + honest test metrics to MLflow ----------------------------------------------------
with mlflow.start_run(run_name="fop_hpo_champion") as run:
    mlflow.set_tags({
        "project": "first_order_propensity",
        "stage": "hpo_champion",
        "feature_table": FEATURE_TABLE,
        "hpo_mechanism": "MlflowSparkStudy",
        "imbalance_strategy": IMBALANCE_STRATEGY,
        "hpo_objective_metric": HPO_OBJECTIVE_METRIC,
    })
    mlflow.log_params({
        "as_of_date": cutoff_ts,
        "performance_end": performance_end_ts,
        "selected_store_features": ",".join(_numeric_present),
        "include_segment": INCLUDE_SEGMENT,
        "n_trials": N_TRIALS,
        "n_jobs": N_JOBS,
        "target_neg_per_pos": TARGET_NEG_PER_POS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "champion_num_boost_round": best_num_round,
        **{f"best_{k}": v for k, v in best_params.items()},
    })
    mlflow.log_metrics({
        f"best_val_{HPO_OBJECTIVE_METRIC}": best_business_metric,
        "val_f2_threshold": f2_threshold,          # chosen on VAL, applied once to test
        "val_f2_at_threshold": val_f2_at_threshold,
        "test_base_rate": test_base_rate,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
        "test_top_decile_lift": test_top_decile_lift,
        "test_f2_at_val_threshold": test_f2_at_val_thr,
    })
    print(f"Logged champion run: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Follow-ons (deliberately OUT of scope here)
# MAGIC This notebook stops at "best params found + champion honestly evaluated + logged to MLflow". The next
# MAGIC steps, each its own notebook/task:
# MAGIC
# MAGIC * **UC model registration** — log the champion as an MLflow model and register it in Unity Catalog
# MAGIC   (with the selected `FeatureLookup` so scoring is point-in-time correct).
# MAGIC * **Probability calibration** — undersampling (default strategy) distorts absolute probabilities;
# MAGIC   calibrate (e.g. Platt / isotonic on a population-rate fold) before any threshold-based decisioning.
# MAGIC * **Operating-threshold selection** — this notebook derives an F2 threshold on the population-rate VAL
# MAGIC   fold (for the reported test F2) and otherwise optimizes threshold-free ranking; a full cost-based
# MAGIC   deployment threshold (v7 shipped at 0.45) still belongs downstream.
# MAGIC * **Model serving** — batch scoring job or real-time endpoint.
# MAGIC * **Feature-table materialization** — scheduled refresh of the shared feature table for new as-of dates.
# MAGIC * **`segment` finding-#1 remediation** — replace the current-state join with an effective-dated /
# MAGIC   SCD2 `user_segment_mapping` joined as-of `feature_ts`, then move `segment` into the PIT `FeatureLookup`.
# MAGIC * **Single-node Optuna** — intentionally NOT provided; `MlflowSparkStudy` is the sole HPO mechanism per
# MAGIC   the build request.
