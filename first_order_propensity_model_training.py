# Databricks notebook source
# MAGIC %md
# MAGIC # Scapia — First-Order-Propensity — Model TRAINING + UC REGISTRATION
# MAGIC
# MAGIC Third notebook in the chain, consuming the outputs of the first two:
# MAGIC
# MAGIC 1. `first_order_propensity_user_features.py` — the shared per-user **feature store** table.
# MAGIC 2. `first_order_propensity_spine_and_selection.py` — the spine + **feature selection** run
# MAGIC    (`selected_features.json`).
# MAGIC 3. `first_order_propensity_hpo.py` — the distributed **HPO** run (`fop_hpo_champion` -> `best_params`).
# MAGIC 4. **THIS notebook** — refit the champion on the frozen preprocessing, calibrate, assemble a
# MAGIC    self-contained `pyfunc`, and **register it to Unity Catalog** with a signature + alias.
# MAGIC
# MAGIC ## What this notebook does (the 10-step contract)
# MAGIC 1. **Consume upstream via REQUIRED widgets** — `hpo_champion_run_id` (-> `best_params`) and
# MAGIC    `selection_run_id` (-> `selected_features.json`). Both fail-fast if blank; nothing is fabricated.
# MAGIC    `INCLUDE_SEGMENT` is DERIVED from whether `segment` is in the selected features.
# MAGIC 2. **Rebuild the training set exactly as HPO does** — same spine, same PIT `create_training_set` on the
# MAGIC    selected features, same split logic + seed. `spark.sql` only — no connector, no PATs, no egress.
# MAGIC 3. **FREEZE the preprocessing** — the segment top-N one-hot vocabulary is fit on the **TRAIN fold only**
# MAGIC    and captured as frozen state (`handle_unknown='ignore'` semantics). The SAME frozen transform builds
# MAGIC    the training matrices AND runs inside `predict`, so there is provably no data-dependent encoding at
# MAGIC    predict time. This is the audit's train/serve-skew fix.
# MAGIC 4. **Fit the champion XGBoost** with `best_params`, honoring the HPO imbalance strategy (undersample the
# MAGIC    train fold + `scale_pos_weight=1` by default), for the champion's **frozen** boosting-round count
# MAGIC    (`champion_num_boost_round` from HPO). The final fit does **NOT** early-stop on VAL — HPO already
# MAGIC    tuned against VAL, so re-using it here would make the downstream calibration/threshold optimistic.
# MAGIC 5. **Optional calibration** (`CALIBRATE`, default true) — isotonic/Platt fit on the **population-rate VAL
# MAGIC    holdout** (never the undersampled train, never test). A decile ranking is ALWAYS emitted (rank-only
# MAGIC    consumers). Both calibrated + uncalibrated are logged. The F2 threshold is chosen on VAL and applied
# MAGIC    to test exactly once. **No test-label peeking anywhere.**
# MAGIC 6. **Assemble ONE `mlflow.pyfunc.PythonModel`** carrying the full chain: frozen preprocessing -> booster
# MAGIC    -> optional calibrator, via the `artifacts=` dict.
# MAGIC 7. **Infer the signature** with `infer_signature` + an `input_example` (mandatory for UC registration).
# MAGIC 8. **Register to Unity Catalog** — `set_registry_uri('databricks-uc')`, three-level
# MAGIC    `registered_model_name` (default `mlops_data_science.models.first_order_propensity`).
# MAGIC 9. **Set the alias** — DEFAULT: `@champion` if the model has none yet, else `@challenger`. The
# MAGIC    champion-promotion gate is documented below; a new version never auto-promotes over a live champion.
# MAGIC 10. **One honest held-out TEST evaluation** (ROC-AUC, top-decile lift + capture, F2 at the VAL-chosen
# MAGIC     threshold) logged to the registration run alongside the registered version.
# MAGIC
# MAGIC ## Out of scope (later deliverables — deliberately NOT built here)
# MAGIC No Model Serving endpoint, no batch scorer / Lakeflow Job, no Lakehouse Monitoring, no retraining loop.

# COMMAND ----------

# MAGIC %md
# MAGIC ## API grounding — what was verified against current docs
# MAGIC The MLflow / UC surface below was grounded against the official docs (retrieved 2026-07-21):
# MAGIC
# MAGIC * `mlflow.pyfunc.log_model(...)` — accepts `python_model`, `artifacts` (dict), `signature`,
# MAGIC   `input_example`, `registered_model_name`, `pip_requirements`, `code_paths`. First positional is
# MAGIC   `artifact_path` (deprecated in MLflow 3) with `name=` also accepted; this notebook uses `name=`.
# MAGIC   Ref: `https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html`
# MAGIC * `mlflow.pyfunc.PythonModel` — `load_context(self, context)` and
# MAGIC   `predict(self, context, model_input, params=None)`. Artifacts reachable via
# MAGIC   `context.artifacts["<key>"]`. Same ref as above.
# MAGIC * `mlflow.models.infer_signature(model_input, model_output)` — used with an `input_example`.
# MAGIC * `MlflowClient.set_registered_model_alias(name, alias, version)` — verified parameter order.
# MAGIC   Ref: `https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html`
# MAGIC * `mlflow.set_registry_uri('databricks-uc')` + three-level `catalog.schema.model` name — the standard
# MAGIC   UC registration path; a **signature is mandatory** for UC registration.
# MAGIC   Ref: `https://docs.databricks.com/aws/en/mlflow/models` (Log, load, and register MLflow models).
# MAGIC
# MAGIC **Registered-version resolution (resolved — no TODO):** `ModelInfo.registered_model_version` is not
# MAGIC reliably populated across MLflow versions (open upstream feature request), so the registered version is
# MAGIC resolved by an **exact `run_id` match** against `search_model_versions` — polling with bounded
# MAGIC exponential backoff to absorb registration-index lag, and **raising** if no version matching this run
# MAGIC appears. It never falls back to the latest version (which, under concurrent registrations, could be an
# MAGIC unrelated model). See `_resolve_registered_version` in the registration cell.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Dependencies
# MAGIC MLflow 3.0, XGBoost, scikit-learn and `databricks-feature-engineering` all ship with Databricks Runtime
# MAGIC 17.0 ML+. The install below just pins recent versions so `create_training_set` (PIT join),
# MAGIC `mlflow.pyfunc` UC registration and `infer_signature` are all available. Safe to skip on a current ML
# MAGIC Runtime.

# COMMAND ----------

# MAGIC %pip install -U "mlflow>=3.0" databricks-feature-engineering xgboost scikit-learn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. CONFIG
# MAGIC Names and source tables are kept **verbatim** from the feature-store / spine / HPO notebooks so the
# MAGIC point-in-time join and the reproduced split line up exactly with what HPO tuned.

# COMMAND ----------

# ---------------------------------------------------------------------------
# CONFIG — edit here (verbatim from the HPO notebook so the PIT join + split match)
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

# Only `segment` is supported as a non-store (current-state) feature in this pipeline.
SUPPORTED_NON_STORE_FEATURES = ["segment"]

# --- Reproducibility / split (IDENTICAL to HPO so the reproduced folds + frozen vocab match) --
RANDOM_STATE = 42
TEST_FRACTION = 0.20   # held out, touched exactly once at the final eval
VAL_FRACTION = 0.25    # fraction of the (non-test) remainder used as the val fold (-> ~60/20/20)

# --- Imbalance handling (methodology fix (a): NO double correction) ---------
# v7 undersamples the majority (negatives) in TRAIN to 3:1 (neg:pos). Default; overridable from the champion.
TARGET_NEG_PER_POS = 3

# --- XGBoost (methodology: do NOT tune n_estimators; the final fit uses the HPO-frozen round count) -----
# The final booster is trained for the champion's frozen `champion_num_boost_round` (carried over from the
# HPO run) with NO VAL early stopping — see SECTION 5. So there is deliberately no MAX_BOOST_ROUNDS /
# EARLY_STOPPING_ROUNDS here: consulting VAL in the final fit would double-use the calibration/threshold
# holdout that HPO already tuned against. WATCH_METRIC is retained only as the booster's eval_metric label.
WATCH_METRIC = "auc"

# Optional driver-memory down-sampling of the pandas frame (mirrors the sibling notebooks). None = full set.
SAMPLE_FRACTION = None

# Cap on distinct one-hot categories for `segment` (matches the sibling notebooks' _build_model_matrix).
SEGMENT_TOP_CATS = 8

# XGBoost params that must be cast to int when reloaded from MLflow (params are stored as strings).
INT_PARAMS = {"max_depth", "min_child_weight"}

# ---------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1a. Widgets — REQUIRED upstream run ids + runtime knobs
# MAGIC The two upstream run ids are **required** and fail-fast if blank (the same discipline HPO uses for
# MAGIC `selection_run_id`) — nothing is fabricated:
# MAGIC
# MAGIC * `hpo_champion_run_id` — the `fop_hpo_champion` MLflow run; its logged `best_*` params ARE `best_params`.
# MAGIC * `selection_run_id` — the feature-selection run; its `selected_features.json` fixes the exact subset.
# MAGIC
# MAGIC `as_of_date` is left blank on purpose: it then **inherits the champion run's `as_of_date`**, guaranteeing
# MAGIC the reproduced spine / split / frozen vocab are exactly the ones HPO tuned. (Unlike HPO, this notebook
# MAGIC never auto-derives a drifting cutoff — a mismatched cutoff would silently retune the vocabulary.)

# COMMAND ----------

import json

import mlflow
from mlflow.tracking import MlflowClient

dbutils.widgets.text(
    "hpo_champion_run_id", "",
    "REQUIRED: MLflow run_id of the fop_hpo_champion run (source of best_params).",
)
dbutils.widgets.text(
    "selection_run_id", "",
    "REQUIRED: MLflow run_id of the feature-selection run (source of selected_features.json).",
)
dbutils.widgets.text(
    "as_of_date", "",
    "Cutoff / as-of (blank = INHERIT the champion run's as_of_date so the split matches what HPO tuned).",
)
dbutils.widgets.dropdown(
    "imbalance_strategy", "auto",
    ["auto", "undersample_fixed_spw1", "tune_spw_no_undersample"],
    "Imbalance handling. 'auto' = inherit the strategy the champion was tuned under (recommended).",
)
dbutils.widgets.dropdown("calibrate", "yes", ["yes", "no"], "Fit a probability calibrator on the VAL holdout.")
dbutils.widgets.dropdown(
    "calibration_method", "isotonic", ["isotonic", "sigmoid"],
    "Calibrator family: isotonic (non-parametric) or sigmoid (Platt).",
)
dbutils.widgets.text(
    "registered_model_name", "mlops_data_science.models.first_order_propensity",
    "Three-level UC model name (catalog.schema.model).",
)
dbutils.widgets.dropdown(
    "alias_mode", "auto", ["auto", "champion", "challenger", "none"],
    "auto = @champion if none exists else @challenger. 'champion' forces promotion (see the gate below).",
)

HPO_CHAMPION_RUN_ID = dbutils.widgets.get("hpo_champion_run_id").strip()
SELECTION_RUN_ID = dbutils.widgets.get("selection_run_id").strip()
CALIBRATE = dbutils.widgets.get("calibrate") == "yes"
CAL_METHOD = dbutils.widgets.get("calibration_method")
REGISTERED_MODEL_NAME = dbutils.widgets.get("registered_model_name").strip()
ALIAS_MODE = dbutils.widgets.get("alias_mode")
_IMBALANCE_WIDGET = dbutils.widgets.get("imbalance_strategy")

if not HPO_CHAMPION_RUN_ID:
    raise ValueError(
        "`hpo_champion_run_id` widget is empty. Set it to the MLflow run_id of the fop_hpo_champion run "
        "(the one that logged best_* params). Refusing to fabricate hyperparameters — FAILING FAST."
    )
if not SELECTION_RUN_ID:
    raise ValueError(
        "`selection_run_id` widget is empty. Set it to the MLflow run_id of the feature-selection run "
        "(the one that logged selected_features.json). Refusing to guess a feature subset — FAILING FAST."
    )
if REGISTERED_MODEL_NAME.count(".") != 2:
    raise ValueError(
        f"registered_model_name must be a three-level UC name 'catalog.schema.model'; got "
        f"'{REGISTERED_MODEL_NAME}'. FAILING FAST."
    )

print(f"HPO champion run : {HPO_CHAMPION_RUN_ID}")
print(f"Selection run    : {SELECTION_RUN_ID}")
print(f"UC model name    : {REGISTERED_MODEL_NAME}")
print(f"Calibrate        : {CALIBRATE} ({CAL_METHOD})   Alias mode: {ALIAS_MODE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1b. Load `best_params` (from the champion run) + the SELECTED feature list
# MAGIC `best_params` is reconstructed from the `best_*` params the HPO champion logged (strings -> numerics).
# MAGIC The imbalance strategy, `target_neg_per_pos` and `as_of_date` are read back from the same run so the
# MAGIC refit reproduces the tuned setup rather than re-deciding it here.

# COMMAND ----------

_client = MlflowClient()


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
            f"final_non_store_features — nothing selected to train on. FAILING FAST."
        )
    return store, non_store


def load_best_params_from_champion(run_id):
    """Reconstruct best_params from the champion run's `best_*` params (MLflow stores params as strings)."""
    run = _client.get_run(run_id)
    params = run.data.params
    tags = run.data.tags
    if tags.get("stage") != "hpo_champion":
        print(f"WARNING: run {run_id} has stage tag '{tags.get('stage')}' (expected 'hpo_champion'). "
              f"Proceeding, but double-check this is the intended champion run.")
    best = {}
    for k, v in params.items():
        if not k.startswith("best_"):
            continue
        name = k[len("best_"):]
        if name in INT_PARAMS:
            best[name] = int(round(float(v)))
        else:
            best[name] = float(v)
    if not best:
        raise ValueError(
            f"No `best_*` params found on run {run_id}; this does not look like a fop_hpo_champion run. "
            f"Refusing to train without tuned hyperparameters — FAILING FAST."
        )
    return best, params, tags


BEST_PARAMS, _champ_params_all, _champ_tags = load_best_params_from_champion(HPO_CHAMPION_RUN_ID)

# Imbalance strategy: inherit from the champion unless the widget forces a value.
_champ_imbalance = _champ_tags.get("imbalance_strategy", "undersample_fixed_spw1")
if _IMBALANCE_WIDGET == "auto":
    IMBALANCE_STRATEGY = _champ_imbalance
else:
    IMBALANCE_STRATEGY = _IMBALANCE_WIDGET
    if IMBALANCE_STRATEGY != _champ_imbalance:
        print(f"WARNING: overriding imbalance strategy to '{IMBALANCE_STRATEGY}' but the champion was tuned "
              f"under '{_champ_imbalance}'. best_params may not be optimal for the override.")

# target_neg_per_pos: inherit the champion's undersampling ratio if it logged one.
if "target_neg_per_pos" in _champ_params_all:
    TARGET_NEG_PER_POS = int(round(float(_champ_params_all["target_neg_per_pos"])))

# Champion's own early-stopping round count (cross-checked against the re-derived value later).
_champ_num_round = None
if "champion_num_boost_round" in _champ_params_all:
    _champ_num_round = int(round(float(_champ_params_all["champion_num_boost_round"])))

# Selected feature subset (consumed, not guessed). segment included ONLY if selection retained it.
SELECTED_STORE_FEATURES, _final_non_store = load_selected_features_from_run(SELECTION_RUN_ID)
_unsupported_non_store = [f for f in _final_non_store if f not in SUPPORTED_NON_STORE_FEATURES]
if _unsupported_non_store:
    print(f"WARNING: selection listed non-store features with no encoder here -> SKIPPED: {_unsupported_non_store}")
INCLUDE_SEGMENT = "segment" in _final_non_store  # DERIVED, not hardcoded
CATEGORICAL_FEATURES = ["segment"] if INCLUDE_SEGMENT else []

if not SELECTED_STORE_FEATURES and not INCLUDE_SEGMENT:
    raise ValueError("Resolved feature set is empty (no store features and segment not included). FAILING FAST.")

print(f"best_params ({len(BEST_PARAMS)}): {BEST_PARAMS}")
print(f"Imbalance strategy      : {IMBALANCE_STRATEGY}  (target_neg_per_pos={TARGET_NEG_PER_POS})")
print(f"Champion num_boost_round: {_champ_num_round}")
print(f"Selected store features : {SELECTED_STORE_FEATURES}")
print(f"Include segment         : {INCLUDE_SEGMENT}  (from final_non_store_features={_final_non_store})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 1 — Spine (label + eligibility)
# MAGIC Reused verbatim from `first_order_propensity_hpo.py`: the LABEL / ELIGIBILITY half of the v7 ETL query,
# MAGIC emitting exactly `internal_user_id`, `feature_ts`, `output`. Status filter kept as
# MAGIC `IN ('COMPLETE','CANCELLED')`; the first-order anti-join drops anyone who had already ordered as of the
# MAGIC cutoff. `as_of_date` is inherited from the champion run so the population matches what HPO tuned.

# COMMAND ----------

from datetime import datetime, timedelta


def _sql_in_list(values):
    """Render a Python list as a SQL IN-list of single-quoted literals."""
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def _parse_ts(ts_str: str) -> datetime:
    """Parse 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' into a datetime."""
    return datetime.fromisoformat(ts_str.strip())


def resolve_as_of(widget_value: str, champion_as_of) -> str:
    """Cutoff resolver. Blank widget INHERITS the champion's as_of_date (so the tuned split reproduces).

    Never auto-derives a drifting cutoff here — a different cutoff would silently refit the frozen vocab and
    break the match with what HPO tuned. FAILS FAST if neither a widget value nor a champion as_of exists.
    """
    widget_value = (widget_value or "").strip()
    if widget_value:
        return widget_value
    if champion_as_of:
        print("as_of_date blank -> inheriting the champion run's as_of_date to reproduce the tuned split.")
        return str(champion_as_of)
    raise ValueError(
        "as_of_date is blank AND the champion run logged no as_of_date. Set the as_of_date widget explicitly "
        "to the cutoff HPO tuned on. FAILING FAST."
    )


def build_spine_sql(cutoff: str, performance_end: str, ist: int) -> str:
    """LABEL + ELIGIBILITY only. Emits (internal_user_id, feature_ts, output). Reused verbatim from HPO."""
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


cutoff_ts = resolve_as_of(dbutils.widgets.get("as_of_date"), _champ_params_all.get("as_of_date"))
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
# MAGIC `create_training_set` does the as-of join on the **selected** store features only, then `segment` is
# MAGIC attached **current-state** (finding-#1 leakage caveat, preserved from HPO) and the v7 engagement gate is
# MAGIC applied against the feature columns. Identical to HPO so the reproduced training frame matches.

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

# --- Attach `segment` CURRENT-STATE (finding-#1 caveat) + v7 engagement gate ----------
analysis_df = training_df

if INCLUDE_SEGMENT:
    # Reproduces v7's `coalesce(usm.segment_name, 'No Segment') AS segment`, joined current-state.
    segment_df = spark.sql(f"SELECT internal_user_id, segment_name FROM {USER_SEGMENT_MAPPING}")
    analysis_df = (
        analysis_df.join(segment_df, on=ENTITY_KEY, how="left")
        .withColumnRenamed("segment_name", "segment")
        .fillna({"segment": "No Segment"})
    )

# v7's engagement gate, applied AGAINST THE FEATURE TABLE columns (only the recency columns that were
# actually selected). NULL recency == "never", so IS NOT NULL reproduces v7's *_search IS NOT NULL, and
# days_since_last_app_open < 90 reproduces v7's coalesce(...,9999) < 90.
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
    print("No recency/app-open columns in the selected subset -> engagement gate skipped (documented "
          "divergence unless those columns are selected).")

analysis_df = analysis_df.cache()
_n_analysis = analysis_df.count()
print(f"Analysis (PIT training) set: {_n_analysis:,} rows, {len(analysis_df.columns)} columns")
print(f"Feature set for training   : {SELECTED_STORE_FEATURES + CATEGORICAL_FEATURES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 3 — The model class = frozen preprocessing + booster + calibrator
# MAGIC
# MAGIC The `pyfunc` is defined **here, before training**, on purpose. Its frozen transform
# MAGIC (`_build_matrix`) is the SAME code path that builds the training matrices below — so train/serve
# MAGIC symmetry is guaranteed *by construction*, not by two hand-kept-in-sync implementations. This is the
# MAGIC strongest form of the audit's train/serve-skew fix.
# MAGIC
# MAGIC ### What was frozen vs. what v7 re-derived every run
# MAGIC v7's model matrix re-computed `top_segments = df['segment'].value_counts().head(8)` on the **whole
# MAGIC dataset** each run and then `pd.get_dummies(..., drop_first=True)` — so the encoding could differ
# MAGIC between training and serving (train/serve skew) and leaked test-fold category frequencies into the
# MAGIC vocabulary. Here the segment top-N vocabulary is fit on the **TRAIN fold only**, stored as frozen state
# MAGIC (with an explicit `Other` catch-all == `handle_unknown='ignore'`), and applied unchanged everywhere.
# MAGIC There is **no** `value_counts().head(N)` / `get_dummies` re-derivation inside `predict`.
# MAGIC
# MAGIC ### Why v7's numeric BUCKETING is deliberately NOT applied
# MAGIC v7 also bucketed the numerics (`bucket_card_txn` / `bucket_search` / `bucket_coins` / `bucket_recency`
# MAGIC / `bucket_ad`) and one-hot-encoded them. HPO, however, **tuned on the RAW numeric values** (XGBoost's
# MAGIC native missing handling for NULL recency) + the segment one-hot — see the HPO notebook's `_build_model_matrix`.
# MAGIC Because this notebook consumes `best_params`, it MUST use the **same preprocessing HPO tuned**, or the
# MAGIC hyperparameters would apply to a different, higher-dimensional representation. So the only
# MAGIC data-dependent encoder in the tuned pipeline is the segment vocabulary — and that is what gets frozen.
# MAGIC (If a future model tunes on bucketed numerics, the identical freezing mechanism — store the boundaries,
# MAGIC apply them in `_build_matrix` — extends to those `bucket_*` cut points; none exist in this tuned rep.)

# COMMAND ----------

import mlflow.pyfunc


class FirstOrderPropensityModel(mlflow.pyfunc.PythonModel):
    """Self-contained First-Order-Propensity model: frozen preprocessing -> XGBoost booster -> calibrator.

    All state travels via `artifacts`:
      * ``preprocessing`` — JSON: numeric_features, categorical_features, frozen segment vocab, feature_names
        (frozen column order), best_num_round, calibration_method, decile_score_edges, f2_threshold.
      * ``booster``       — XGBoost booster saved via ``Booster.save_model`` (predicted with a frozen
        iteration_range so only the early-stopping-selected trees are used).
      * ``calibrator``    — OPTIONAL pickled sklearn calibrator (IsotonicRegression or LogisticRegression).

    ``predict`` runs the WHOLE chain so preprocessing always travels with the model. Output columns:
      * ``raw_score``              — uncalibrated probability == the monotonic ranking score.
      * ``calibrated_probability`` — calibrated probability (== raw_score when no calibrator was fit).
      * ``decile``                 — 1..10 via FROZEN VAL score boundaries (1 = top decile); rank-only,
                                     robust to miscalibration and independent of the scoring batch's size.
    """

    # ---- frozen preprocessing (the SAME transform used to build the training matrices) --------------
    @staticmethod
    def _build_matrix(model_input, pp):
        """Raw float32 numerics (NaN kept for XGBoost) + segment one-hot on a FROZEN train-fit vocab.

        Mirrors HPO's `build_model_matrix`, but the vocabulary + final column order are passed in (frozen),
        never re-derived. Unseen/rare categories fold into the explicit `Other` bucket, and the final
        reindex to `feature_names` (fill 0.0) gives `handle_unknown='ignore'` semantics.
        """
        import re

        import numpy as np
        import pandas as pd

        pdf_part = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        parts = []
        for c in pp["numeric_features"]:
            if c in pdf_part.columns:
                col = pd.to_numeric(pdf_part[c], errors="coerce").astype("float32")
            else:
                col = pd.Series(np.nan, index=pdf_part.index, dtype="float32")
            parts.append(col.rename(c))
        for c in pp["categorical_features"]:
            cats = pp["vocab"][c]
            if c in pdf_part.columns:
                vals = pdf_part[c].astype("object").fillna("No Segment")
            else:
                vals = pd.Series("No Segment", index=pdf_part.index, dtype="object")
            vals = vals.where(vals.isin(cats), "Other")          # unseen/rare -> explicit Other bucket
            vals = pd.Categorical(vals, categories=cats)          # FIXED categories -> stable, complete columns
            dummies = pd.get_dummies(vals, prefix=c, dtype="float32")
            dummies.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", str(dc)) for dc in dummies.columns]
            dummies.index = pdf_part.index
            parts.append(dummies)
        X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=pdf_part.index)
        feature_names = pp.get("feature_names")
        if feature_names:  # None only during the fit-time TRAIN build that DEFINES feature_names
            X = X.reindex(columns=feature_names, fill_value=0.0)
        return X

    @staticmethod
    def _apply_calibration(uncal, method, calibrator):
        """Map uncalibrated -> calibrated probabilities. Returns uncal unchanged when no calibrator."""
        import numpy as np

        uncal = np.asarray(uncal, dtype="float64")
        if calibrator is not None and method == "isotonic":
            out = calibrator.predict(uncal)
        elif calibrator is not None and method == "sigmoid":
            eps = 1e-6
            p = np.clip(uncal, eps, 1.0 - eps)
            z = np.log(p / (1.0 - p)).reshape(-1, 1)             # Platt scales the log-odds
            out = calibrator.predict_proba(z)[:, 1]
        else:
            out = uncal
        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def _assign_decile(raw_score, edges):
        """Decile from FROZEN VAL score quantile boundaries. 1 = top decile (highest score)."""
        import numpy as np

        raw_score = np.asarray(raw_score, dtype="float64")
        edges = np.asarray(edges, dtype="float64")               # 9 inner quantile cut points, ascending
        k = np.searchsorted(edges, raw_score, side="right")      # 0..9 ; 9 = highest scores
        return (10 - k).astype("int64")

    # ---- MLflow lifecycle ---------------------------------------------------------------------------
    def load_context(self, context):
        import json
        import pickle

        import xgboost as xgb

        with open(context.artifacts["preprocessing"], "r") as fh:
            self._pp = json.load(fh)
        self._booster = xgb.Booster()
        self._booster.load_model(context.artifacts["booster"])
        self._calibrator = None
        cal_path = context.artifacts.get("calibrator")
        if cal_path:
            with open(cal_path, "rb") as fh:
                self._calibrator = pickle.load(fh)

    def predict(self, context, model_input, params=None):
        import numpy as np
        import pandas as pd
        import xgboost as xgb

        X = self._build_matrix(model_input, self._pp)
        dm = xgb.DMatrix(X.to_numpy(dtype="float32"), feature_names=list(X.columns))
        best_num_round = int(self._pp["best_num_round"])
        raw = self._booster.predict(dm, iteration_range=(0, best_num_round))
        cal = self._apply_calibration(raw, self._pp.get("calibration_method", "none"), self._calibrator)
        dec = self._assign_decile(raw, self._pp["decile_score_edges"])
        return pd.DataFrame(
            {
                "raw_score": np.asarray(raw, dtype="float64"),
                "calibrated_probability": np.asarray(cal, dtype="float64"),
                "decile": np.asarray(dec, dtype="int64"),
            },
            index=X.index,
        )


print("FirstOrderPropensityModel defined (frozen preprocessing + booster + optional calibrator).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 4 — Split (HPO seed/logic) + fit the frozen vocab on TRAIN only
# MAGIC Rows are split FIRST (same seed / fractions / stratification as HPO), THEN the segment one-hot
# MAGIC vocabulary is fit on the **TRAIN split only** and frozen. The training matrices are built with the
# MAGIC model's own `_build_matrix`, so they are byte-for-byte the transform `predict` will run. Test labels
# MAGIC are untouched here.

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

    Identical to HPO's fit_segment_vocab. Returns {feature: [categories..., 'Other']} — a FIXED category set
    applied unchanged to val/test AND frozen into the model.
    """
    vocab = {}
    for c in categorical_features:
        vals = pdf_train[c].astype("object").fillna("No Segment")
        cats = list(vals.value_counts().head(top_cats).index)
        if "Other" not in cats:
            cats.append("Other")
        vocab[c] = cats
    return vocab


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


# SPLIT ROWS FIRST (same seed/logic as HPO), then fit the vocab on TRAIN only. Test labels stay sealed.
_idx = pdf.index.to_numpy()
_idx_trv, _idx_test = train_test_split(
    _idx, test_size=TEST_FRACTION, random_state=RANDOM_STATE, stratify=y_all.to_numpy(),
)
_idx_tr_raw, _idx_val = train_test_split(
    _idx_trv, test_size=VAL_FRACTION, random_state=RANDOM_STATE, stratify=y_all.loc[_idx_trv].to_numpy(),
)

# Vocabulary learned from the TRAIN split ONLY -> no train/test leakage, matches what HPO froze.
_vocab = fit_segment_vocab(pdf.loc[_idx_tr_raw], _categorical_present, SEGMENT_TOP_CATS)

# FROZEN preprocessing state. feature_names starts None: the TRAIN build below DEFINES the column order.
PREPROCESSING = {
    "numeric_features": _numeric_present,
    "categorical_features": _categorical_present,
    "vocab": _vocab,
    "feature_names": None,
}

# Build matrices with the MODEL's own transform (single source of truth for train/serve symmetry).
X_tr_raw = FirstOrderPropensityModel._build_matrix(pdf.loc[_idx_tr_raw], PREPROCESSING)
PREPROCESSING["feature_names"] = list(X_tr_raw.columns)  # freeze the column order from the TRAIN fold
FEATURE_NAMES = PREPROCESSING["feature_names"]
X_val = FirstOrderPropensityModel._build_matrix(pdf.loc[_idx_val], PREPROCESSING)
X_test = FirstOrderPropensityModel._build_matrix(pdf.loc[_idx_test], PREPROCESSING)

y_tr_raw = y_all.loc[_idx_tr_raw]
y_val = y_all.loc[_idx_val]
y_test = y_all.loc[_idx_test]  # NOT inspected until the single final evaluation

# Methodology fix (a): the two mutually-exclusive imbalance strategies (honoring the champion's choice).
if IMBALANCE_STRATEGY == "undersample_fixed_spw1":
    X_tr, y_tr = undersample_majority(X_tr_raw, y_tr_raw, TARGET_NEG_PER_POS, RANDOM_STATE)
    print(f"Undersampled TRAIN to {TARGET_NEG_PER_POS}:1 — dropped {len(y_tr_raw) - len(y_tr):,} negatives. "
          f"scale_pos_weight FIXED at 1.")
else:
    X_tr, y_tr = X_tr_raw, y_tr_raw
    print("No undersampling — scale_pos_weight taken from best_params (population prior in TRAIN).")

print(f"Feature columns ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")
print(f"TRAIN: {X_tr.shape[0]:,} rows (rate {y_tr.mean():.3%})")
print(f"VAL  : {X_val.shape[0]:,} rows (rate {y_val.mean():.3%})   [population prior — calibration/threshold holdout]")
print(f"TEST : {X_test.shape[0]:,} rows   [held out; labels untouched until the single final evaluation]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 5 — Fit the champion XGBoost (best_params, FROZEN round count — VAL is NOT consulted)
# MAGIC The booster is fit on the (undersampled) **TRAIN** fold with `best_params` for the champion's **frozen**
# MAGIC number of boosting rounds (`champion_num_boost_round`, carried over from the HPO champion run). The
# MAGIC final fit does **NOT** early-stop on VAL and never puts VAL in its `evals` watchlist.
# MAGIC
# MAGIC **Why no VAL early stopping here (the fix):** HPO already tuned the hyperparameters — including the
# MAGIC boosting-round count — against VAL. If the final fit early-stopped on VAL again, VAL would be
# MAGIC double-used and the calibration + F2-threshold + decile edges derived from it in SECTION 6 would be
# MAGIC optimistically biased. By freezing the round count from HPO and keeping VAL out of the fit entirely,
# MAGIC **VAL becomes a genuine holdout** used ONLY for calibration / threshold / decile-edge selection, and
# MAGIC TEST stays sealed for the single final eval. The fit stays TRAIN-only (VAL is not folded back in).
# MAGIC
# MAGIC If the champion run logged no `champion_num_boost_round`, this cell **FAILS FAST** rather than
# MAGIC silently falling back to VAL early stopping.

# COMMAND ----------

import xgboost as xgb

# FAIL FAST if the frozen champion round count is unavailable — never silently re-introduce VAL early
# stopping (that would double-use the calibration/threshold holdout). `_champ_num_round` was loaded from the
# HPO champion run in cell 1b.
if _champ_num_round is None:
    raise ValueError(
        "The HPO champion run logged no `champion_num_boost_round`, so the frozen boosting-round count is "
        "unavailable. Refusing to early-stop the final fit on VAL — that would double-use the VAL fold that "
        "calibration + threshold selection rely on as a genuine holdout. Re-run HPO so the champion logs "
        "champion_num_boost_round, or point hpo_champion_run_id at a run that has it. FAILING FAST."
    )
best_num_round = _champ_num_round  # FROZEN: exactly the rounds HPO's champion was evaluated at

champ_params = {
    "objective": "binary:logistic",
    "eval_metric": WATCH_METRIC,
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
}
champ_params.update(BEST_PARAMS)  # tuned params (includes scale_pos_weight ONLY if it was tuned)
if IMBALANCE_STRATEGY != "tune_spw_no_undersample":
    champ_params["scale_pos_weight"] = 1.0  # fix (a): no double correction under undersampling

_dtrain = xgb.DMatrix(X_tr.to_numpy(dtype="float32"), label=y_tr.to_numpy(dtype="int32"), feature_names=FEATURE_NAMES)
# _dval is a scoring matrix ONLY — it feeds the VAL calibration/threshold/decile fit in SECTION 6. It is
# deliberately NOT passed to xgb.train below (no `evals`, no `early_stopping_rounds`) so VAL never touches
# the fit and stays a genuine holdout.
_dval = xgb.DMatrix(X_val.to_numpy(dtype="float32"), label=y_val.to_numpy(dtype="int32"), feature_names=FEATURE_NAMES)

champion = xgb.train(
    champ_params,
    _dtrain,
    num_boost_round=best_num_round,  # FROZEN from HPO — no early stopping, no VAL watchlist
    verbose_eval=False,
)
print(f"Champion fit on TRAIN ({X_tr.shape[0]:,} rows) for the FROZEN {best_num_round} rounds "
      f"(champion_num_boost_round from HPO). VAL was NOT used in the fit.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 6 — Threshold (on VAL) + calibration (on VAL) + decile boundaries (on VAL)
# MAGIC Everything decision-related is fit on the population-rate **VAL** holdout — never on the undersampled
# MAGIC train, never on test:
# MAGIC * **Calibration** — isotonic or Platt (sigmoid) fit on VAL uncalibrated scores vs VAL labels.
# MAGIC * **F2 threshold** — swept on the VAL *served* probability (calibrated if calibrating), applied to test
# MAGIC   exactly once later.
# MAGIC * **Decile boundaries** — quantiles of the VAL uncalibrated (ranking) score, frozen so `predict` can
# MAGIC   assign a population-anchored decile without needing the whole scoring batch.
# MAGIC
# MAGIC Both calibrated and uncalibrated quality (Brier) are logged.

# COMMAND ----------

from sklearn.metrics import brier_score_loss, fbeta_score, roc_auc_score


def best_f2_threshold(y_true, proba):
    """Threshold that maximizes F2 on the GIVEN fold. Call on VALIDATION, apply once to test. Returns
    (threshold, f2_at_thr). Never choose the threshold on the held-out test set (audit Tier 1 #3)."""
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


# Uncalibrated VAL scores (FROZEN round count). VAL never touched the fit, so it is a genuine holdout here.
val_uncal = champion.predict(_dval, iteration_range=(0, best_num_round))
_y_val = y_val.to_numpy()

# --- Calibrator fit on the population-rate VAL holdout ------------------------------------------------
calibrator = None
CALIBRATION_METHOD_EFF = "none"
if CALIBRATE:
    if CAL_METHOD == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(val_uncal, _y_val)
        CALIBRATION_METHOD_EFF = "isotonic"
    else:
        from sklearn.linear_model import LogisticRegression

        _eps = 1e-6
        _p = np.clip(val_uncal, _eps, 1.0 - _eps)
        _z = np.log(_p / (1.0 - _p)).reshape(-1, 1)
        calibrator = LogisticRegression(C=1e6, solver="lbfgs")  # near-unregularized Platt scaling on the logit
        calibrator.fit(_z, _y_val)
        CALIBRATION_METHOD_EFF = "sigmoid"
    print(f"Fitted {CALIBRATION_METHOD_EFF} calibrator on the VAL holdout ({len(_y_val):,} rows, "
          f"rate {_y_val.mean():.3%}).")
else:
    print("CALIBRATE=no -> shipping uncalibrated scores (raw_score == calibrated_probability).")


def _apply_cal(uncal):
    """Notebook-side calibration using the model's OWN static method (keeps train == serve)."""
    return FirstOrderPropensityModel._apply_calibration(uncal, CALIBRATION_METHOD_EFF, calibrator)


# --- F2 threshold on VAL *served* probability (monotonic; directly applicable downstream) -------------
val_served = _apply_cal(val_uncal)
f2_threshold, val_f2_at_thr = best_f2_threshold(_y_val, val_served)

# --- Decile boundaries: quantiles of VAL uncalibrated (ranking) scores -------------------------------
decile_edges = [float(q) for q in np.quantile(val_uncal, np.linspace(0.1, 0.9, 9))]

# --- VAL quality (both calibrated + uncalibrated) ----------------------------------------------------
val_roc_auc = float(roc_auc_score(_y_val, val_uncal))          # rank metric — identical for cal/uncal
val_brier_uncal = float(brier_score_loss(_y_val, val_uncal))
val_brier_served = float(brier_score_loss(_y_val, val_served))
print(f"VAL ROC-AUC          : {val_roc_auc:.4f}")
print(f"VAL Brier (uncal)    : {val_brier_uncal:.5f}")
print(f"VAL Brier (served)   : {val_brier_served:.5f}  ({'calibrated' if CALIBRATE else 'uncalibrated'})")
print(f"F2 threshold on VAL  : {f2_threshold:.3f} (val F2 = {val_f2_at_thr:.4f})")

# Finalize the FROZEN preprocessing state carried by the model.
PREPROCESSING.update({
    "best_num_round": int(best_num_round),
    "calibration_method": CALIBRATION_METHOD_EFF,
    "decile_score_edges": decile_edges,
    "f2_threshold": float(f2_threshold),
    "as_of_date": cutoff_ts,
    "selected_store_features": list(SELECTED_STORE_FEATURES),
    "include_segment": bool(INCLUDE_SEGMENT),
})

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 7 — Persist artifacts + build signature/input_example
# MAGIC The booster, the frozen preprocessing JSON and the optional calibrator are written to disk and passed
# MAGIC to `log_model(artifacts=...)`. The signature is inferred from an `input_example` (the raw selected
# MAGIC feature columns) and the assembled model's own output — mandatory for UC registration.

# COMMAND ----------

import os
import pickle
import tempfile

import sklearn
from mlflow.models import infer_signature

_art_dir = tempfile.mkdtemp()
_booster_path = os.path.join(_art_dir, "booster.json")
champion.save_model(_booster_path)

_pp_path = os.path.join(_art_dir, "preprocessing.json")
with open(_pp_path, "w") as fh:
    json.dump(PREPROCESSING, fh, indent=2)

MODEL_ARTIFACTS = {"booster": _booster_path, "preprocessing": _pp_path}
if calibrator is not None:
    _cal_path = os.path.join(_art_dir, "calibrator.pkl")
    with open(_cal_path, "wb") as fh:
        pickle.dump(calibrator, fh)
    MODEL_ARTIFACTS["calibrator"] = _cal_path
print(f"Model artifacts: {list(MODEL_ARTIFACTS)}")

# input_example = the RAW selected feature columns the model expects at serve time.
_input_cols = _numeric_present + _categorical_present
input_example = pdf.loc[_idx_test, _input_cols].head(5).reset_index(drop=True).copy()
# Cast numerics to double so the signature never types a count column as int-with-missing (MLflow enforces
# integer columns strictly; recency columns legitimately carry NaN). Categorical `segment` stays object.
if _numeric_present:
    input_example[_numeric_present] = input_example[_numeric_present].astype("float64")

# Reproduce the model output locally via the SAME static methods predict() uses -> exact output schema.
_ex_X = FirstOrderPropensityModel._build_matrix(input_example, PREPROCESSING)
_ex_raw = champion.predict(
    xgb.DMatrix(_ex_X.to_numpy(dtype="float32"), feature_names=list(_ex_X.columns)),
    iteration_range=(0, best_num_round),
)
_ex_cal = FirstOrderPropensityModel._apply_calibration(_ex_raw, CALIBRATION_METHOD_EFF, calibrator)
_ex_dec = FirstOrderPropensityModel._assign_decile(_ex_raw, decile_edges)
output_example = pd.DataFrame({
    "raw_score": np.asarray(_ex_raw, dtype="float64"),
    "calibrated_probability": np.asarray(_ex_cal, dtype="float64"),
    "decile": np.asarray(_ex_dec, dtype="int64"),
})
signature = infer_signature(input_example, output_example)
print("Signature inferred:")
print(signature)

# Pin the training-time library versions so the registered environment is reproducible.
import cloudpickle

PIP_REQUIREMENTS = [
    f"mlflow=={mlflow.__version__}",
    f"xgboost=={xgb.__version__}",
    f"scikit-learn=={sklearn.__version__}",
    f"pandas=={pd.__version__}",
    f"numpy=={np.__version__}",
    f"cloudpickle=={cloudpickle.__version__}",
]
print(f"pip_requirements: {PIP_REQUIREMENTS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 8 — Honest TEST evaluation (test touched ONCE) + register to UC
# MAGIC The held-out TEST fold is scored **once** here at the true population base rate: ROC-AUC, PR-AUC,
# MAGIC top-decile lift + capture, and F2 at the VAL-chosen threshold. Then the assembled `pyfunc` is logged
# MAGIC and **registered to Unity Catalog** with the signature + input_example, and all metrics/params are
# MAGIC logged to the same registration run.

# COMMAND ----------

from sklearn.metrics import average_precision_score


def _top_decile_lift(y_true, proba):
    """Booker rate in the top-10% by score / population base rate. == v7's decile-1 lift_vs_base."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    k = max(int(round(n * 0.10)), 1)
    top = np.argsort(-proba)[:k]
    base = y_true.mean()
    if base <= 0:
        return 0.0
    return float(y_true[top].mean() / base)


# --- The single, honest TEST touch (population base rate) ---------------------------------------------
_dtest = xgb.DMatrix(X_test.to_numpy(dtype="float32"), feature_names=FEATURE_NAMES)
test_uncal = champion.predict(_dtest, iteration_range=(0, best_num_round))
test_served = _apply_cal(test_uncal)
_y_test = y_test.to_numpy()

_n_test = len(_y_test)
_k_test = max(int(round(_n_test * 0.10)), 1)
_top_test = np.argsort(-test_uncal)[:_k_test]                    # rank on the (uncalibrated) ranking score

test_base_rate = float(_y_test.mean())
test_roc_auc = float(roc_auc_score(_y_test, test_served))        # threshold-free (== on uncal; monotonic)
test_pr_auc = float(average_precision_score(_y_test, test_served))
test_top_decile_lift = _top_decile_lift(_y_test, test_uncal)
test_top_decile_capture = float(_y_test[_top_test].sum() / max(int(_y_test.sum()), 1))
test_f2_at_val_thr = float(fbeta_score(_y_test, (test_served >= f2_threshold).astype(int), beta=2, zero_division=0))
test_brier_uncal = float(brier_score_loss(_y_test, test_uncal))
test_brier_served = float(brier_score_loss(_y_test, test_served))

print("=== Honest TEST metrics (population base rate) ===")
print(f"Base rate            : {test_base_rate:.3%}  ({int(_y_test.sum()):,} positives / {_n_test:,})")
print(f"ROC-AUC              : {test_roc_auc:.4f}")
print(f"PR-AUC               : {test_pr_auc:.4f}")
print(f"Top-decile lift      : {test_top_decile_lift:.3f}x")
print(f"Top-decile capture   : {test_top_decile_capture:.3%}  (share of ALL bookers in the top 10%)")
print(f"F2 @ val threshold   : {test_f2_at_val_thr:.4f}  (threshold {f2_threshold:.3f} chosen on VAL)")
print(f"Brier (served)       : {test_brier_served:.5f}  (uncal {test_brier_uncal:.5f})")

# Decile lift table (v7-style: Decile 1 = top 10%).
_eval = pd.DataFrame({"actual": _y_test, "proba": test_uncal})
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

# --- Register: set UC registry, log the pyfunc + metrics, resolve the new version --------------------
import time

mlflow.set_registry_uri("databricks-uc")
# Registry ops MUST target UC. `_client` (created earlier) reads tracking runs; use a UC-scoped client for
# registration/alias so version lookups and aliases land in Unity Catalog, not the workspace registry.
_uc_client = MlflowClient(registry_uri="databricks-uc")


def _resolve_registered_version(client, model_name, run_id, max_attempts=6, base_delay=0.5):
    """Resolve the just-registered UC version by EXACT `run_id` match ONLY — never 'latest'.

    ``ModelInfo.registered_model_version`` is not reliably populated across MLflow versions, and falling back
    to the newest version is unsafe: under concurrent registrations the newest version could belong to a
    DIFFERENT model, which would then be aliased. So the version is found by matching model versions on THIS
    run's id. Registration can lag a moment behind ``log_model``, so we poll with bounded exponential backoff
    and, if no version whose ``run_id``/``source`` matches this run appears within the bound, we RAISE.
    """
    def _matches_run(mv):
        if getattr(mv, "run_id", None) == run_id:
            return True
        # Some backends leave run_id empty but set source to 'runs:/<run_id>/<artifact_path>'.
        src = getattr(mv, "source", "") or ""
        return f"/{run_id}/" in src or src.endswith(f"/{run_id}")

    last_err = None
    for attempt in range(max_attempts):
        try:
            mvs = client.search_model_versions(f"name='{model_name}'")
            exact = [mv for mv in mvs if _matches_run(mv)]
            if exact:
                # Exact run_id match only. If (pathologically) several versions map to this one run, take the
                # newest OF THOSE MATCHES — still never an unrelated version.
                return str(sorted(exact, key=lambda m: int(m.version))[-1].version)
        except Exception as exc:  # transient search / propagation error -> retry within the bound
            last_err = exc
        if attempt < max_attempts - 1:
            time.sleep(min(base_delay * (2 ** attempt), 8.0))
    raise RuntimeError(
        f"Could not resolve a registered version of {model_name} whose run_id matches {run_id} after "
        f"{max_attempts} attempts. Refusing to fall back to the latest version (it could be an unrelated / "
        f"concurrently registered model). Last search error: {last_err}."
    )


with mlflow.start_run(run_name="fop_model_registration") as run:
    mlflow.set_tags({
        "project": "first_order_propensity",
        "stage": "model_registration",
        "feature_table": FEATURE_TABLE,
        "hpo_champion_run_id": HPO_CHAMPION_RUN_ID,
        "selection_run_id": SELECTION_RUN_ID,
        "imbalance_strategy": IMBALANCE_STRATEGY,
        "calibration_method": CALIBRATION_METHOD_EFF,
    })
    mlflow.log_params({
        "as_of_date": cutoff_ts,
        "performance_end": performance_end_ts,
        "selected_store_features": ",".join(_numeric_present),
        "include_segment": INCLUDE_SEGMENT,
        "target_neg_per_pos": TARGET_NEG_PER_POS,
        "num_boost_round_source": "hpo_champion_frozen",  # no VAL early stopping in the final fit
        "champion_num_boost_round": best_num_round,
        "calibrate": CALIBRATE,
        "n_features": len(FEATURE_NAMES),
        **{f"best_{k}": v for k, v in BEST_PARAMS.items()},
    })
    mlflow.log_metrics({
        # VAL (holdout) — threshold + calibration were fit here.
        "val_roc_auc": val_roc_auc,
        "val_f2_threshold": f2_threshold,
        "val_f2_at_threshold": val_f2_at_thr,
        "val_brier_uncal": val_brier_uncal,
        "val_brier_served": val_brier_served,
        # TEST (touched once) — the honest headline numbers.
        "test_base_rate": test_base_rate,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
        "test_top_decile_lift": test_top_decile_lift,
        "test_top_decile_capture": test_top_decile_capture,
        "test_f2_at_val_threshold": test_f2_at_val_thr,
        "test_brier_uncal": test_brier_uncal,
        "test_brier_served": test_brier_served,
    })

    model_info = mlflow.pyfunc.log_model(
        name="model",                                   # MLflow 3 idiom (artifact_path= is the 2.x equivalent)
        python_model=FirstOrderPropensityModel(),
        artifacts=MODEL_ARTIFACTS,
        signature=signature,
        input_example=input_example,
        registered_model_name=REGISTERED_MODEL_NAME,
        pip_requirements=PIP_REQUIREMENTS,
    )
    run_id = run.info.run_id
    print(f"Logged + registered pyfunc in run {run_id}")

registered_version = _resolve_registered_version(_uc_client, REGISTERED_MODEL_NAME, run_id)
print(f"Registered {REGISTERED_MODEL_NAME} version {registered_version}")

# COMMAND ----------

# --- Best-effort round-trip check: load the logged model back and predict on the input_example --------
# Proves the whole chain (frozen preprocessing -> booster -> calibrator) survives log/load. Live-only.
try:
    _loaded = mlflow.pyfunc.load_model(model_info.model_uri)
    _rt = _loaded.predict(input_example)
    assert list(_rt.columns) == ["raw_score", "calibrated_probability", "decile"], _rt.columns
    print(f"Round-trip OK — loaded model returned columns {list(_rt.columns)} for {_rt.shape[0]} rows.")
except Exception as _exc:
    print(f"WARNING: round-trip load/predict check did not complete: {_exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 9 — Alias + the champion-promotion GATE
# MAGIC **DEFAULT (`alias_mode=auto`)**: set `@champion` only if the model has **no** champion yet; otherwise
# MAGIC set `@challenger`. A freshly trained version is **never** auto-promoted over a live champion — and even
# MAGIC an explicit `alias_mode=champion` is **REFUSED (raises)** when a champion already exists on a different
# MAGIC version. Moving a live champion is possible only via the explicit human promotion command below.
# MAGIC
# MAGIC ### Champion-promotion gate (a human decision — do NOT automate here)
# MAGIC Promoting a `@challenger` to `@champion` is a deliberate, reviewed step, not a side effect of training.
# MAGIC Before moving `@champion`:
# MAGIC 1. Compare the challenger's honest TEST metrics (ROC-AUC, top-decile lift + capture, F2 @ threshold)
# MAGIC    against the incumbent champion's, on the **same** as-of split.
# MAGIC 2. Confirm calibration (Brier) did not regress and that the decile monotonicity holds.
# MAGIC 3. Get sign-off, then move the alias explicitly (one line, run by a human):
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.tracking import MlflowClient
# MAGIC client = MlflowClient(registry_uri="databricks-uc")
# MAGIC client.set_registered_model_alias("mlops_data_science.models.first_order_propensity", "champion", "7")
# MAGIC ```
# MAGIC
# MAGIC The serving/scoring layer pins `@champion`; challengers are evaluated in shadow until promoted.

# COMMAND ----------

# --- Apply the alias per alias_mode (auto = champion if none exists, else challenger) -----------------
def _current_alias_version(client, model_name, alias):
    try:
        return client.get_model_version_by_alias(model_name, alias).version
    except Exception:
        return None


existing_champion = _current_alias_version(_uc_client, REGISTERED_MODEL_NAME, "champion")

if ALIAS_MODE == "none":
    chosen_alias = None
elif ALIAS_MODE == "auto":
    chosen_alias = "champion" if existing_champion is None else "challenger"
else:
    chosen_alias = ALIAS_MODE  # explicit 'champion' or 'challenger'

# HARD promotion gate: a new version may take @champion ONLY when the model has no champion yet. If a
# champion already exists on a DIFFERENT version, REFUSE to overwrite it here (raise) — promoting over a
# live champion is an explicit, human-gated decision, not a training side effect. This holds regardless of
# alias_mode, so `alias_mode=champion` cannot bypass the gate.
if chosen_alias == "champion" and existing_champion is not None and str(existing_champion) != str(registered_version):
    raise RuntimeError(
        f"Refusing to move @champion from live version v{existing_champion} to v{registered_version}. "
        f"Promoting over an existing champion is a human-gated decision, not a side effect of training. "
        f"This version is registered and available as a challenger candidate; compare it against the "
        f"incumbent on the same as-of split, and if it wins, run the explicit promotion command:\n"
        f"    MlflowClient(registry_uri='databricks-uc').set_registered_model_alias("
        f"'{REGISTERED_MODEL_NAME}', 'champion', '{registered_version}')"
    )

if chosen_alias is None:
    print(f"alias_mode=none -> version {registered_version} registered without an alias.")
elif chosen_alias == "champion" and existing_champion is not None:
    # existing_champion == registered_version here (the gate above already rejected any mismatch), so this
    # is an idempotent no-op rather than an overwrite of a different live champion.
    print(f"@champion already points to version {registered_version} — leaving it in place (idempotent).")
else:
    _uc_client.set_registered_model_alias(REGISTERED_MODEL_NAME, chosen_alias, registered_version)
    print(f"Set @{chosen_alias} -> {REGISTERED_MODEL_NAME} version {registered_version} "
          f"(existing champion before this run: {existing_champion}).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Follow-ons (deliberately OUT of scope here)
# MAGIC This notebook stops at "champion refit on the frozen preprocessing + calibrated + registered to UC +
# MAGIC honestly evaluated". The next steps, each its own notebook/task:
# MAGIC
# MAGIC * **Model Serving** — a batch scorer (Lakeflow Job) or a real-time endpoint on the `@champion` alias.
# MAGIC * **Lakehouse Monitoring** — inference + drift monitoring on the scored output table.
# MAGIC * **Retraining loop** — scheduled re-run of features -> selection -> HPO -> this notebook for new
# MAGIC   as-of dates, with the champion-promotion gate above kept as a human checkpoint.
# MAGIC * **`segment` finding-#1 remediation** — replace the current-state join with an effective-dated /
# MAGIC   SCD2 `user_segment_mapping` joined as-of `feature_ts`, then move `segment` into the PIT `FeatureLookup`.
