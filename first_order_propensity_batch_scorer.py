# Databricks notebook source
# MAGIC %md
# MAGIC # Scapia — First-Order-Propensity — Phase-4 BATCH SCORER (operationalize the champion)
# MAGIC
# MAGIC Fourth notebook in the chain. The first three built + registered the model:
# MAGIC
# MAGIC 1. `first_order_propensity_user_features.py` — the shared per-user **feature store** table.
# MAGIC 2. `first_order_propensity_spine_and_selection.py` — the spine + **feature selection**.
# MAGIC 3. `first_order_propensity_model_training.py` — refit + calibrate + **register the champion to UC**.
# MAGIC 4. **THIS notebook** — score the *currently-eligible* first-order population with the promoted
# MAGIC    `@champion`, produce a **score + decile per user**, and write an idempotent, governed **Delta**
# MAGIC    output table for downstream activation.
# MAGIC
# MAGIC ## The single most important correctness decision — matches how the model was LOGGED
# MAGIC The training notebook logged the champion as a **plain `mlflow.pyfunc` model**
# MAGIC (`mlflow.pyfunc.log_model(...)`, SECTION 8) — **not** with `FeatureEngineeringClient.log_model(...)`.
# MAGIC It does *not* carry feature metadata / a feature_spec, so `fe.score_batch(...)` is **not** applicable
# MAGIC (that API requires a model logged via `fe.log_model`). Instead, training built its matrix by an explicit
# MAGIC point-in-time `fe.create_training_set(...)` join and then passed the resulting **raw feature columns**
# MAGIC into the pyfunc. So this scorer **mirrors that exact path**:
# MAGIC
# MAGIC * rebuild the scoring feature matrix with the SAME `fe.create_training_set(...)` as-of join
# MAGIC   (`label=None`, since inference has no label),
# MAGIC * feed the model the **same raw input columns** its signature declares,
# MAGIC * call `model.predict` (distributed via `mlflow.pyfunc.spark_udf`).
# MAGIC
# MAGIC The pyfunc carries the **frozen preprocessing** (segment vocab, decile score edges, booster,
# MAGIC calibrator). We therefore do **no** `fit` / `value_counts` / `get_dummies` here — the model's own
# MAGIC `predict` runs the frozen transform, so train/serve feature construction is identical *by construction*.
# MAGIC
# MAGIC ## Scoring contract implemented here
# MAGIC 1. Load the champion by **ALIAS only** (`models:/mlops_data_science.models.first_order_propensity@champion`) —
# MAGIC    never a hard-coded version. Resolve the concrete version + run_id from the alias; **fail loud** if it
# MAGIC    does not resolve.
# MAGIC 2. Build the **scoring population** by reusing the spine eligibility logic — carded-before-cutoff +
# MAGIC    first-order anti-join — but with **NO label and NO forward performance window** (inference only).
# MAGIC    `spark.sql` only: no `databricks-sql-connector`, no PAT, no egress.
# MAGIC 3. **Point-in-time** feature construction as-of the scoring date (identical to training's `create_training_set`).
# MAGIC 4. Emit **score + decile** per user. The decile uses the **frozen VAL score edges carried inside the
# MAGIC    model** (population-anchored, computed at training) — see Section 4 — so deciles are comparable across
# MAGIC    scoring runs and independent of this batch's size.
# MAGIC 5. Write deciles to a **NEW governed UC Delta table** (default `mlops_data_science.default.first_order_propensity_scores`),
# MAGIC    with provenance columns, **idempotent per scoring date**.
# MAGIC 6. Log a lightweight MLflow scoring run for lineage (best-effort / guarded).
# MAGIC
# MAGIC ## Out of scope (future phases — documented at the end, deliberately NOT built here)
# MAGIC Real-time Model Serving endpoint, Lakehouse Monitoring (inference/drift), retraining loop.

# COMMAND ----------

# MAGIC %md
# MAGIC ## API grounding — what was verified against current docs
# MAGIC Grounded against the official docs (retrieved 2026-07-21):
# MAGIC
# MAGIC * `FeatureEngineeringClient.create_training_set(df, feature_lookups, label=None, exclude_columns, ...)`
# MAGIC   — `label` **may be `None`** ("To create a training set without a label field, i.e. for unsupervised
# MAGIC   training set, specify `label = None`."). The returned `TrainingSet` has `.load_df()`.
# MAGIC   Ref: `https://api-docs.databricks.com/python/feature-engineering/latest/feature_engineering.client.html`
# MAGIC * `FeatureEngineeringClient.score_batch(...)` — **requires** a model logged with `fe.log_model` (feature
# MAGIC   metadata). Our champion is a plain pyfunc, so this scorer does **not** use `score_batch`. Same ref.
# MAGIC * `mlflow.pyfunc.spark_udf(spark, model_uri, result_type=None, env_manager=None, ...)` — `result_type`
# MAGIC   accepts a **DDL struct string** for multi-column model output; `env_manager="local"` scores in the
# MAGIC   current cluster environment (no env rebuild — required here since the box has no PyPI egress). When the
# MAGIC   model has a signature, the eval DataFrame's column names must match the signature.
# MAGIC   Ref: `https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html`
# MAGIC * `MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(name, alias)` — resolves the
# MAGIC   `@champion` alias to a concrete `ModelVersion` (`.version`, `.run_id`). Raises `RESOURCE_DOES_NOT_EXIST`
# MAGIC   when the alias is absent — we surface that as a fail-loud error.
# MAGIC   Ref: `https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html`
# MAGIC * `PyFuncModel.metadata.signature.inputs` (a `Schema`, with `.input_names()`) — the serve-time contract
# MAGIC   for the exact raw input columns. Same pyfunc ref.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Dependencies
# MAGIC MLflow 3.0, XGBoost, scikit-learn and `databricks-feature-engineering` all ship with Databricks Runtime
# MAGIC 17.0 ML+. The install below pins recent versions so `create_training_set` (PIT join), `spark_udf` and the
# MAGIC pyfunc's own imports (XGBoost booster + sklearn calibrator, invoked inside `predict`) are all available on
# MAGIC every node. Notebook-scoped `%pip` propagates to executors, which `spark_udf(env_manager="local")` needs.
# MAGIC Safe to skip on a current ML Runtime.

# COMMAND ----------

# MAGIC %pip install -U "mlflow>=3.0" databricks-feature-engineering xgboost scikit-learn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. CONFIG
# MAGIC Source-table names, keys, IST offset and the qualifying status/category filters are kept **verbatim**
# MAGIC from the spine / training notebooks so the eligibility population reproduces exactly (minus the label).

# COMMAND ----------

# ---------------------------------------------------------------------------
# CONFIG — edit here (verbatim from the spine / training notebooks)
# ---------------------------------------------------------------------------

# Feature table produced by first_order_propensity_user_features.py (verbatim).
CATALOG = "mlops_data_science"
SCHEMA = "features"
TABLE = "first_order_propensity_user_features"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"

# Entity key + point-in-time key — must match the feature table's PK / timeseries column exactly.
ENTITY_KEY = "internal_user_id"
FEATURE_TS = "feature_ts"

# --- Source tables for the scoring population (read-only; eligibility ONLY) --
# NOTE: unlike training, ORDERS_TABLE is used ONLY for the first-order anti-join (and the default-date probe).
# There is NO label / performance-window read here — inference has no forward outcome.
ORDERS_TABLE = "rds_main.scapiadb.orders"                    # first-order anti-join only
ONBOARDED_USERS_FACT = "simple.crud.onboarded_users_fact"    # carded universe + onboarding date
# `segment` is NOT in the feature table (audit finding #1 — no as-of column). It is joined CURRENT-STATE
# below IF and only if the champion's signature declares it, exactly as training's SECTION 2 does.
USER_SEGMENT_MAPPING = "simple.crud.user_segment_mapping"
SEGMENT_DEFAULT = "No Segment"  # v7's coalesce(segment_name, 'No Segment')

# IST offset. created_at in the RDS-sourced tables is UTC; the reference query anchors the cutoff in IST and
# subtracts this offset before comparing against the UTC values.
IST_OFFSET_MINUTES = 330

# Business-confirmed status filter — kept EXACTLY as the spine/training notebooks have it.
QUALIFYING_STATUSES = ["COMPLETE", "CANCELLED"]
QUALIFYING_PRODUCT_CATEGORIES = [
    "FLIGHT", "BUS", "TRAIN", "HOTEL_STAY",
    "ECOMMERCE", "EXPERIENCE", "VISA", "HOLIDAY",
]

# The one supported current-state (non-store) feature, matching the sibling notebooks.
SUPPORTED_NON_STORE_FEATURES = ["segment"]

# spark_udf environment manager. "local" uses the current cluster env (fast, no PyPI egress needed — this box
# blackholes PyPI). Switch to "virtualenv" ONLY on a cluster with egress if you want the model's pinned env.
SPARK_UDF_ENV_MANAGER = "local"

# The pyfunc's fixed output columns (asserted after load so a model-contract change fails loud, not silent).
EXPECTED_MODEL_OUTPUT_COLS = ["raw_score", "calibrated_probability", "decile"]

# ---------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1a. Widgets — scoring date + model alias + output target (fail-fast on required config)
# MAGIC * `scoring_date` — the as-of cutoff. **Blank = latest as-of**, derived with the SAME orders probe the
# MAGIC   sibling notebooks use for their default, but **without** subtracting the performance window (there is no
# MAGIC   forward label to reserve at scoring time — see the resolver's docstring).
# MAGIC * `registered_model_name` + `model_alias` — the champion is loaded by **alias only** (never a version).
# MAGIC * `output_catalog` / `output_schema` / `output_table` — the NEW governed scores table. The default is a
# MAGIC   **new** table, deliberately NOT the existing `mlops_data_science.default.first_order_propen_output`
# MAGIC   (we do not silently change the schema of whatever a current job writes).

# COMMAND ----------

import json

import mlflow
from mlflow.tracking import MlflowClient

dbutils.widgets.text(
    "scoring_date", "",
    "As-of / cutoff (YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS', IST). Blank = latest as-of (no forward window).",
)
dbutils.widgets.text(
    "registered_model_name", "mlops_data_science.models.first_order_propensity",
    "Three-level UC model name (catalog.schema.model).",
)
dbutils.widgets.text("output_catalog", "mlops_data_science", "Output catalog for the scores table.")
dbutils.widgets.text("output_schema", "default", "Output schema for the scores table.")
dbutils.widgets.text(
    "output_table", "first_order_propensity_scores",
    "Output table name. NEW governed table (NOT the existing first_order_propen_output).",
)

REGISTERED_MODEL_NAME = dbutils.widgets.get("registered_model_name").strip()
OUTPUT_CATALOG = dbutils.widgets.get("output_catalog").strip()
OUTPUT_SCHEMA = dbutils.widgets.get("output_schema").strip()
OUTPUT_TABLE_NAME = dbutils.widgets.get("output_table").strip()

if REGISTERED_MODEL_NAME.count(".") != 2:
    raise ValueError(
        f"registered_model_name must be a three-level UC name 'catalog.schema.model'; got "
        f"'{REGISTERED_MODEL_NAME}'. FAILING FAST."
    )
if not (OUTPUT_CATALOG and OUTPUT_SCHEMA and OUTPUT_TABLE_NAME):
    raise ValueError("output_catalog / output_schema / output_table must all be non-empty. FAILING FAST.")

# STRICTLY champion-pinned — a code-level CONSTANT, deliberately NOT a runtime widget, so a scheduled job can
# never be silently repointed at another alias and score with a non-promoted model. Scoring runs the PROMOTED
# @champion ONLY; challenger / shadow scoring is intentionally OUT OF SCOPE. This stays ALIAS-based (not a
# hard-coded version): it always follows whichever version is currently promoted to @champion. Changing this
# requires an explicit code edit, not a widget change.
MODEL_ALIAS = "champion"

OUTPUT_TABLE = f"{OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.{OUTPUT_TABLE_NAME}"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"  # ALIAS ONLY — never a hard-coded version

print(f"UC model name : {REGISTERED_MODEL_NAME}")
print(f"Model URI     : {MODEL_URI}")
print(f"Output table  : {OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 1 — Resolve + load the champion (ALIAS ONLY)
# MAGIC The alias is resolved to a concrete version + run_id for the provenance columns, then the pyfunc is
# MAGIC loaded. **Fail loud** if the `@champion` alias does not resolve — we never fall back to a version. The
# MAGIC model's input **signature** gives the exact raw feature columns to build; its output columns are asserted
# MAGIC against the known contract so a silent model-shape change is caught here rather than corrupting the write.

# COMMAND ----------

from mlflow.exceptions import RestException

mlflow.set_registry_uri("databricks-uc")
_uc_client = MlflowClient(registry_uri="databricks-uc")

_ALIAS_NOT_FOUND_CODE = "RESOURCE_DOES_NOT_EXIST"


def resolve_champion(client, model_name, alias):
    """Resolve `model_name@alias` to (version, run_id). FAIL LOUD if the alias is not set — never guess a version."""
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
    except RestException as exc:
        if getattr(exc, "error_code", None) == _ALIAS_NOT_FOUND_CODE:
            raise RuntimeError(
                f"@{alias} alias is not set on {model_name}. The batch scorer runs the PROMOTED champion only; "
                f"promote a version first (see the training notebook's SECTION 9 promotion command). FAILING FAST."
            ) from exc
        raise  # auth / throttling / any other registry error -> fail LOUD, never silently proceed
    return str(mv.version), (getattr(mv, "run_id", None) or "")


CHAMPION_VERSION, CHAMPION_RUN_ID = resolve_champion(_uc_client, REGISTERED_MODEL_NAME, MODEL_ALIAS)
print(f"Resolved @{MODEL_ALIAS} -> version {CHAMPION_VERSION} (run_id {CHAMPION_RUN_ID or 'unknown'})")

# Load the pyfunc BY ALIAS (not by the resolved version) so scoring always tracks the promoted champion.
loaded_model = mlflow.pyfunc.load_model(MODEL_URI)

# The raw input columns the model expects == its input signature. This is the serve-time contract; the
# store-resident subset + optional current-state `segment` are derived from it (no hardcoded feature list).
_sig = getattr(loaded_model.metadata, "signature", None)
if _sig is None or _sig.inputs is None:
    raise ValueError(
        "Champion model has no input signature — cannot determine the raw scoring feature columns. A signature "
        "is mandatory for UC registration, so this indicates a malformed model. FAILING FAST."
    )
INPUT_COLS = list(_sig.inputs.input_names())
if not INPUT_COLS:
    raise ValueError("Champion input signature has no named columns; cannot map scoring features. FAILING FAST.")

STORE_FEATURES = [c for c in INPUT_COLS if c not in SUPPORTED_NON_STORE_FEATURES]  # PIT-lookupable subset
INCLUDE_SEGMENT = "segment" in INPUT_COLS  # DERIVED from the signature, not hardcoded

# Assert the output contract (raw_score, calibrated_probability, decile) so a shape change fails loud here.
_out = getattr(loaded_model.metadata, "signature", None)
_out_names = list(_out.outputs.input_names()) if (_out is not None and _out.outputs is not None) else []
if _out_names and _out_names != EXPECTED_MODEL_OUTPUT_COLS:
    raise ValueError(
        f"Champion output columns {_out_names} != expected {EXPECTED_MODEL_OUTPUT_COLS}. The scorer's decile / "
        f"score extraction assumes that contract. Refusing to write against an unknown output shape. FAILING FAST."
    )

print(f"Model input columns ({len(INPUT_COLS)}): {INPUT_COLS}")
print(f"  store-resident (PIT lookup): {STORE_FEATURES}")
print(f"  include segment (current-state): {INCLUDE_SEGMENT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 2 — Scoring population (eligibility ONLY — NO label, NO performance window)
# MAGIC Reuses the spine eligibility from `first_order_propensity_spine_and_selection.py`, adapted for inference:
# MAGIC
# MAGIC * **carded before cutoff** → `WHERE ob.onboarding_completion_date < cutoff` (same eligibility window),
# MAGIC * **first-order anti-join** → drop anyone who already had a qualifying order as of the cutoff (this is
# MAGIC   *first*-order propensity).
# MAGIC
# MAGIC What is deliberately **removed** vs. the training spine: the `performance_users` label CTE, the `output`
# MAGIC column, and the forward `performance_end` window. At scoring time there is no observed outcome, so the
# MAGIC training-time leakage concern around the forward window does not apply here — but we keep the population
# MAGIC definition otherwise identical so we score the same kind of user the model was trained on.
# MAGIC
# MAGIC The `segment` current-state caveat (audit finding #1) is carried unchanged from the sibling notebooks:
# MAGIC `simple.crud.user_segment_mapping` has no effective-date column, so `segment` reflects the user's segment
# MAGIC *now*. At scoring time we WANT the current segment, so this is benign here; the caveat is noted only for
# MAGIC consistency with the training feature schema the model expects.

# COMMAND ----------

from datetime import datetime, timedelta


def _sql_in_list(values):
    """Render a Python list as a SQL IN-list of single-quoted literals."""
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def _parse_ts(ts_str: str) -> datetime:
    """Parse 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' into a datetime."""
    return datetime.fromisoformat(ts_str.strip())


def resolve_scoring_date(widget_value: str) -> str:
    """Resolve the scoring cutoff. Blank = latest as-of.

    The sibling notebooks default to (latest qualifying-order time in IST) MINUS PERFORMANCE_DAYS, because
    TRAINING must reserve a forward window in which to observe the label. Scoring observes no label, so we
    reuse the SAME orders probe (the "how the other notebooks derive the default date" mechanism) but do NOT
    subtract the performance window: we score as-of the latest available signal. Always pass an explicit
    `scoring_date` for reproducible scheduled runs; the auto default drifts as new orders arrive.

    Feature availability: the PIT join (Section 3) picks the latest `feature_ts <= cutoff` in the feature
    table, so the feature-store job must have materialized features for (or before) this scoring date.
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
    if probe is None or probe["max_date"] is None:
        raise ValueError(
            "Could not derive a default scoring_date (no qualifying orders found). Set the scoring_date widget "
            "explicitly. FAILING FAST."
        )
    return probe["max_date"].strftime("%Y-%m-%d %H:%M:%S")


def build_scoring_population_sql(cutoff: str, ist: int) -> str:
    """ELIGIBILITY ONLY. Emits (internal_user_id, feature_ts). NO label, NO performance window."""
    statuses = _sql_in_list(QUALIFYING_STATUSES)
    categories = _sql_in_list(QUALIFYING_PRODUCT_CATEGORIES)
    return f"""
WITH d AS (
    SELECT timestamp '{cutoff}' AS cutoff
),

-- Carded universe + onboarding (first card-issue) date.
ob AS (
    SELECT user_id AS internal_user_id,
           CAST(MIN(card_issue_time) AS timestamp) AS onboarding_completion_date
    FROM {ONBOARDED_USERS_FACT}
    WHERE card_issue_time IS NOT NULL
    GROUP BY 1
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
    (SELECT cutoff FROM d) AS feature_ts       -- as-of key = cutoff (drives the PIT feature join)
FROM ob
LEFT JOIN pre_cutoff_orderers pco ON pco.internal_user_id = ob.internal_user_id
WHERE ob.onboarding_completion_date < (SELECT cutoff FROM d)  -- eligibility window: carded before cutoff
  AND pco.internal_user_id IS NULL                            -- first-order anti-join: never ordered as of cutoff
"""


SCORING_CUTOFF_TS = resolve_scoring_date(dbutils.widgets.get("scoring_date"))
SCORING_DATE_STR = _parse_ts(SCORING_CUTOFF_TS).strftime("%Y-%m-%d")  # DATE partition / idempotency key

population_df = spark.sql(build_scoring_population_sql(SCORING_CUTOFF_TS, IST_OFFSET_MINUTES))
population_df = population_df.cache()  # count + create_training_set both consume it
_n_population = population_df.count()
print(f"scoring cutoff / feature_ts : {SCORING_CUTOFF_TS}")
print(f"scoring_date (partition key): {SCORING_DATE_STR}")
print(f"Eligible scoring population : {_n_population:,} users (carded before cutoff, no prior first order)")
if _n_population == 0:
    raise ValueError(
        f"Scoring population is empty for cutoff {SCORING_CUTOFF_TS}. Check the cutoff and that upstream source "
        f"tables are populated. Refusing to write an empty scores partition. FAILING FAST."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 3 — Point-in-time feature matrix (identical to training's `create_training_set`)
# MAGIC `create_training_set` does the as-of join on exactly the champion's **store-resident** feature subset
# MAGIC (derived from the signature in Section 1), with `label=None` (inference). Then `segment` is attached
# MAGIC **current-state** IF the model uses it, and the v7 engagement gate is applied against whichever recency
# MAGIC columns the model actually uses — mirroring training's SECTION 2 so the scored population matches the
# MAGIC trained one. No encoding happens here: the pyfunc carries the frozen preprocessing.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()

# Point-in-time lookup of the champion's store-resident subset. label=None -> inference/scoring set.
scoring_set = fe.create_training_set(
    df=population_df,
    feature_lookups=[
        FeatureLookup(
            table_name=FEATURE_TABLE,
            lookup_key=ENTITY_KEY,
            timestamp_lookup_key=FEATURE_TS,  # as-of join -> latest feature_ts <= cutoff (no future leakage)
            feature_names=STORE_FEATURES,
        )
    ],
    label=None,                       # NO label at scoring time (unsupervised training set)
    exclude_columns=[FEATURE_TS],     # keep the join key out of the feature matrix (as training did)
)

scoring_features_df = scoring_set.load_df()
print(f"PIT scoring columns ({len(scoring_features_df.columns)}): {scoring_features_df.columns}")

# COMMAND ----------

# --- Attach `segment` CURRENT-STATE (only if the champion uses it) + v7 engagement gate --------------
scoring_df = scoring_features_df

if INCLUDE_SEGMENT:
    # Reproduces v7's `coalesce(usm.segment_name, 'No Segment') AS segment`, joined current-state.
    segment_df = spark.sql(f"SELECT internal_user_id, segment_name FROM {USER_SEGMENT_MAPPING}")
    scoring_df = (
        scoring_df.join(segment_df, on=ENTITY_KEY, how="left")
        .withColumnRenamed("segment_name", "segment")
        .fillna({"segment": SEGMENT_DEFAULT})
    )

# v7's engagement gate, applied AGAINST THE FEATURE TABLE columns the model actually uses (guarded on
# presence exactly like training's SECTION 2). NULL recency == "never", so IS NOT NULL reproduces v7's
# *_search IS NOT NULL, and days_since_last_app_open < 90 reproduces v7's coalesce(...,9999) < 90.
_gate_terms = []
if "days_since_last_app_open" in scoring_df.columns:
    _gate_terms.append("days_since_last_app_open < 90")
for _rc in [
    "days_since_last_flight_search", "days_since_last_hotel_search",
    "days_since_last_bus_search", "days_since_last_train_search",
]:
    if _rc in scoring_df.columns:
        _gate_terms.append(f"{_rc} IS NOT NULL")
if _gate_terms:
    scoring_df = scoring_df.filter(" OR ".join(_gate_terms))
    print(f"Applied v7 engagement gate on the model's recency columns: {_gate_terms}")
else:
    print("No recency/app-open columns in the model's feature subset -> engagement gate skipped (documented "
          "divergence, matching training when those columns are not selected).")

scoring_df = scoring_df.cache()
_n_scoring = scoring_df.count()
print(f"Scoring feature set ready: {_n_scoring:,} rows, feeding model columns {INPUT_COLS}")
if _n_scoring == 0:
    raise ValueError("Scoring feature set is empty after the engagement gate. Refusing to write. FAILING FAST.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 4 — Score (distributed) with the champion pyfunc
# MAGIC Scoring runs **distributed** via `mlflow.pyfunc.spark_udf` (the eligible population can be large; this
# MAGIC avoids pulling it to the driver). The UDF is fed the model's raw input columns; numerics are cast to
# MAGIC `double` to match the training-time signature (recency columns legitimately carry NULL). We do **no**
# MAGIC preprocessing here — the pyfunc's frozen transform runs inside `predict`.
# MAGIC
# MAGIC **Decile rule (documented, reproducible):** we use the model's **own `decile` output**, which is computed
# MAGIC from the **frozen VAL score edges captured at training** (`decile_score_edges` inside the model). These
# MAGIC are FIXED edges carried from training, so a user's decile is population-anchored and identical regardless
# MAGIC of how many users are in *this* scoring batch — the correct choice for an operational score meant to be
# MAGIC compared across runs. (We deliberately do NOT recompute a rank-based decile on the scoring batch, which
# MAGIC would drift with batch composition.) `model_score` is the calibrated probability; `raw_score` is the
# MAGIC monotonic ranking score the decile derives from, kept for lineage.

# COMMAND ----------

from pyspark.sql import functions as F

# Cast numerics to double (match the signature) and keep segment as string; select exactly the input columns.
_cast_cols = []
for _c in INPUT_COLS:
    if _c in SUPPORTED_NON_STORE_FEATURES:
        _cast_cols.append(F.col(_c).cast("string").alias(_c))
    else:
        _cast_cols.append(F.col(_c).cast("double").alias(_c))
scoring_casted = scoring_df.select(F.col(ENTITY_KEY), *_cast_cols)

# Multi-output pyfunc -> DDL struct result_type. env_manager="local" uses the cluster env (no PyPI egress).
_RESULT_DDL = "raw_score double, calibrated_probability double, decile bigint"
score_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=MODEL_URI,                 # ALIAS ONLY — scores with the promoted champion
    result_type=_RESULT_DDL,
    env_manager=SPARK_UDF_ENV_MANAGER,
)

# Pass columns in signature order; the model's predict returns raw_score / calibrated_probability / decile.
scored = scoring_casted.withColumn("_pred", score_udf(*[F.col(c) for c in INPUT_COLS]))
scored = scored.select(
    F.col(ENTITY_KEY),
    F.col("_pred.raw_score").alias("raw_score"),
    F.col("_pred.calibrated_probability").alias("calibrated_probability"),
    F.col("_pred.decile").alias("decile"),
)
print("Scoring UDF wired (distributed). Columns:", scored.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 5 — Assemble provenance + IDEMPOTENT Delta write
# MAGIC Every row carries full provenance: `internal_user_id`, `model_score` (calibrated probability),
# MAGIC `raw_score` (ranking score), `decile`, `scoring_date`, `feature_ts`, `scored_at`, `model_version`,
# MAGIC `model_run_id`. The write is **idempotent per scoring date**: the table is partitioned by `scoring_date`
# MAGIC and each run does an `overwrite` with `replaceWhere scoring_date = '<date>'`, so re-running the same date
# MAGIC replaces exactly that partition (no duplicates) while other dates are untouched.

# COMMAND ----------

out_df = (
    scored.select(
        F.col(ENTITY_KEY),
        F.col("calibrated_probability").cast("double").alias("model_score"),
        F.col("raw_score").cast("double").alias("raw_score"),
        F.col("decile").cast("int").alias("decile"),
    )
    .withColumn("scoring_date", F.lit(SCORING_DATE_STR).cast("date"))
    .withColumn("feature_ts", F.lit(SCORING_CUTOFF_TS).cast("timestamp"))
    .withColumn("scored_at", F.current_timestamp())
    .withColumn("model_version", F.lit(str(CHAMPION_VERSION)))
    .withColumn("model_run_id", F.lit(CHAMPION_RUN_ID))
)

# IDEMPOTENT per scoring date. First run creates the (scoring_date-partitioned) table; later runs overwrite
# ONLY the current date's partition via replaceWhere, so a re-run never duplicates rows and never disturbs
# other dates. All rows written carry scoring_date = SCORING_DATE_STR, satisfying the replaceWhere predicate.
if spark.catalog.tableExists(OUTPUT_TABLE):
    (
        out_df.write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", f"scoring_date = '{SCORING_DATE_STR}'")
        .saveAsTable(OUTPUT_TABLE)
    )
    _write_mode = f"overwrite (replaceWhere scoring_date = '{SCORING_DATE_STR}')"
else:
    (
        out_df.write.format("delta")
        .mode("overwrite")
        .partitionBy("scoring_date")
        .saveAsTable(OUTPUT_TABLE)
    )
    _write_mode = "created new partitioned table"

ROW_COUNT = spark.table(OUTPUT_TABLE).filter(F.col("scoring_date") == F.lit(SCORING_DATE_STR)).count()
print(f"Wrote {ROW_COUNT:,} scored users to {OUTPUT_TABLE} [{_write_mode}]")
print(f"  scoring_date={SCORING_DATE_STR}  model_version={CHAMPION_VERSION}  run_id={CHAMPION_RUN_ID or 'unknown'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 6 — Lightweight MLflow scoring run (lineage; best-effort)
# MAGIC A tiny run for lineage — params (`model_version`, `model_run_id`, `scoring_date`, `output_table`) and a
# MAGIC `row_count` metric. No heavy artifacts. **Guarded**: a logging hiccup prints a warning and never fails the
# MAGIC scoring write, which has already committed above.

# COMMAND ----------

try:
    with mlflow.start_run(run_name="fop_batch_scoring") as _run:
        mlflow.set_tags({
            "project": "first_order_propensity",
            "stage": "batch_scoring",
            "feature_table": FEATURE_TABLE,
            "model_name": REGISTERED_MODEL_NAME,
            "model_alias": MODEL_ALIAS,
            "output_table": OUTPUT_TABLE,
        })
        mlflow.log_params({
            "model_version": CHAMPION_VERSION,
            "model_run_id": CHAMPION_RUN_ID,
            "scoring_date": SCORING_DATE_STR,
            "scoring_cutoff_ts": SCORING_CUTOFF_TS,
            "include_segment": INCLUDE_SEGMENT,
            "n_input_features": len(INPUT_COLS),
        })
        mlflow.log_metrics({
            "eligible_population": float(_n_population),
            "scored_population": float(_n_scoring),
            "row_count": float(ROW_COUNT),
        })
        print(f"Logged scoring lineage run {_run.info.run_id}")
except Exception as _exc:  # noqa: BLE001 — lineage is best-effort; the scores are already written
    print(f"WARNING: MLflow scoring lineage run did not complete ({_exc}); the scores write above is unaffected.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 7 — Scheduling this as a recurring, champion-pinned Lakeflow Job
# MAGIC This is a **standalone notebook**, not a Databricks Asset Bundle. Schedule it as a single-task Lakeflow
# MAGIC Job that runs the notebook on a cadence (e.g. daily). It is **champion-pinned automatically**: it loads
# MAGIC `models:/<name>@champion`, so promoting a new champion (via the training notebook's human-gated
# MAGIC promotion command) changes what the *next* scheduled run scores — no job edit required.
# MAGIC
# MAGIC Leave the `scoring_date` job parameter blank to score the latest as-of each run, or set it explicitly for
# MAGIC a backfill. **Upstream dependency:** the feature-store job (`first_order_propensity_user_features.py`)
# MAGIC must have materialized features for (or before) the scoring date; chain it as an upstream task or a
# MAGIC separate earlier schedule.
# MAGIC
# MAGIC ```python
# MAGIC # Create the recurring, champion-pinned scoring job with the Databricks SDK (run once, from any client).
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks.sdk.service import jobs
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC created = w.jobs.create(
# MAGIC     name="fop_batch_scorer_daily",
# MAGIC     tasks=[
# MAGIC         jobs.Task(
# MAGIC             task_key="score_first_order_propensity",
# MAGIC             notebook_task=jobs.NotebookTask(
# MAGIC                 notebook_path="/Repos/scapia-ml/feature_store/first_order_propensity_batch_scorer",
# MAGIC                 base_parameters={
# MAGIC                     # NOTE: there is no model-alias parameter — the scorer is champion-pinned in code
# MAGIC                     # (MODEL_ALIAS="champion"). Promotion is the only way to change what is scored.
# MAGIC                     "scoring_date": "",  # blank = latest as-of; set a date for a backfill
# MAGIC                     "registered_model_name": "mlops_data_science.models.first_order_propensity",
# MAGIC                     "output_catalog": "mlops_data_science",
# MAGIC                     "output_schema": "default",
# MAGIC                     "output_table": "first_order_propensity_scores",
# MAGIC                 },
# MAGIC             ),
# MAGIC         )
# MAGIC     ],
# MAGIC     schedule=jobs.CronSchedule(
# MAGIC         quartz_cron_expression="0 0 6 * * ?",  # 06:00 daily
# MAGIC         timezone_id="Asia/Kolkata",
# MAGIC     ),
# MAGIC )
# MAGIC print(created.job_id)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Out of scope (future phases — deliberately NOT built here)
# MAGIC This notebook stops at "score the eligible population with the champion + write an idempotent, governed
# MAGIC scores table". The next steps, each its own deliverable:
# MAGIC
# MAGIC * **Real-time Model Serving** — a `@champion`-pinned serving endpoint for on-demand scoring (this notebook
# MAGIC   is the batch counterpart).
# MAGIC * **Lakehouse Monitoring** — an inference/drift monitor on this scores table (and on realized outcomes once
# MAGIC   the label window closes) to watch score distribution + calibration drift over time.
# MAGIC * **Retraining loop** — a scheduled re-run of features -> selection -> HPO -> training for new as-of dates,
# MAGIC   with the champion-promotion gate kept as a human checkpoint.
# MAGIC * **`segment` finding-#1 remediation** — an effective-dated / SCD2 `user_segment_mapping` joined as-of
# MAGIC   `feature_ts`, moving `segment` into the point-in-time `FeatureLookup` (removes the current-state caveat
# MAGIC   from both training and scoring).
