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
# MAGIC The training notebook now logs the champion via **`FeatureEngineeringClient.log_model(training_set=...)`**
# MAGIC (SECTION 8), so the point-in-time **`FeatureLookup`s are packaged INTO the model**. That makes
# MAGIC `fe.score_batch(...)` the correct — and far simpler — scoring path: it performs the as-of feature lookup
# MAGIC automatically from the packaged lineage and passes the **raw looked-up feature columns** straight into the
# MAGIC pyfunc's `predict`, where the frozen preprocessing runs. So this scorer:
# MAGIC
# MAGIC * builds ONLY a **scoring spine** (`internal_user_id`, `feature_ts`, + the current-state `segment`
# MAGIC   passthrough the model expects) — **no** `create_training_set` / `spark_udf` feature rebuild,
# MAGIC * calls `fe.score_batch(model_uri=<@champion>, df=<spine>)` and lets Feature Engineering do the PIT join,
# MAGIC * reads the model's own multi-column output (`raw_score`, `calibrated_probability`, `decile`).
# MAGIC
# MAGIC The pyfunc carries the **frozen preprocessing** (segment vocab, decile score edges, booster,
# MAGIC calibrator), so we do **no** `fit` / `value_counts` / `get_dummies` / feature retrieval here — the model's
# MAGIC own `predict` runs the frozen transform on the FE-supplied columns, so train/serve feature construction
# MAGIC (retrieval AND encoding) is identical *by construction*.
# MAGIC
# MAGIC ## Scoring contract implemented here
# MAGIC 1. Load the champion by **ALIAS only** (`models:/mlops_data_science.models.first_order_propensity@champion`) —
# MAGIC    never a hard-coded version. Resolve the concrete version + run_id from the alias; **fail loud** if it
# MAGIC    does not resolve.
# MAGIC 2. Build the **scoring spine** by reusing the spine eligibility logic — carded-before-cutoff +
# MAGIC    first-order anti-join — but with **NO label and NO forward performance window** (inference only).
# MAGIC    `spark.sql` only: no `databricks-sql-connector`, no PAT, no egress.
# MAGIC 3. **`fe.score_batch`** does the point-in-time feature lookup as-of `feature_ts` (from the packaged
# MAGIC    `FeatureLookup`s) and scores — no manual feature matrix.
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
# MAGIC Grounded against the official docs + canonical Databricks examples (retrieved 2026-07-23):
# MAGIC
# MAGIC * `FeatureEngineeringClient.score_batch(model_uri, df, result_type='double', env_manager='local',`
# MAGIC   `params=None, ...)` — scores a model logged with `fe.log_model` (our champion now is). The input `df`
# MAGIC   must "(1) contain columns for lookup keys required to join feature data from feature tables, (2) contain
# MAGIC   columns for all source keys required to score the model, (3) not contain a column `prediction`". For a
# MAGIC   **time-series** feature table the `df` "must contain a timestamp column with the same name and DataType
# MAGIC   as the `timestamp_lookup_key`" — here that is `feature_ts`. Feature Engineering **auto-joins** the
# MAGIC   feature values (do NOT pre-join them). The returned DataFrame is: all columns of `df` + all looked-up
# MAGIC   feature values + **a column `prediction`** holding the model output. `result_type` accepts a Spark type
# MAGIC   / **DDL struct string** for a multi-column pyfunc output (verified: taxi example uses `fe.score_batch`
# MAGIC   for a pyfunc; `surrogate_modeling` uses `result_type=ArrayType(DoubleType())` for multi-value output),
# MAGIC   so `prediction` becomes a struct we unpack. `env_manager="local"` scores in the current cluster
# MAGIC   environment (no env rebuild — required here since the box has no PyPI egress).
# MAGIC   Ref: `https://api-docs.databricks.com/python/feature-engineering/latest/feature_engineering.client.html`
# MAGIC   and the Databricks "model inference with Feature Engineering" / UC taxi Feature Engineering example.
# MAGIC * **Passthrough (non-lookup) columns**: columns present in `df` that are not in any feature table are
# MAGIC   passed to `predict` unchanged (UC taxi example: raw `trip_distance`). This is how current-state
# MAGIC   `segment` (finding #1 — no as-of column) reaches the model, exactly as it did at training time.
# MAGIC * `MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(name, alias)` — resolves the
# MAGIC   `@champion` alias to a concrete `ModelVersion` (`.version`, `.run_id`). Raises `RESOURCE_DOES_NOT_EXIST`
# MAGIC   when the alias is absent — we surface that as a fail-loud error.
# MAGIC   Ref: `https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html`
# MAGIC * `include_segment` / the store-feature subset are read back from the champion's **training run params**
# MAGIC   (logged in SECTION 8 of the training notebook) rather than re-derived, so the scorer's spine matches
# MAGIC   what the model was trained + packaged with — no hardcoded feature list.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Dependencies
# MAGIC MLflow 3.0, XGBoost, scikit-learn and `databricks-feature-engineering` all ship with Databricks Runtime
# MAGIC 17.0 ML+. The install below pins recent versions so `fe.score_batch` (PIT join + distributed scoring) and
# MAGIC the pyfunc's own imports (XGBoost booster + sklearn calibrator, invoked inside `predict`) are all available
# MAGIC on every node. Notebook-scoped `%pip` propagates to executors, which `score_batch(env_manager="local")`
# MAGIC needs. Safe to skip on a current ML Runtime.

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

# score_batch environment manager. "local" uses the current cluster env (fast, no PyPI egress needed — this
# box blackholes PyPI). Switch to "virtualenv" ONLY on a cluster with egress if you want the model's pinned env.
SCORE_BATCH_ENV_MANAGER = "local"

# The pyfunc's fixed output columns, in order. score_batch returns them inside its `prediction` struct; this
# DDL is the result_type we pass, and the field order/names are asserted below so a model-contract change
# fails loud, not silent. `decile` is bigint to match the pyfunc's int64 output.
EXPECTED_MODEL_OUTPUT_COLS = ["raw_score", "calibrated_probability", "decile"]
SCORE_RESULT_DDL = "raw_score double, calibrated_probability double, decile bigint"

# ---------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1a. Widgets — scoring date + model name + output target (fail-fast on required config)
# MAGIC * `scoring_date` — the as-of cutoff. **Blank = latest as-of**, derived with the SAME orders probe the
# MAGIC   sibling notebooks use for their default, but **without** subtracting the performance window (there is no
# MAGIC   forward label to reserve at scoring time — see the resolver's docstring).
# MAGIC * `registered_model_name` — the three-level UC model name. The alias is **not** a widget: it is the
# MAGIC   code-level constant `MODEL_ALIAS = "champion"` (see the cell below), so the champion is loaded by
# MAGIC   **alias only** (never a version) and a scheduled job cannot be silently repointed. Challenger / shadow
# MAGIC   scoring is out of scope and would require a code edit.
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
# MAGIC ## SECTION 1 — Resolve the champion (ALIAS ONLY)
# MAGIC The alias is resolved to a concrete version + run_id for the provenance columns. **Fail loud** if the
# MAGIC `@champion` alias does not resolve — we never fall back to a version. We do **not** load the pyfunc or
# MAGIC build a feature matrix here: `fe.score_batch` (Section 4) loads the model by URI and auto-joins the
# MAGIC packaged features. `include_segment` is read back from the champion's **training run params** (logged in
# MAGIC the training notebook's SECTION 8) so we know whether to attach the current-state `segment` passthrough
# MAGIC the model expects — deterministic, not re-derived from a feature list.

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

# Whether the model uses current-state `segment` is read back from the champion's TRAINING run params
# (logged as `include_segment` in the training notebook's SECTION 8). We do NOT load the pyfunc or inspect a
# feature list here: fe.score_batch (Section 4) auto-joins the packaged store features, and segment is the
# only passthrough column the scorer must supply on the spine. Default False (segment not attached) if the
# param is absent — the model simply would not have been packaged with a segment input in that case.
def _resolve_include_segment(client, run_id):
    """Read `include_segment` from the champion's training run params. Returns bool; defaults False if absent."""
    if not run_id:
        print("WARNING: champion has no run_id; cannot read include_segment from training params -> False.")
        return False
    try:
        raw = client.get_run(run_id).data.params.get("include_segment")
    except Exception as exc:  # noqa: BLE001 — a missing/unreadable run should not silently attach segment
        print(f"WARNING: could not read training run {run_id} params ({exc}); include_segment -> False.")
        return False
    return str(raw).strip().lower() == "true"


INCLUDE_SEGMENT = _resolve_include_segment(MlflowClient(), CHAMPION_RUN_ID)  # tracking client reads run params
print(f"  include segment (current-state passthrough): {INCLUDE_SEGMENT}")

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
population_df = population_df.cache()  # count + score_batch spine both consume it
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
# MAGIC ## SECTION 3 — Scoring spine (keys + feature_ts + segment passthrough) — NO manual feature build
# MAGIC Because the champion is packaged with its `FeatureLookup`s, we do **not** rebuild the feature matrix.
# MAGIC The spine that `fe.score_batch` consumes needs only:
# MAGIC
# MAGIC * `internal_user_id` — the lookup key,
# MAGIC * `feature_ts` — the timeseries key (same name + type as the packaged `timestamp_lookup_key`), which
# MAGIC   drives the as-of join to the latest `feature_ts <= cutoff`,
# MAGIC * `segment` — attached **current-state** IF the model was trained with it (a **passthrough** column, not
# MAGIC   a lookup: finding #1, no as-of table), exactly as training attached it to its training-set spine.
# MAGIC
# MAGIC The store features are looked up automatically by `score_batch` (Section 4) — no `create_training_set`,
# MAGIC no `spark_udf`, no `feature_names` list here. The v7 engagement gate needs the looked-up recency columns,
# MAGIC which only exist AFTER the auto-join, so it is applied to the `score_batch` OUTPUT in Section 4 (scoring
# MAGIC is per-row, so gating after scoring drops exactly the same users as gating before — identical result).

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Build the score_batch spine: keys + feature_ts, plus the current-state `segment` passthrough IF the model
# uses it. NO FeatureLookup / create_training_set — score_batch auto-joins the packaged store features.
score_spine = population_df
if INCLUDE_SEGMENT:
    # Reproduces v7's `coalesce(usm.segment_name, 'No Segment') AS segment`, joined current-state — the same
    # passthrough column the training-set spine carried, so the model receives segment exactly as at training.
    segment_df = spark.sql(f"SELECT internal_user_id, segment_name FROM {USER_SEGMENT_MAPPING}")
    score_spine = (
        score_spine.join(segment_df, on=ENTITY_KEY, how="left")
        .withColumnRenamed("segment_name", "segment")
        .fillna({"segment": SEGMENT_DEFAULT})
    )
print(f"Score_batch spine columns: {score_spine.columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 4 — Score with `fe.score_batch` (auto PIT feature lookup) + v7 engagement gate
# MAGIC `fe.score_batch` loads the champion by URI, performs the **point-in-time feature lookup** from the
# MAGIC packaged `FeatureLookup`s (as-of `feature_ts`), and passes the raw looked-up columns into the pyfunc's
# MAGIC `predict` — where the frozen preprocessing runs. It returns all spine columns + the looked-up feature
# MAGIC values + a `prediction` column (here a struct, from our `result_type` DDL). We do **no** preprocessing.
# MAGIC
# MAGIC The v7 engagement gate is applied to the returned looked-up recency columns (guarded on presence exactly
# MAGIC like training's SECTION 2). NULL recency == "never", so `IS NOT NULL` reproduces v7's `*_search IS NOT
# MAGIC NULL`, and `days_since_last_app_open < 90` reproduces v7's `coalesce(...,9999) < 90`. Because scoring is
# MAGIC per-row and independent, gating the scored output drops exactly the users a pre-scoring gate would.
# MAGIC
# MAGIC **Decile rule (documented, reproducible):** we use the model's **own `decile` output**, computed from the
# MAGIC **frozen VAL score edges captured at training** (`decile_score_edges` inside the model). These FIXED edges
# MAGIC make a user's decile population-anchored and identical regardless of this batch's size — the correct
# MAGIC choice for an operational score compared across runs. `model_score` is the calibrated probability;
# MAGIC `raw_score` is the monotonic ranking score the decile derives from, kept for lineage.

# COMMAND ----------

from pyspark.sql import functions as F

# score_batch: PIT-joins the packaged store features as-of feature_ts and scores. Multi-output pyfunc ->
# DDL struct result_type so `prediction` is a struct we unpack. env_manager="local" uses the cluster env
# (no PyPI egress). df carries ONLY the lookup key + timeseries key + segment passthrough.
scored_raw = fe.score_batch(
    model_uri=MODEL_URI,                 # ALIAS ONLY — scores with the promoted champion
    df=score_spine,
    result_type=SCORE_RESULT_DDL,
    env_manager=SCORE_BATCH_ENV_MANAGER,
)

# v7 engagement gate on the looked-up recency columns score_batch just joined in (guarded on presence).
_gate_terms = []
if "days_since_last_app_open" in scored_raw.columns:
    _gate_terms.append("days_since_last_app_open < 90")
for _rc in [
    "days_since_last_flight_search", "days_since_last_hotel_search",
    "days_since_last_bus_search", "days_since_last_train_search",
]:
    if _rc in scored_raw.columns:
        _gate_terms.append(f"{_rc} IS NOT NULL")
if _gate_terms:
    scored_raw = scored_raw.filter(" OR ".join(_gate_terms))
    print(f"Applied v7 engagement gate on the model's recency columns: {_gate_terms}")
else:
    print("No recency/app-open columns in the model's feature subset -> engagement gate skipped (documented "
          "divergence, matching training when those columns are not selected).")

# Unpack the `prediction` struct into the fixed output contract (raw_score, calibrated_probability, decile).
# Assert the struct fields match EXPECTED_MODEL_OUTPUT_COLS so a silent model-shape change fails loud here.
_pred_field = scored_raw.schema["prediction"]
_pred_fields = [f.name for f in _pred_field.dataType.fields]
if _pred_fields != EXPECTED_MODEL_OUTPUT_COLS:
    raise ValueError(
        f"score_batch prediction fields {_pred_fields} != expected {EXPECTED_MODEL_OUTPUT_COLS}. The scorer's "
        f"decile / score extraction assumes that contract. Refusing to write against an unknown output shape. "
        f"FAILING FAST."
    )
scored = scored_raw.select(
    F.col(ENTITY_KEY),
    F.col("prediction.raw_score").alias("raw_score"),
    F.col("prediction.calibrated_probability").alias("calibrated_probability"),
    F.col("prediction.decile").alias("decile"),
)
scored = scored.cache()
_n_scoring = scored.count()
print(f"Scored population: {_n_scoring:,} rows. Output columns: {scored.columns}")
if _n_scoring == 0:
    raise ValueError("Scored set is empty after the engagement gate. Refusing to write. FAILING FAST.")

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
