# Databricks notebook source
# MAGIC %md
# MAGIC # Scapia — First-Order-Propensity — Phase-5 LAKEHOUSE MONITORING + RETRAINING TRIGGER
# MAGIC
# MAGIC Fifth (and final wired) notebook in the chain. The first four built, registered and operationalized
# MAGIC the model:
# MAGIC
# MAGIC 1. `01_first_order_propensity_user_features.py` — the shared per-user **feature store** table.
# MAGIC 2. `02_first_order_propensity_spine_and_selection.py` — the spine + **feature selection**.
# MAGIC 3. `03_first_order_propensity_hpo.py` — distributed **HPO** (`fop_hpo_champion` -> `best_params`).
# MAGIC 4. `04_first_order_propensity_model_training.py` — refit + calibrate + **register the champion to UC**
# MAGIC    (`@champion` / `@challenger` alias flow, human-gated promotion).
# MAGIC 5. `05_first_order_propensity_batch_scorer.py` — score the eligible population with `@champion`, write
# MAGIC    the governed **scores Delta table** (`…default.first_order_propensity_scores`).
# MAGIC 6. **THIS notebook** — stand up **Databricks Lakehouse Monitoring** (an **InferenceLog** profile) on the
# MAGIC    scorer's output so we can watch **score/feature drift** continuously and **model-quality decay**
# MAGIC    (ROC-AUC, top-decile lift, F2) as the ground-truth labels arrive; and **document the retraining
# MAGIC    trigger** that feeds back into the HPO -> train/register (@challenger, human-promoted) loop.
# MAGIC
# MAGIC ## The one Scapia-specific reality this notebook is built around: **labels arrive ~90 days late**
# MAGIC The label is *"did this user place their first qualifying order in the 90-day performance window after
# MAGIC scoring?"* (see `04_…_model_training.py`, SECTION 1: a first qualifying order in
# MAGIC `(cutoff, cutoff + PERFORMANCE_DAYS]`, `PERFORMANCE_DAYS = 90`). So **at scoring time the label does not
# MAGIC exist** — the scores table has *no* label column. The outcome is only observable once the 90-day window
# MAGIC closes. This notebook therefore builds a dedicated **monitored inference table** that mirrors the scores
# MAGIC table **plus a `label` column that is `NULL` until the window matures**, and is re-`MERGE`d each run so
# MAGIC newly-matured actuals backfill into the label. The InferenceLog monitor **tolerates the NULL labels** —
# MAGIC drift on predictions/features is computed from day one, and the label-dependent metrics (ROC-AUC, F2,
# MAGIC lift) simply populate later, for the windows whose labels have arrived.
# MAGIC
# MAGIC ## Scope implemented here
# MAGIC 1. **Config + widgets** — scores table (default = the scorer's real output table), monitoring assets
# MAGIC    schema, granularities, slicing columns; fail-fast on blank required values.
# MAGIC 2. **Label backfill (no egress)** — `spark.sql` join of the scores table to realized first orders,
# MAGIC    materializing the monitored inference table with a maturity-aware `label` (NULL until the 90-day
# MAGIC    window has closed).
# MAGIC 3. **Idempotent InferenceLog monitor** — `problem_type = classification`, wired to
# MAGIC    prediction / prediction-proba / label / timestamp / model-id columns, daily + weekly + monthly
# MAGIC    granularities, sliced by decile (and segment if enriched). Update-or-create, never blind re-create.
# MAGIC 4. **Business custom metrics** — top-decile **lift** + **F2** (aggregate -> derived) plus a **drift**
# MAGIC    metric on the lift, so lift-decay is trackable beyond the generic ROC-AUC.
# MAGIC 5. **Retraining-trigger doc + a guarded read-only helper** — threshold logic on ROC-AUC / lift decay,
# MAGIC    feeding the existing HPO -> train/register pipeline. The trigger job itself is documented as a future
# MAGIC    Lakeflow Job (OUT of scope here).
# MAGIC
# MAGIC ## Out of scope (future phases — documented, deliberately NOT built here)
# MAGIC The retraining-trigger **job / orchestration DAB**, **model serving / online inference**, and
# MAGIC **alerting integrations** (email/Slack) beyond a documented stub.

# COMMAND ----------

# MAGIC %md
# MAGIC ## API grounding — what was verified, and the monitoring-API drift you must know about
# MAGIC A hallucinated monitor API is the worst outcome, so the surface below was grounded against the
# MAGIC **installed SDK's own source/docstrings** and the current official docs (retrieved 2026-07-22). Nothing
# MAGIC here is guessed; every place a signature could not be pinned is marked `# TODO(verify-api)`.
# MAGIC
# MAGIC ### The three monitoring surfaces (this is the "moved across versions" gotcha)
# MAGIC * **Legacy** `databricks.lakehouse_monitoring` module (`lm.create_monitor(...)`, `lm.InferenceLog(...)`)
# MAGIC   — the *old* import. **Not used here** and not installed in current runtimes.
# MAGIC * **Current GA** `WorkspaceClient.quality_monitors` (`QualityMonitorsAPI`) — **what this notebook uses.**
# MAGIC   Confirmed present in the installed `databricks-sdk==0.82.0`. Create/update signature verified from the
# MAGIC   SDK directly (see the create cell). This is the stable, documented path.
# MAGIC   Ref: `https://docs.databricks.com/aws/en/lakehouse-monitoring/create-monitor-api`
# MAGIC * **Emerging successor** `WorkspaceClient.data_quality` (`DataQualityAPI`, `create_monitor(Monitor(...))`
# MAGIC   with `DataProfilingConfig` / `InferenceLogConfig` / `AggregationGranularity` enums) — *also* present in
# MAGIC   `databricks-sdk==0.82.0`. It is the newer shape (unified data-quality + **UC Anomaly Detection**). We
# MAGIC   deliberately target the **GA `quality_monitors`** API for stability and document the `data_quality`
# MAGIC   equivalent at the end so the migration path is explicit. Do NOT mix the two: their field names differ
# MAGIC   (`prediction_col` vs `prediction_column`; string `"1 day"` vs `AGGREGATION_GRANULARITY_1_DAY`).
# MAGIC
# MAGIC ### Verified `quality_monitors` surface (from `databricks.sdk.service.catalog`, SDK 0.82.0)
# MAGIC * `WorkspaceClient.quality_monitors.create(table_name, output_schema_name, assets_dir, *,`
# MAGIC   `inference_log=None, custom_metrics=None, schedule=None, slicing_exprs=None, skip_builtin_dashboard=None,`
# MAGIC   `baseline_table_name=None, notifications=None, ...) -> MonitorInfo` — signature read off the installed API.
# MAGIC * `…quality_monitors.get(table_name) -> MonitorInfo` (raises `databricks.sdk.errors.NotFound` when no
# MAGIC   monitor exists on the table — this is how we make creation idempotent), and
# MAGIC   `…quality_monitors.update(table_name, output_schema_name, *, inference_log=…, custom_metrics=…, …)`.
# MAGIC * `…quality_monitors.run_refresh(table_name) -> MonitorRefreshInfo` (confirmed method name for THIS API;
# MAGIC   the `data_quality` API instead exposes `create_refresh`).
# MAGIC * `MonitorInferenceLog(problem_type, timestamp_col, granularities, prediction_col, model_id_col,`
# MAGIC   `label_col=None, prediction_proba_col=None)` — field list read off the installed dataclass.
# MAGIC * `MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION` (value `"PROBLEM_TYPE_CLASSIFICATION"`).
# MAGIC * `MonitorMetric(name, definition, input_columns, output_data_type, type)` with
# MAGIC   `MonitorMetricType.{CUSTOM_METRIC_TYPE_AGGREGATE|DERIVED|DRIFT}`. `output_data_type` is a **Spark-type
# MAGIC   JSON string**, e.g. `T.StructField("output", T.DoubleType()).json()`; the `definition` is a Jinja SQL
# MAGIC   template using `` `{{input_column}}` `` for aggregate metrics, prior metric **names** for derived
# MAGIC   metrics, and `{{current_df}}` / `{{base_df}}` for drift metrics.
# MAGIC   Ref: `https://docs.databricks.com/aws/en/lakehouse-monitoring/custom-metrics`
# MAGIC * Built-in classification metrics land in `MonitorInfo.profile_metrics_table_name` (e.g.
# MAGIC   `roc_auc_score`, `log_loss`, `accuracy_score`, `f1_score`, `precision`, `recall`); drift lands in
# MAGIC   `MonitorInfo.drift_metrics_table_name`. Exact per-version column names in those output tables are
# MAGIC   resolved **dynamically** in the retraining-signal helper (and flagged `# TODO(verify-api)`), never
# MAGIC   hard-asserted.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Dependencies
# MAGIC `databricks-sdk` (for `quality_monitors`) ships with Databricks Runtime; the monitoring APIs are most
# MAGIC stable on a recent SDK, so we pin one. No modelling / scoring libraries are needed here — this notebook
# MAGIC only reads the existing governed scores table and stands up the monitor. Safe to skip on a current
# MAGIC Runtime that already has a recent `databricks-sdk`.

# COMMAND ----------

# MAGIC %pip install -U "databricks-sdk>=0.30"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. CONFIG
# MAGIC Names are kept **verbatim** from the batch scorer (`05_…_batch_scorer.py`) and the training notebook
# MAGIC (`04_…_model_training.py`) so the monitored table references the columns the scores table *actually has*
# MAGIC and the label reproduces the training definition exactly.

# COMMAND ----------

# ---------------------------------------------------------------------------
# CONFIG — edit here.
# ---------------------------------------------------------------------------

# --- The scorer's REAL output table + its REAL columns ----------------------
# Verbatim from 05_first_order_propensity_batch_scorer.py, SECTION 5 (the idempotent Delta write):
#   internal_user_id, model_score (calibrated probability, double), raw_score (double),
#   decile (int, 1 = top decile), scoring_date (date, partition/idempotency key),
#   feature_ts (timestamp = the scoring cutoff), scored_at (timestamp), model_version (string),
#   model_run_id (string). There is NO label column — labels arrive ~90 days later.
SCORES_CATALOG = "mlops_data_science"
SCORES_SCHEMA = "default"
SCORES_TABLE_NAME = "first_order_propensity_scores"  # default output of the batch scorer

ENTITY_KEY = "internal_user_id"          # PK of the scores table (with scoring_date)
SCORE_PROBA_COL = "model_score"          # calibrated probability the scorer writes
RAW_SCORE_COL = "raw_score"              # monotonic ranking score (lineage)
DECILE_COL = "decile"                    # 1 = top decile
SCORING_DATE_COL = "scoring_date"        # DATE — the as-of / idempotency key
FEATURE_TS_COL = "feature_ts"            # TIMESTAMP — the scoring cutoff
SCORED_AT_COL = "scored_at"              # TIMESTAMP — when the scoring job ran
MODEL_VERSION_COL = "model_version"      # champion version (string) — drives model_id_col below
MODEL_RUN_ID_COL = "model_run_id"        # champion run_id (string, lineage)

# --- Label source (read-only; label ONLY; NO egress) -----------------------
# Verbatim from the training/scorer spine: the label is a first qualifying order in the performance window.
ORDERS_TABLE = "rds_main.scapiadb.orders"                    # first-order label source
USER_SEGMENT_MAPPING = "simple.crud.user_segment_mapping"    # optional current-state segment slice
SEGMENT_DEFAULT = "No Segment"                               # v7's coalesce(segment_name, 'No Segment')

# IST offset — created_at in the RDS-sourced orders is UTC; the spine anchors the window in IST and subtracts
# this offset before comparing against the UTC order times. Kept identical to 04_/05_ so the label matches.
IST_OFFSET_MINUTES = 330

# Performance (label observation) window length in days after the cutoff — training uses 90.
PERFORMANCE_DAYS = 90

# Business-confirmed qualifying filter — kept EXACTLY as the training/scorer spine has it.
QUALIFYING_STATUSES = ["COMPLETE", "CANCELLED"]
QUALIFYING_PRODUCT_CATEGORIES = [
    "FLIGHT", "BUS", "TRAIN", "HOTEL_STAY",
    "ECOMMERCE", "EXPERIENCE", "VISA", "HOLIDAY",
]

# --- Monitoring-side operational activation policy --------------------------
# Scapia ACTIVATES the top decile(s). InferenceLog classification metrics (precision/recall/F-beta) need a
# DISCRETE predicted class; the model itself emits only a score + decile, so we derive the discrete
# prediction from the decile under the documented activation policy below (this is NOT re-scoring or
# re-deriving model logic — the decile is the model's own output; targeting the top-K deciles is a
# downstream business decision that monitoring should track). ROC-AUC / log-loss still use the continuous
# probability (prediction_proba_col), so the ranking quality is monitored independently of this threshold.
TARGET_TOP_DECILES = 1  # predicted_label = 1 when decile <= this (default: target the top decile only)

# --- Monitor granularities + retraining thresholds --------------------------
# Plain-string granularities are the documented form for the quality_monitors (catalog) API
# (the data_quality API uses AGGREGATION_GRANULARITY_* enums instead — see the migration note at the end).
GRANULARITIES = ["1 day", "1 week", "1 month"]

# Documented retraining bounds (see the retraining-trigger section). Baselines should be set from the
# champion's honest TEST metrics logged in 04_…_model_training.py (test_roc_auc, test_top_decile_lift).
RETRAIN_ROC_AUC_MIN = 0.62          # retrain if a matured window's ROC-AUC decays below this
RETRAIN_TOP_DECILE_LIFT_MIN = 2.0   # retrain if top-decile lift decays below this (ranking value lost)

# ---------------------------------------------------------------------------

SCORES_TABLE = f"{SCORES_CATALOG}.{SCORES_SCHEMA}.{SCORES_TABLE_NAME}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1a. Widgets — monitored table + assets schema + slicing (fail-fast on required config)
# MAGIC * `scores_table` — the scorer's output table to monitor (default = its real output).
# MAGIC * `monitor_output_catalog` / `monitor_output_schema` — where the **metric tables + dashboard assets**
# MAGIC   land. This schema **must already exist** (the monitor writes its profile/drift tables here).
# MAGIC * `monitored_table` — the NEW governed **inference table** we materialize (scores + maturity-aware
# MAGIC   `label`). Deliberately a *new* table, never the scores table itself (we do not add a mutable label
# MAGIC   column to the immutable scores output).
# MAGIC * `include_segment_slice` — if `yes`, enrich the monitored table with **current-state** `segment`
# MAGIC   (from `user_segment_mapping`, same caveat as the scorer) and add it as a slicing dimension.
# MAGIC * `run_initial_refresh` — if `yes`, kick a first metric refresh now (consumes serverless compute).

# COMMAND ----------

dbutils.widgets.text("scores_table", SCORES_TABLE, "Scores table to monitor (catalog.schema.table).")
dbutils.widgets.text("monitor_output_catalog", SCORES_CATALOG, "Catalog for the monitor's metric tables/assets.")
dbutils.widgets.text("monitor_output_schema", SCORES_SCHEMA, "Schema for the monitor's metric tables/assets (must exist).")
dbutils.widgets.text(
    "monitored_table", f"{SCORES_CATALOG}.{SCORES_SCHEMA}.first_order_propensity_monitoring",
    "NEW governed inference table (scores + maturity-aware label) the monitor points at.",
)
dbutils.widgets.dropdown("include_segment_slice", "yes", ["yes", "no"], "Enrich + slice by current-state segment.")
dbutils.widgets.dropdown("run_initial_refresh", "no", ["yes", "no"], "Kick a metric refresh now (serverless cost).")

SCORES_TABLE = dbutils.widgets.get("scores_table").strip()
MONITOR_OUT_CATALOG = dbutils.widgets.get("monitor_output_catalog").strip()
MONITOR_OUT_SCHEMA = dbutils.widgets.get("monitor_output_schema").strip()
MONITORED_TABLE = dbutils.widgets.get("monitored_table").strip()
INCLUDE_SEGMENT = dbutils.widgets.get("include_segment_slice") == "yes"
RUN_INITIAL_REFRESH = dbutils.widgets.get("run_initial_refresh") == "yes"

# Fail fast on any blank / malformed required value — never stand up a monitor against a half-specified target.
for _name, _val in [
    ("scores_table", SCORES_TABLE),
    ("monitor_output_catalog", MONITOR_OUT_CATALOG),
    ("monitor_output_schema", MONITOR_OUT_SCHEMA),
    ("monitored_table", MONITORED_TABLE),
]:
    if not _val:
        raise ValueError(f"Required widget '{_name}' is blank. FAILING FAST.")
if SCORES_TABLE.count(".") != 2 or MONITORED_TABLE.count(".") != 2:
    raise ValueError(
        f"scores_table and monitored_table must be three-level UC names 'catalog.schema.table'; got "
        f"'{SCORES_TABLE}' and '{MONITORED_TABLE}'. FAILING FAST."
    )

# The monitor's metric tables + dashboard assets go here (must be an existing schema).
MONITOR_OUTPUT_SCHEMA = f"{MONITOR_OUT_CATALOG}.{MONITOR_OUT_SCHEMA}"

# Fail fast if the scores table the whole monitor is built on does not exist yet.
if not spark.catalog.tableExists(SCORES_TABLE):
    raise ValueError(
        f"Scores table {SCORES_TABLE} does not exist. Run 05_first_order_propensity_batch_scorer.py first so "
        f"there is a scored population to monitor. FAILING FAST."
    )

print(f"Scores table (source)     : {SCORES_TABLE}")
print(f"Monitored inference table : {MONITORED_TABLE}")
print(f"Monitor output schema     : {MONITOR_OUTPUT_SCHEMA}")
print(f"Include segment slice     : {INCLUDE_SEGMENT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 1 — Build the monitored inference table with a **maturity-aware label** (NO egress)
# MAGIC The monitor needs one table carrying the four InferenceLog inputs — **prediction**, **label**,
# MAGIC **timestamp**, **model-id** — plus the dimensions we slice on. The scores table has everything except
# MAGIC the label, so we join the **realized first orders** onto it, reproducing the training label exactly.
# MAGIC
# MAGIC ### The labels-arrive-later handling (the crux)
# MAGIC For a user scored as-of `feature_ts`, the label window is `(feature_ts, feature_ts + 90d]`. We compute a
# MAGIC **label-observation watermark** = the latest qualifying-order time we have actually ingested (IST). A
# MAGIC scoring window is only **mature** when it ends at/before that watermark. Then:
# MAGIC
# MAGIC * **window still open** (`feature_ts + 90d > watermark`) -> `label = NULL` — the outcome is **not yet
# MAGIC   observable**. We do *not* call it 0 (that would censor unmatured users and bias every metric), and we
# MAGIC   do *not* fill only the early positives (that would bias booker-rate upward). The whole open window is
# MAGIC   NULL, and the InferenceLog monitor tolerates that — drift is still computed, label metrics wait.
# MAGIC * **window matured** -> `label = 1` iff the user's first qualifying order fell in the IST-adjusted window
# MAGIC   `(feature_ts, feature_ts + 90d]`, else `0`. Because the scored population is strictly *first*-order
# MAGIC   (the scorer's anti-join already removed prior orderers), the user's **first** qualifying order is
# MAGIC   exactly the event this label counts.
# MAGIC
# MAGIC Re-running this notebook re-`MERGE`s the table, so as the watermark advances, previously-NULL labels
# MAGIC flip to their observed 0/1 and the monitor's next refresh recomputes those windows' quality metrics.

# COMMAND ----------

def _sql_in_list(values):
    """Render a Python list as a SQL IN-list of single-quoted literals (matches the sibling notebooks)."""
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def build_monitored_source_sql() -> str:
    """Scores table + maturity-aware label + activation prediction + (optional) current-state segment.

    All comparisons mirror the training/scorer spine: order created_at is UTC, the window is anchored in IST
    and shifted back by IST_OFFSET_MINUTES before comparing against the UTC order times.
    """
    statuses = _sql_in_list(QUALIFYING_STATUSES)
    categories = _sql_in_list(QUALIFYING_PRODUCT_CATEGORIES)

    seg_select = ""
    seg_join = ""
    if INCLUDE_SEGMENT:
        # Current-state segment (finding-#1 caveat, identical to the scorer): user_segment_mapping has no
        # as-of column, so this is the segment "now". Fine for a slicing dimension.
        seg_select = f", COALESCE(usm.segment_name, '{SEGMENT_DEFAULT}') AS segment"
        seg_join = f"LEFT JOIN {USER_SEGMENT_MAPPING} usm ON usm.internal_user_id = s.{ENTITY_KEY}"

    return f"""
WITH watermark AS (
    -- How far qualifying-order observations actually extend, expressed in IST (matches the scorer's probe).
    SELECT CAST(MAX(CAST(created_at AS timestamp)) + INTERVAL {IST_OFFSET_MINUTES} MINUTE AS timestamp)
               AS label_observed_until
    FROM {ORDERS_TABLE}
    WHERE status IN ({statuses})
      AND product_category IN ({categories})
),

-- FIRST qualifying order per user (UTC). The population is first-order, so MIN is that user's first order.
first_orders AS (
    SELECT user_id AS internal_user_id,
           MIN(CAST(created_at AS timestamp)) AS first_order_utc
    FROM {ORDERS_TABLE}
    WHERE status IN ({statuses})
      AND product_category IN ({categories})
      AND user_id IS NOT NULL
    GROUP BY 1
)

SELECT
    s.{ENTITY_KEY},
    s.{SCORE_PROBA_COL},
    s.{RAW_SCORE_COL},
    s.{DECILE_COL},
    -- Discrete activation prediction from the model's OWN decile (documented policy, not re-scoring).
    CASE WHEN s.{DECILE_COL} <= {TARGET_TOP_DECILES} THEN 1 ELSE 0 END AS predicted_label,
    -- Maturity-aware label: NULL while the 90-day window is still open; observed 0/1 once it has matured.
    CASE
        WHEN s.{FEATURE_TS_COL} + INTERVAL {PERFORMANCE_DAYS} DAY > w.label_observed_until
            THEN CAST(NULL AS INT)                                      -- window still open -> unknown
        WHEN fo.first_order_utc IS NOT NULL
             AND fo.first_order_utc >  s.{FEATURE_TS_COL}                                  - INTERVAL {IST_OFFSET_MINUTES} MINUTE
             AND fo.first_order_utc <= (s.{FEATURE_TS_COL} + INTERVAL {PERFORMANCE_DAYS} DAY) - INTERVAL {IST_OFFSET_MINUTES} MINUTE
            THEN 1
        ELSE 0
    END AS label,
    s.{SCORING_DATE_COL},
    CAST(s.{SCORING_DATE_COL} AS timestamp) AS scoring_ts,   -- day-aligned prediction timestamp for windows
    s.{FEATURE_TS_COL},
    s.{SCORED_AT_COL},
    s.{MODEL_VERSION_COL},
    s.{MODEL_RUN_ID_COL}{seg_select}
FROM {SCORES_TABLE} s
CROSS JOIN watermark w
LEFT JOIN first_orders fo ON fo.internal_user_id = s.{ENTITY_KEY}
{seg_join}
"""


monitored_src = spark.sql(build_monitored_source_sql())
monitored_src = monitored_src.cache()
_n_rows = monitored_src.count()
_n_labeled = monitored_src.filter("label IS NOT NULL").count()
_n_positive = monitored_src.filter("label = 1").count()
print(f"Monitored source rows : {_n_rows:,}")
print(f"  matured (label known): {_n_labeled:,}  ({_n_labeled / max(_n_rows, 1):.1%})")
print(f"  positives so far     : {_n_positive:,}")
print(f"  still open (NULL lbl): {_n_rows - _n_labeled:,}  (90-day window not yet closed)")
if _n_rows == 0:
    raise ValueError(f"{SCORES_TABLE} produced 0 monitored rows. Nothing to monitor. FAILING FAST.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1b. Materialize / backfill the monitored table (idempotent `MERGE`, Change Data Feed on)
# MAGIC First run creates the Delta table with **Change Data Feed** enabled (so the monitor can process only
# MAGIC changed rows on refresh). Later runs `MERGE` on `(internal_user_id, scoring_date)`: matured labels
# MAGIC **update** existing rows and new scoring dates **insert** — so the same date is never duplicated and
# MAGIC late labels flow in without a full rewrite.

# COMMAND ----------

_SRC_VIEW = "_fop_monitoring_src"
monitored_src.createOrReplaceTempView(_SRC_VIEW)

_KEYS = [ENTITY_KEY, SCORING_DATE_COL]
_ALL_COLS = monitored_src.columns
_SET_COLS = [c for c in _ALL_COLS if c not in _KEYS]

if not spark.catalog.tableExists(MONITORED_TABLE):
    spark.sql(
        f"""
        CREATE TABLE {MONITORED_TABLE}
        USING DELTA
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
        AS SELECT * FROM {_SRC_VIEW}
        """
    )
    _merge_mode = "created new monitored table (CDF enabled)"
else:
    _set_clause = ",\n            ".join(f"t.{c} = s.{c}" for c in _SET_COLS)
    _on_clause = " AND ".join(f"t.{k} = s.{k}" for k in _KEYS)
    spark.sql(
        f"""
        MERGE INTO {MONITORED_TABLE} t
        USING {_SRC_VIEW} s
          ON {_on_clause}
        WHEN MATCHED THEN UPDATE SET
            {_set_clause}
        WHEN NOT MATCHED THEN INSERT *
        """
    )
    _merge_mode = "merged (labels backfilled, new dates inserted)"

_tbl_rows = spark.table(MONITORED_TABLE).count()
print(f"{MONITORED_TABLE}: {_tbl_rows:,} rows [{_merge_mode}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 2 — Business custom metrics (top-decile lift + F2, and a lift-decay drift metric)
# MAGIC The built-in classification metrics (ROC-AUC, log-loss, precision/recall/F1) are necessary but not
# MAGIC sufficient: **Scapia ranks by decile**, so what matters operationally is that the **top decile keeps
# MAGIC concentrating bookers** (lift) and that the top-decile activation stays high-recall (F2 weights recall
# MAGIC 4:1). We add these as custom metrics so **lift-decay is trackable** directly:
# MAGIC
# MAGIC * **Aggregate** building blocks (per window / per slice, over rows whose label has matured — `AVG`/`SUM`
# MAGIC   naturally ignore the NULL open-window labels): TP / FP / FN for the top-decile activation, and the
# MAGIC   decile-1 vs overall booker counts.
# MAGIC * **Derived** metrics: `fop_top_decile_lift` = (decile-1 booker rate) / (overall booker rate), and
# MAGIC   `fop_f2` = 5·TP / (5·TP + 4·FN + FP). Both return `NULL` until labels mature (guarded with `nullif`).
# MAGIC * **Drift** metric: `fop_top_decile_lift_delta` = `{{current_df}}.fop_top_decile_lift -
# MAGIC   {{base_df}}.fop_top_decile_lift` — the window-over-window change in lift, i.e. **lift decay**, which
# MAGIC   the retraining trigger reads.
# MAGIC
# MAGIC (Grounded in the confirmed custom-metric syntax: `` `{{input_column}}` `` is not needed for these because
# MAGIC they span multiple columns, so `input_columns=[":table"]`; derived metrics reference prior aggregate
# MAGIC **names**; `output_data_type` is a Spark-type JSON string.)

# COMMAND ----------

from databricks.sdk.service.catalog import MonitorMetric, MonitorMetricType
from pyspark.sql import types as T

_DOUBLE = T.StructField("output", T.DoubleType()).json()
_LONG = T.StructField("output", T.LongType()).json()

CUSTOM_METRICS = [
    # --- AGGREGATE building blocks (span label/predicted_label/decile -> input_columns=[":table"]) --------
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
        name="fop_true_positive",
        input_columns=[":table"],
        definition="SUM(CASE WHEN predicted_label = 1 AND label = 1 THEN 1 ELSE 0 END)",
        output_data_type=_LONG,
    ),
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
        name="fop_false_positive",
        input_columns=[":table"],
        definition="SUM(CASE WHEN predicted_label = 1 AND label = 0 THEN 1 ELSE 0 END)",
        output_data_type=_LONG,
    ),
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
        name="fop_false_negative",
        input_columns=[":table"],
        definition="SUM(CASE WHEN predicted_label = 0 AND label = 1 THEN 1 ELSE 0 END)",
        output_data_type=_LONG,
    ),
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
        name="fop_decile1_bookers",
        input_columns=[":table"],
        definition="SUM(CASE WHEN decile = 1 AND label = 1 THEN 1 ELSE 0 END)",
        output_data_type=_LONG,
    ),
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
        name="fop_decile1_users",
        input_columns=[":table"],
        definition="SUM(CASE WHEN decile = 1 AND label IS NOT NULL THEN 1 ELSE 0 END)",
        output_data_type=_LONG,
    ),
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
        name="fop_labeled_bookers",
        input_columns=[":table"],
        definition="SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END)",
        output_data_type=_LONG,
    ),
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
        name="fop_labeled_users",
        input_columns=[":table"],
        definition="SUM(CASE WHEN label IS NOT NULL THEN 1 ELSE 0 END)",
        output_data_type=_LONG,
    ),
    # --- DERIVED metrics: reference the aggregate NAMES above; NULL-safe via nullif ----------------------
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_DERIVED,
        name="fop_top_decile_lift",
        input_columns=[":table"],
        definition=(
            "(fop_decile1_bookers / nullif(fop_decile1_users, 0)) "
            "/ nullif(fop_labeled_bookers / nullif(fop_labeled_users, 0), 0)"
        ),
        output_data_type=_DOUBLE,
    ),
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_DERIVED,
        name="fop_f2",
        input_columns=[":table"],
        definition=(
            "(5.0 * fop_true_positive) "
            "/ nullif((5.0 * fop_true_positive) + (4.0 * fop_false_negative) + fop_false_positive, 0)"
        ),
        output_data_type=_DOUBLE,
    ),
    # --- DRIFT metric: window-over-window change in lift == lift decay -----------------------------------
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_DRIFT,
        name="fop_top_decile_lift_delta",
        input_columns=[":table"],
        definition="{{current_df}}.fop_top_decile_lift - {{base_df}}.fop_top_decile_lift",
        output_data_type=_DOUBLE,
    ),
]
print(f"Defined {len(CUSTOM_METRICS)} custom metrics: {[m.name for m in CUSTOM_METRICS]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 3 — Create (or update) the InferenceLog monitor — **idempotent**
# MAGIC `problem_type = classification`, wired to:
# MAGIC * `prediction_col = predicted_label` — the discrete top-decile activation class (for precision/recall/F1),
# MAGIC * `prediction_proba_col = model_score` — the calibrated probability (for **ROC-AUC** / log-loss),
# MAGIC * `label_col = label` — the maturity-aware actual (NULL until the 90-day window closes; tolerated),
# MAGIC * `timestamp_col = scoring_ts` — day-aligned prediction time for the daily/weekly/monthly windows,
# MAGIC * `model_id_col = model_version` — so **drift is comparable across champion versions**.
# MAGIC
# MAGIC Slicing: by `decile` (and `segment` when enriched). **Idempotency:** `get()` the monitor first —
# MAGIC `update` if it already exists, `create` only on `NotFound`. We never blindly re-create.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
    MonitorCronSchedule,
)

w = WorkspaceClient()

# The four InferenceLog inputs (+ prediction_proba_col so ROC-AUC/log-loss use the probability).
INFERENCE_LOG = MonitorInferenceLog(
    problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
    prediction_col="predicted_label",     # discrete activation class
    prediction_proba_col=SCORE_PROBA_COL,  # "model_score" — calibrated probability
    label_col="label",                    # NULL until the window matures (tolerated by the monitor)
    timestamp_col="scoring_ts",
    model_id_col=MODEL_VERSION_COL,        # "model_version"
    granularities=GRANULARITIES,
)

# Slice by decile always; add segment when the monitored table was enriched with it.
SLICING_EXPRS = [DECILE_COL]
if INCLUDE_SEGMENT:
    SLICING_EXPRS.append("segment")

# Weekly refresh at 08:00 IST (labels trickle in daily; weekly keeps serverless cost modest).
SCHEDULE = MonitorCronSchedule(
    quartz_cron_expression="0 0 8 ? * MON",  # 08:00 every Monday
    timezone_id="Asia/Kolkata",
)

# assets_dir: a per-user workspace path for the monitor's dashboard/query assets. current_user() is
# resolved via spark.sql (no egress). Kept stable across runs so update() reuses the same assets.
CURRENT_USER = spark.sql("SELECT current_user() AS u").first()["u"]
ASSETS_DIR = f"/Workspace/Users/{CURRENT_USER}/lakehouse_monitoring/{MONITORED_TABLE.replace('.', '__')}"

# COMMAND ----------

def create_or_update_monitor():
    """Idempotent: update the monitor if one already exists on MONITORED_TABLE, else create it.

    Only `NotFound` (no monitor on this table yet) routes to create(); any other error propagates so a real
    failure (auth, bad schema, throttling) is never mistaken for "no monitor exists".
    """
    try:
        existing = w.quality_monitors.get(table_name=MONITORED_TABLE)
    except NotFound:
        existing = None

    if existing is not None:
        info = w.quality_monitors.update(
            table_name=MONITORED_TABLE,
            output_schema_name=MONITOR_OUTPUT_SCHEMA,
            inference_log=INFERENCE_LOG,
            custom_metrics=CUSTOM_METRICS,
            slicing_exprs=SLICING_EXPRS,
            schedule=SCHEDULE,
        )
        action = "updated existing"
    else:
        info = w.quality_monitors.create(
            table_name=MONITORED_TABLE,
            output_schema_name=MONITOR_OUTPUT_SCHEMA,
            assets_dir=ASSETS_DIR,
            inference_log=INFERENCE_LOG,
            custom_metrics=CUSTOM_METRICS,
            slicing_exprs=SLICING_EXPRS,
            schedule=SCHEDULE,
            skip_builtin_dashboard=False,  # let the monitor build its drift/quality dashboard
        )
        action = "created new"
    return info, action


monitor_info, _action = create_or_update_monitor()
print(f"Monitor {_action} on {MONITORED_TABLE}")
print(f"  status                 : {monitor_info.status}")
print(f"  profile metrics table  : {monitor_info.profile_metrics_table_name}")
print(f"  drift metrics table    : {monitor_info.drift_metrics_table_name}")
print(f"  dashboard id           : {getattr(monitor_info, 'dashboard_id', None)}")
print(f"  assets dir             : {ASSETS_DIR}")
print(f"  slicing                : {SLICING_EXPRS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b. Optional: kick an initial metric refresh
# MAGIC Creating a monitor triggers an initial refresh automatically (it analyzes the prior ~30 days on first
# MAGIC creation, then processes new/changed rows each scheduled run). This cell **optionally** forces a refresh
# MAGIC now (`run_initial_refresh=yes`) — off by default because it consumes serverless compute. Guarded so a
# MAGIC refresh hiccup never fails the notebook; the monitor is already created above.

# COMMAND ----------

if RUN_INITIAL_REFRESH:
    try:
        refresh = w.quality_monitors.run_refresh(table_name=MONITORED_TABLE)
        print(f"Kicked refresh {refresh.refresh_id} (state {refresh.state}) on {MONITORED_TABLE}.")
    except Exception as _exc:  # noqa: BLE001 — refresh is best-effort; the monitor is already stood up
        print(f"WARNING: run_refresh did not start ({_exc}); the monitor is created and will refresh on schedule.")
else:
    print("run_initial_refresh=no -> relying on the monitor's own initial + scheduled refreshes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 4 — Retraining trigger: threshold logic (doc) + a guarded read-only signal helper
# MAGIC
# MAGIC ### The trigger logic (documented)
# MAGIC The monitor writes per-window, per-slice metrics into `profile_metrics_table_name`. Once a window's
# MAGIC 90-day labels have matured, that window carries a real **ROC-AUC** and our custom **`fop_top_decile_lift`**
# MAGIC / **`fop_f2`**. A retrain is warranted when model quality decays past a **documented bound**:
# MAGIC
# MAGIC * **ROC-AUC decay** — a matured window's overall `roc_auc_score` drops below `RETRAIN_ROC_AUC_MIN`
# MAGIC   (default `0.62`), i.e. ranking power has eroded relative to the champion's honest TEST ROC-AUC
# MAGIC   (`test_roc_auc`, logged in `04_…_model_training.py`).
# MAGIC * **Top-decile lift decay** — `fop_top_decile_lift` drops below `RETRAIN_TOP_DECILE_LIFT_MIN`
# MAGIC   (default `2.0×`), i.e. the top decile no longer concentrates enough bookers to justify decile-ranked
# MAGIC   activation. The `fop_top_decile_lift_delta` **drift** metric shows the slope of that decay.
# MAGIC
# MAGIC When a bound is breached, the retrain flows into the **existing** pipeline unchanged:
# MAGIC `01 features -> 02 selection -> 03 HPO -> 04 train/register`. Per SECTION 9 of `04_…_model_training.py`,
# MAGIC training registers the new version as **`@challenger`** and **never auto-promotes over a live champion** —
# MAGIC promotion to `@champion` stays the human-gated `set_registered_model_alias(...)` step. So the trigger
# MAGIC only ever *produces a challenger to review*; it cannot silently swap the scored model.
# MAGIC
# MAGIC ### Baselines
# MAGIC Set `RETRAIN_ROC_AUC_MIN` / `RETRAIN_TOP_DECILE_LIFT_MIN` from the champion's honest TEST metrics
# MAGIC (logged at registration). A common rule is "retrain at ~90% of the champion's TEST metric" — e.g. if
# MAGIC `test_roc_auc = 0.69`, set the floor near `0.62`.

# COMMAND ----------

from pyspark.sql import functions as F


def evaluate_retraining_signal(granularity="1 week"):
    """READ-ONLY, guarded. Inspect the latest MATURED window's quality metrics and report whether a retrain
    bound is breached. Returns a dict signal; never raises, never triggers anything.

    The profile-metrics output schema is monitor-version-dependent, so column names are resolved
    DYNAMICALLY (only referenced if present) rather than hard-asserted.
    # TODO(verify-api): confirm the exact profile-metrics column names on the live monitor
    # (roc_auc_score / log_type / window / slice_key and the custom-metric columns) via
    # w.quality_monitors.get(MONITORED_TABLE).profile_metrics_table_name — they can vary across
    # monitor/SDK versions. The dynamic column probing below degrades gracefully if a name differs.
    """
    signal = {"evaluated": False, "retrain_recommended": False, "reasons": []}
    try:
        info = w.quality_monitors.get(table_name=MONITORED_TABLE)
        pm_name = info.profile_metrics_table_name
        if not pm_name or not spark.catalog.tableExists(pm_name):
            print("Profile metrics table not available yet (monitor may not have refreshed). Skipping.")
            return signal

        pm = spark.table(pm_name)
        cols = set(pm.columns)

        # Overall slice only (no slice_key) + this granularity + observed input rows (log_type='INPUT').
        df = pm
        if "granularity" in cols:
            df = df.filter(F.col("granularity") == F.lit(granularity))
        if "slice_key" in cols:
            df = df.filter(F.col("slice_key").isNull())        # overall population, not a per-slice row
        if "log_type" in cols:
            df = df.filter(F.col("log_type") == F.lit("INPUT"))

        # Only windows whose labels have matured carry a non-null ROC-AUC / lift.
        has_roc = "roc_auc_score" in cols
        has_lift = "fop_top_decile_lift" in cols
        if has_roc:
            df = df.filter(F.col("roc_auc_score").isNotNull())
        elif has_lift:
            df = df.filter(F.col("fop_top_decile_lift").isNotNull())
        else:
            print("Neither roc_auc_score nor fop_top_decile_lift present yet — labels likely not matured. Skipping.")
            return signal

        if "window" in cols:
            df = df.orderBy(F.col("window.end").desc())        # latest matured window first
        latest = df.limit(1).collect()
        if not latest:
            print("No matured windows with quality metrics yet. Skipping (labels still in the 90-day window).")
            return signal

        row = latest[0]
        signal["evaluated"] = True
        signal["window"] = str(row["window"]) if "window" in cols else None

        if has_roc and row["roc_auc_score"] is not None:
            roc = float(row["roc_auc_score"])
            signal["roc_auc"] = roc
            if roc < RETRAIN_ROC_AUC_MIN:
                signal["retrain_recommended"] = True
                signal["reasons"].append(f"ROC-AUC {roc:.4f} < floor {RETRAIN_ROC_AUC_MIN}")
        if has_lift and row["fop_top_decile_lift"] is not None:
            lift = float(row["fop_top_decile_lift"])
            signal["top_decile_lift"] = lift
            if lift < RETRAIN_TOP_DECILE_LIFT_MIN:
                signal["retrain_recommended"] = True
                signal["reasons"].append(f"top-decile lift {lift:.3f}x < floor {RETRAIN_TOP_DECILE_LIFT_MIN}x")

        if signal["retrain_recommended"]:
            print(f"RETRAIN SIGNAL: {signal['reasons']} -> kick 01->02->03->04, register @challenger, human-review.")
        else:
            print(f"Model within bounds on the latest matured window: {signal}")
        return signal
    except Exception as _exc:  # noqa: BLE001 — this is a read-only advisory helper; never fail the notebook
        print(f"WARNING: retraining-signal check did not complete ({_exc}). It is advisory only.")
        return signal


# Advisory only — safe to run on every notebook execution. It prints, and does not trigger any job.
_ = evaluate_retraining_signal()

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 5 — Future phase: the retraining-trigger **Lakeflow Job** (documented, OUT of scope)
# MAGIC The actual trigger orchestration is deliberately **not built here**. When you wire it, keep it a thin
# MAGIC scheduled job that (1) refreshes this monitor, (2) reads the signal above, and (3) *conditionally* runs
# MAGIC the existing HPO -> train/register notebooks — which register a **`@challenger`** for **human** promotion.
# MAGIC The scorer stays pinned to `@champion`, so nothing the trigger does can change the scored model without
# MAGIC a human moving the alias. Sketch (run once from any client):
# MAGIC
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks.sdk.service import jobs
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC created = w.jobs.create(
# MAGIC     name="fop_retraining_trigger_weekly",
# MAGIC     tasks=[
# MAGIC         jobs.Task(
# MAGIC             task_key="refresh_monitor_and_check",
# MAGIC             notebook_task=jobs.NotebookTask(
# MAGIC                 notebook_path="/Repos/scapia-ml/feature_store/first_order_propensity_monitoring",
# MAGIC                 base_parameters={"run_initial_refresh": "yes"},
# MAGIC             ),
# MAGIC         ),
# MAGIC         # A downstream task would read evaluate_retraining_signal() and, if breached, run the
# MAGIC         # 03_hpo + 04_training notebooks. Promotion to @champion stays a HUMAN step (see 04 SECTION 9).
# MAGIC     ],
# MAGIC     schedule=jobs.CronSchedule(
# MAGIC         quartz_cron_expression="0 0 9 ? * MON",  # 09:00 every Monday, after the monitor refresh
# MAGIC         timezone_id="Asia/Kolkata",
# MAGIC     ),
# MAGIC )
# MAGIC print(created.job_id)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 6 — Alerting stub (future phase, documented — NOT built here)
# MAGIC Alerting integrations are OUT of scope. The monitor's `notifications` field takes an email destination
# MAGIC on failure; richer routing (Slack, PagerDuty) belongs in the trigger job or a Databricks SQL alert on
# MAGIC the profile-metrics table. Documented stub only:
# MAGIC
# MAGIC ```python
# MAGIC from databricks.sdk.service.catalog import MonitorNotifications, MonitorDestination
# MAGIC
# MAGIC notifications = MonitorNotifications(
# MAGIC     on_failure=MonitorDestination(email_addresses=["ml-oncall@scapia.example"]),
# MAGIC )
# MAGIC # pass notifications=notifications into quality_monitors.create/update above.
# MAGIC # For metric-threshold alerts (e.g. ROC-AUC below the floor), create a Databricks SQL Alert on the
# MAGIC # profile-metrics table instead — that is the supported path for value-based alerting.
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## SECTION 7 — Migration note: the emerging `data_quality` API (for when you upgrade)
# MAGIC This notebook uses the **GA `quality_monitors`** API. The newer `WorkspaceClient.data_quality` API
# MAGIC (`create_monitor(Monitor(object_type="table", object_id=<table_id>, data_profiling_config=...))` with
# MAGIC `DataProfilingConfig` / `InferenceLogConfig` / `AggregationGranularity`) is the successor and also covers
# MAGIC **UC Anomaly Detection**. Field-name differences to watch when migrating:
# MAGIC * `prediction_col` -> `prediction_column`, `label_col` -> `label_column`, `timestamp_col` ->
# MAGIC   `timestamp_column`, `model_id_col` -> `model_id_column`.
# MAGIC * `granularities=["1 day", ...]` (strings) -> `granularities=[AggregationGranularity.AGGREGATION_GRANULARITY_1_DAY, ...]`.
# MAGIC * `output_schema_name="cat.sch"` -> `output_schema_id=<schema_id>`; the monitor targets `object_id`
# MAGIC   (the table id) instead of `table_name`; refresh is `create_refresh` instead of `run_refresh`.
# MAGIC
# MAGIC ```python
# MAGIC # data_quality equivalent (do NOT mix with the quality_monitors calls above):
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks.sdk.service.dataquality import (
# MAGIC     Monitor, DataProfilingConfig, InferenceLogConfig, InferenceProblemType, AggregationGranularity,
# MAGIC )
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC schema = w.schemas.get(full_name="mlops_data_science.default")
# MAGIC table = w.tables.get(full_name="mlops_data_science.default.first_order_propensity_monitoring")
# MAGIC config = DataProfilingConfig(
# MAGIC     output_schema_id=schema.schema_id,
# MAGIC     assets_dir="/Workspace/Users/<me>/lakehouse_monitoring/fop",
# MAGIC     inference_log=InferenceLogConfig(
# MAGIC         problem_type=InferenceProblemType.INFERENCE_PROBLEM_TYPE_CLASSIFICATION,
# MAGIC         prediction_column="predicted_label",
# MAGIC         model_id_column="model_version",
# MAGIC         label_column="label",
# MAGIC         timestamp_column="scoring_ts",
# MAGIC         granularities=[AggregationGranularity.AGGREGATION_GRANULARITY_1_DAY],
# MAGIC     ),
# MAGIC )
# MAGIC info = w.data_quality.create_monitor(
# MAGIC     monitor=Monitor(object_type="table", object_id=table.table_id, data_profiling_config=config),
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Out of scope (future phases — deliberately NOT built here)
# MAGIC * **Retraining-trigger job / orchestration DAB** — SECTION 5 documents the Lakeflow Job shape; wiring it
# MAGIC   (and the conditional run of `03_hpo` + `04_training`) is a separate deliverable.
# MAGIC * **Model serving / online inference** — a `@champion`-pinned real-time endpoint; this notebook monitors
# MAGIC   the *batch* scores table only.
# MAGIC * **Alerting integrations** (email/Slack/PagerDuty) beyond the SECTION 6 stub.
# MAGIC * **`segment` finding-#1 remediation** — an effective-dated / SCD2 `user_segment_mapping` joined as-of
# MAGIC   `feature_ts` (so the segment slice is point-in-time rather than current-state), tracked across
# MAGIC   `04_/05_` and here.
