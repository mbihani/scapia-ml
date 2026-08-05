# Databricks notebook source
# MAGIC %md
# MAGIC # Scapia — Ticket Escalation — 03 · Gold Features
# MAGIC
# MAGIC Aggregates the per-call **silver** table into the ticket-grain **gold** feature table `ticket_features`
# MAGIC — one row per ticket, ready for training and inference. The heavy lifting (pre-escalation slicing,
# MAGIC worst-of aggregation, speech ratios, sentiment trend, `neg_sentiment_ratio`) lives in the shared,
# MAGIC unit-tested module `escalation_features.py`, invoked per ticket via Spark `applyInPandas` so the
# MAGIC **exact same** Python that the tests cover runs at scale.
# MAGIC
# MAGIC ## Grain & windowing (from spec)
# MAGIC * One row per ticket. Only calls **before** the first `real_time_alert='yes'` call are used
# MAGIC   (pre-escalation window). Tickets whose FIRST call already escalated are excluded.
# MAGIC * **Label** (`is_escalated`) = 1 if ANY call in the ticket escalated, else 0. Training only.
# MAGIC * The **20 model features** = 12 numeric + 4 GL worst-of + 4 high-cardinality (raw strings; target-encoded
# MAGIC   later at training). Plus metadata columns (`first_call_datetime`, `last_call_datetime`,
# MAGIC   `total_calls_in_ticket`, `feature_computed_at`) that are NOT model inputs.
# MAGIC
# MAGIC ## Training vs inference
# MAGIC * `mode=training` → drops tickets with no valid (yes/no) alert on any call (unlabelled), writes the label.
# MAGIC * `mode=inference` → keeps every ticket with a pre-escalation window (still-open, not-yet-escalated),
# MAGIC   scored later by `06_batch_inference`. Only tickets whose last call is within
# MAGIC   `CONFIGURE(active_ticket_window_hours)` are retained.
# MAGIC
# MAGIC > Config via widgets — see `escalation/MANIFESTO.md`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Config widgets

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Unity Catalog catalog (CONFIGURE(catalog)).")
dbutils.widgets.text("schema", "ml_escalation", "Unity Catalog schema (CONFIGURE(schema)).")
dbutils.widgets.dropdown(
    "mode", "training", ["training", "inference"],
    "training = labelled, all history. inference = unlabelled active tickets only.",
)
# CONFIGURE(active_ticket_window_hours) — inference recency window. Default 8h.
dbutils.widgets.text(
    "active_ticket_window_hours", "8",
    "inference ONLY: keep tickets whose last call is within N hours (CONFIGURE(active_ticket_window_hours)).",
)

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
MODE = dbutils.widgets.get("mode").strip()
ACTIVE_TICKET_WINDOW_HOURS = int(dbutils.widgets.get("active_ticket_window_hours"))

SILVER_TABLE = f"{CATALOG}.{SCHEMA}.greylabs_calls_clean"
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.ticket_features"

print(f"silver : {SILVER_TABLE}")
print(f"gold   : {GOLD_TABLE}")
print(f"mode   : {MODE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Import the shared feature module
# MAGIC `escalation_features.py` sits next to this notebook. Add the notebook's directory to `sys.path` so the
# MAGIC import works whether this runs as a workspace notebook or a bundled file. Falls back to the repo path.

# COMMAND ----------

import os
import sys

# When run as a Databricks notebook, this resolves to the notebook's workspace dir; as a file, its parent.
def _add_notebook_dir_to_path():
    candidates = []
    try:
        nb_path = (
            dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        )
        # Workspace files are available under /Workspace + the notebook dir.
        candidates.append("/Workspace" + os.path.dirname(nb_path))
        candidates.append(os.path.dirname(nb_path))
    except Exception:
        pass
    candidates.append(os.getcwd())
    for c in candidates:
        if c and c not in sys.path and os.path.isdir(c):
            sys.path.insert(0, c)


_add_notebook_dir_to_path()
import escalation_features as ef  # noqa: E402

print(f"feature module loaded — {len(ef.FEATURE_NAMES)} model features:")
print(f"  numeric   ({len(ef.NUMERIC_FEATURES)}): {ef.NUMERIC_FEATURES}")
print(f"  gl        ({len(ef.CATEGORICAL_GL_FEATURES)}): {ef.CATEGORICAL_GL_FEATURES}")
print(f"  high-card ({len(ef.CATEGORICAL_HIGH_CARD_FEATURES)}): {ef.CATEGORICAL_HIGH_CARD_FEATURES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load silver (valid calls only)

# COMMAND ----------

from pyspark.sql import functions as F

# UTC session tz so current_timestamp() (used in the active-ticket filter) and stored instants agree (item 7).
spark.conf.set("spark.sql.session.timeZone", "UTC")

silver = spark.table(SILVER_TABLE).filter("NOT is_quarantined")

# Only the columns the feature builder needs (keeps the pandas groups small). `call_id` is included so the
# feature module can break tied `call_datetime` values deterministically (item 4); `call_datetime` here is the
# UTC instant produced by 02_silver_clean.
FEATURE_INPUT_COLS = [
    "ticket_id", "call_id", "call_datetime",
    "total_call_duration", "total_nonspeech_duration",
    "agent_talk_duration", "customer_talk_duration", "weighted_average",
    "empathy", "agent_sentiment", "customer_sentiment", "effective_communication",
    "real_time_alert", "repeat_call_flag",
    "scapia_category", "scapia_sub_category", "qrc_category", "qrc_sub_category",
]
silver = silver.select(*FEATURE_INPUT_COLS)
print(f"valid calls        : {silver.count():,}")
print(f"distinct tickets   : {silver.select('ticket_id').distinct().count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Per-ticket aggregation via `applyInPandas`
# MAGIC Each ticket's calls arrive as one pandas DataFrame; `build_ticket_feature_record` runs the exact
# MAGIC unit-tested logic and returns the gold row (or nothing, when the ticket is excluded).

# COMMAND ----------

from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, TimestampType,
)

# Output schema of the gold feature row (must match build_ticket_feature_record keys + feature_computed_at).
GOLD_SCHEMA = StructType(
    [
        StructField("ticket_id", StringType()),
        StructField("is_escalated", IntegerType()),
        StructField("first_call_datetime", TimestampType()),
        StructField("last_call_datetime", TimestampType()),
        StructField("total_calls_in_ticket", IntegerType()),
        # numeric features
        StructField("num_calls_considered", IntegerType()),
        StructField("time_diff_hours", DoubleType()),
        StructField("calls_per_day", DoubleType()),
        StructField("agent_speech_ratio", DoubleType()),
        StructField("customer_speech_ratio", DoubleType()),
        StructField("customer_sentiment_trend", DoubleType()),
        StructField("total_call_duration", DoubleType()),
        StructField("total_nonspeech_duration", DoubleType()),
        StructField("agent_talk_duration", DoubleType()),
        StructField("customer_talk_duration", DoubleType()),
        StructField("weighted_average", DoubleType()),
        StructField("neg_sentiment_ratio", DoubleType()),
        # GL worst-of features
        StructField("gl_empathy", StringType()),
        StructField("gl_agent_sentiment", StringType()),
        StructField("gl_customer_sentiment", StringType()),
        StructField("gl_repeat_calls", StringType()),
        # high-cardinality features
        StructField("scapia_category", StringType()),
        StructField("scapia_sub_category", StringType()),
        StructField("qrc_category", StringType()),
        StructField("qrc_sub_category", StringType()),
    ]
)

_GOLD_COLS = [f.name for f in GOLD_SCHEMA.fields]
_IS_TRAINING = MODE == "training"


def _aggregate_ticket(pdf):
    """Spark applyInPandas UDF: one ticket's calls -> zero or one gold row."""
    import pandas as pd

    # Re-import inside the executor (module ships alongside the notebook / bundle).
    import escalation_features as _ef

    rec = _ef.build_ticket_feature_record(pdf, training=_IS_TRAINING)
    if rec is None:
        return pd.DataFrame(columns=_GOLD_COLS)
    # Ensure every gold column is present and column order matches the schema.
    row = {c: rec.get(c) for c in _GOLD_COLS}
    return pd.DataFrame([row], columns=_GOLD_COLS)


gold = silver.groupBy("ticket_id").applyInPandas(_aggregate_ticket, schema=GOLD_SCHEMA)

# Stamp compute time in Spark (spec: feature_computed_at uses current_timestamp()).
gold = gold.withColumn("feature_computed_at", F.current_timestamp())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Inference recency filter
# MAGIC In inference mode keep only tickets whose LAST CALL is recent (still "active"). This filters on the
# MAGIC ticket's actual last-call timestamp (`last_call_datetime`, a UTC instant), NOT on when features were
# MAGIC computed — a ticket is "active" because a customer called recently. Compared against `current_timestamp()`
# MAGIC under a UTC session tz, so both sides are the same instant scale (item 7). Training keeps all history.

# COMMAND ----------

if MODE == "inference":
    from pyspark.sql import functions as F

    cutoff = F.current_timestamp() - F.expr(f"INTERVAL {ACTIVE_TICKET_WINDOW_HOURS} HOURS")
    before = gold.count()
    gold = gold.filter(F.col("last_call_datetime") >= cutoff)
    print(f"inference recency filter: kept tickets with last call within {ACTIVE_TICKET_WINDOW_HOURS}h (UTC)")
    print(f"  {before:,} -> {gold.count():,} tickets")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write gold

# COMMAND ----------

# Gold is a full snapshot of the current feature state; overwrite is correct for both modes (inference reads
# the latest snapshot; training reads the labelled snapshot). Downstream reads by feature_computed_at.
(
    gold.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_TABLE)
)
print(f"Wrote {GOLD_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validation counts

# COMMAND ----------

out = spark.table(GOLD_TABLE)
_n = out.count()
print(f"gold ticket rows : {_n:,}")
if _IS_TRAINING and _n > 0:
    _pos = out.filter("is_escalated = 1").count()
    print(f"  escalated (1)  : {_pos:,} ({_pos / _n:.3%})")
    print(f"  not escalated  : {_n - _pos:,}")
    if _pos > 0:
        print(f"  scale_pos_weight (neg/pos) = {(_n - _pos) / _pos:.3f}")
# Null audit on model features.
print("\nnull counts per model feature:")
_null_exprs = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in ef.FEATURE_NAMES]
out.select(*_null_exprs).show(truncate=False, vertical=True)
