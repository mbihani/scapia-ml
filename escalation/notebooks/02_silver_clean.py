# Databricks notebook source
# MAGIC %md
# MAGIC # Scapia — Ticket Escalation — 02 · Silver Clean
# MAGIC
# MAGIC Reads bronze `greylabs_raw` (exact Greylabs column names, all string-typed) and produces the typed,
# MAGIC deduplicated **silver** table `greylabs_calls_clean` — one clean row per call.
# MAGIC
# MAGIC ## What it does (all rules from the data-quality spec)
# MAGIC * **Quarantine** rows with `Ticket Id` in `('undefined', '-', null)` — flagged `is_quarantined=true`, kept
# MAGIC   (not dropped) so nothing is silently lost, but excluded downstream.
# MAGIC * Treat `-`, `NA`, `N/A` (and blank/`nan`/`none`) as **null** across all columns.
# MAGIC * Normalize string columns: `strip().lower()` for matching; QRC sub-category additionally title-cased.
# MAGIC * `DateTime` parsed `dd-MM-yyyy HH:mm:ss` and stamped **IST (UTC+05:30)** — stored as a
# MAGIC   timezone-aware timestamp.
# MAGIC * Numeric columns cast to int/float; `Total Non-Speech Duration` **clamped ≥ 0**; `Weighted Average`
# MAGIC   `0.0` → null.
# MAGIC * Sentiment/quality columns mapped to `{bad, neutral, good/positive, null}`.
# MAGIC * `Repeat Calls` free text → `repeat_call_flag` boolean (`'Repeat Call: Yes'` → true).
# MAGIC * **Dropped** columns: `Accurate and Complete Resolution…`, `Correct TAT shared`, `Scapia User Id`,
# MAGIC   `Real Time Alert (Justification)`, `Repeat Calls` (raw text — replaced by the parsed flag).
# MAGIC
# MAGIC ## Incremental
# MAGIC `run_mode=full` rebuilds the whole silver table (overwrite). `run_mode=incremental` (used by the 5-min
# MAGIC inference job) processes only calls whose `DateTime` is within the last `CONFIGURE(inference_lookback_hours)`
# MAGIC and MERGEs them, keeping silver cheap to refresh.
# MAGIC
# MAGIC > Config via widgets — see `escalation/MANIFESTO.md`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Config widgets

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Unity Catalog catalog (CONFIGURE(catalog)).")
dbutils.widgets.text("schema", "ml_escalation", "Unity Catalog schema (CONFIGURE(schema)).")
dbutils.widgets.dropdown(
    "run_mode", "full", ["full", "incremental"],
    "full = rebuild all silver (overwrite). incremental = MERGE only recent calls (5-min inference job).",
)
# CONFIGURE(inference_lookback_hours) — incremental window. Default 8h.
dbutils.widgets.text(
    "inference_lookback_hours", "8",
    "incremental ONLY: only re-process calls newer than now()-N hours (CONFIGURE(inference_lookback_hours)).",
)

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
RUN_MODE = dbutils.widgets.get("run_mode").strip()
INFERENCE_LOOKBACK_HOURS = int(dbutils.widgets.get("inference_lookback_hours"))

RAW_TABLE = f"{CATALOG}.{SCHEMA}.greylabs_raw"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.greylabs_calls_clean"

# IST is UTC+05:30. Greylabs `DateTime` has no tz — it IS local IST wall-clock time.
IST_ZONE = "Asia/Kolkata"

print(f"raw    : {RAW_TABLE}")
print(f"silver : {SILVER_TABLE}")
print(f"run_mode: {RUN_MODE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load bronze (optionally windowed for incremental)

# COMMAND ----------

from pyspark.sql import functions as F

if not spark.catalog.tableExists(RAW_TABLE):
    raise RuntimeError(f"{RAW_TABLE} missing. Run 00_setup_tables / 01_bronze_ingest first. FAILING FAST.")

raw = spark.table(RAW_TABLE)

# Parse DateTime early — needed both for the incremental window and the silver output.
# The parsed timestamp is treated as IST wall-clock (no tz shift); we then attach IST as the session-agnostic
# zone below so the stored value is timezone-correct regardless of the cluster's session tz.
raw = raw.withColumn("_call_dt_ist", F.to_timestamp(F.col("`DateTime`"), "dd-MM-yyyy HH:mm:ss"))

if RUN_MODE == "incremental":
    # now() in IST minus the lookback window. from_utc_timestamp(current, IST) gives IST wall clock.
    cutoff = F.from_utc_timestamp(F.current_timestamp(), IST_ZONE) - F.expr(
        f"INTERVAL {INFERENCE_LOOKBACK_HOURS} HOURS"
    )
    raw = raw.filter(F.col("_call_dt_ist") >= cutoff)
    print(f"incremental: keeping calls newer than now(IST) - {INFERENCE_LOOKBACK_HOURS}h")

_n_raw = raw.count()
print(f"bronze rows in scope : {_n_raw:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Cleaning helpers (SQL expressions)
# MAGIC Column-name → clean-value logic. All null-sentinel handling lives in `_clean_str`, applied before any
# MAGIC value mapping so `-`/`NA`/`N/A`/blank are uniformly null.

# COMMAND ----------

# Values that mean "missing" anywhere in the feed (lowercased comparison).
NULL_SENTINELS = ["-", "na", "n/a", "nan", "none", "null", ""]
_sentinel_sql = ", ".join([f"'{s}'" for s in NULL_SENTINELS])


def _clean_str(col_expr: str):
    """trim+lower a raw column, mapping any sentinel to NULL."""
    e = F.lower(F.trim(F.col(col_expr)))
    return F.when(e.isin([s for s in NULL_SENTINELS]), None).otherwise(e)


def _sentiment_map(clean_col):
    """Map a cleaned sentiment/quality value to {bad, neutral, positive, null}.

    Raw vocab: Bad/Negative/Profane -> bad ; Neutral -> neutral ; Good/Positive/Yes -> positive.
    (Uses 'positive' as the top bucket per the silver schema; the model treats good==positive.)
    """
    return (
        F.when(clean_col.isin("bad", "negative", "profane", "no"), F.lit("bad"))
        .when(clean_col == "neutral", F.lit("neutral"))
        .when(clean_col.isin("good", "positive", "yes"), F.lit("positive"))
        .otherwise(F.lit(None))
    )


def _good_neutral_bad_map(clean_col):
    """Map empathy / effective_communication to {bad, neutral, good, null}."""
    return (
        F.when(clean_col.isin("bad", "negative", "profane", "no"), F.lit("bad"))
        .when(clean_col == "neutral", F.lit("neutral"))
        .when(clean_col.isin("good", "positive", "yes"), F.lit("good"))
        .otherwise(F.lit(None))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Build the silver DataFrame

# COMMAND ----------

_tid = _clean_str("`Ticket Id`")

silver = raw.select(
    # ---- identity / time ----
    _tid.alias("ticket_id"),
    # Attach IST zone so the stored timestamp is timezone-correct (UTC+05:30).
    F.to_utc_timestamp(F.col("_call_dt_ist"), IST_ZONE).alias("call_datetime_utc"),
    F.col("_call_dt_ist").alias("call_datetime"),  # IST wall-clock (as parsed)
    # ---- numeric (cast + clamp) ----
    F.col("`Total Call Duration (In Seconds)`").cast("int").alias("total_call_duration"),
    F.greatest(F.col("`Total Non-Speech Duration (In Seconds)`").cast("int"), F.lit(0)).alias(
        "total_nonspeech_duration"
    ),  # negative non-speech clamped to 0
    F.col("`Customer Talk Duration (In Seconds)`").cast("int").alias("customer_talk_duration"),
    F.col("`Agent Talk Duration (In Seconds)`").cast("int").alias("agent_talk_duration"),
    # Weighted Average 0.0 -> null
    F.when(
        F.col("`Weighted Average (Normalised Score)`").cast("double") == 0.0, None
    ).otherwise(F.col("`Weighted Average (Normalised Score)`").cast("double")).alias("weighted_average"),
    # ---- high-cardinality categories (keep display case; only sentinels -> null) ----
    F.when(_clean_str("`Scapia Categorization (Category)`").isNull(), None)
    .otherwise(F.trim(F.col("`Scapia Categorization (Category)`"))).alias("scapia_category"),
    F.when(_clean_str("`Scapia Categorization (Sub-Category)`").isNull(), None)
    .otherwise(F.trim(F.col("`Scapia Categorization (Sub-Category)`"))).alias("scapia_sub_category"),
    F.when(_clean_str("`QRC (Category)`").isNull(), None)
    .otherwise(F.trim(F.col("`QRC (Category)`"))).alias("qrc_category"),
    # QRC sub-category: normalize inconsistent casing -> initcap of the cleaned value.
    F.when(_clean_str("`QRC (Sub-Category)`").isNull(), None)
    .otherwise(F.initcap(_clean_str("`QRC (Sub-Category)`"))).alias("qrc_sub_category"),
    # ---- sentiment / quality (mapped) ----
    _good_neutral_bad_map(_clean_str("`Empathy (Answer)`")).alias("empathy"),
    _sentiment_map(_clean_str("`Agent Sentiment`")).alias("agent_sentiment"),
    _sentiment_map(_clean_str("`Customer Sentiment`")).alias("customer_sentiment"),
    _good_neutral_bad_map(_clean_str("`Effective Communication (Answer)`")).alias(
        "effective_communication"
    ),
    # ---- target ----
    F.when(_clean_str("`Real Time Alert (Answer)`").isin("no", "yes", "inconclusive"), _clean_str(
        "`Real Time Alert (Answer)`"
    )).otherwise(F.lit(None)).alias("real_time_alert"),
    # ---- repeat-call flag parsed from free text ----
    F.when(F.lower(F.col("`Repeat Calls`")).contains("repeat call: yes"), F.lit(True))
    .when(F.lower(F.col("`Repeat Calls`")).contains("repeat call: no"), F.lit(False))
    .otherwise(F.lit(None)).alias("repeat_call_flag"),
)

# Quarantine flag: bad/blank ticket ids.
silver = silver.withColumn(
    "is_quarantined",
    F.col("ticket_id").isNull() | F.col("ticket_id").isin("undefined", "-"),
).withColumn("ingested_at", F.current_timestamp())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Deduplicate on the natural key `(ticket_id, call_datetime)`
# MAGIC The grain is one row per call. Keep the most-recently-ingested row per key so a re-scored call replaces
# MAGIC its earlier version rather than duplicating it.

# COMMAND ----------

from pyspark.sql.window import Window

w = Window.partitionBy("ticket_id", "call_datetime").orderBy(F.col("ingested_at").desc())
silver = (
    silver.withColumn("_rn", F.row_number().over(w))
    .filter(F.col("_rn") == 1)
    .drop("_rn")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write silver
# MAGIC `full` → overwrite. `incremental` → MERGE on the natural key so only recent calls are touched.

# COMMAND ----------

if RUN_MODE == "full" or not spark.catalog.tableExists(SILVER_TABLE):
    (
        silver.write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(SILVER_TABLE)
    )
    print(f"Overwrote {SILVER_TABLE}")
else:
    silver.createOrReplaceTempView("_silver_updates")
    spark.sql(
        f"""
        MERGE INTO {SILVER_TABLE} t
        USING _silver_updates s
        ON t.ticket_id <=> s.ticket_id AND t.call_datetime <=> s.call_datetime
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """
    )
    print(f"Merged incremental updates into {SILVER_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validation counts

# COMMAND ----------

out = spark.table(SILVER_TABLE)
_n = out.count()
_n_quar = out.filter("is_quarantined").count()
_n_valid = _n - _n_quar
_n_tickets = out.filter("NOT is_quarantined").select("ticket_id").distinct().count()
print(f"silver rows total     : {_n:,}")
print(f"  quarantined         : {_n_quar:,}")
print(f"  valid calls         : {_n_valid:,}")
print(f"  distinct valid tickets: {_n_tickets:,}")
print("\nreal_time_alert distribution (valid rows):")
out.filter("NOT is_quarantined").groupBy("real_time_alert").count().orderBy("real_time_alert").show(truncate=False)
