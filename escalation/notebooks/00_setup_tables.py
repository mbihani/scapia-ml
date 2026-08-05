# Databricks notebook source
# MAGIC %md
# MAGIC # Scapia — Ticket Escalation — 00 · Setup Tables
# MAGIC
# MAGIC Creates (IF NOT EXISTS) the Delta tables the escalation pipeline reads and writes. This notebook is
# MAGIC **idempotent** — it never drops or truncates an existing table, so it is safe to re-run and safe to run
# MAGIC against a workspace where Greylabs is already refreshing `greylabs_raw` live.
# MAGIC
# MAGIC ## Tables created
# MAGIC | Table | Role |
# MAGIC |---|---|
# MAGIC | `greylabs_raw` | Bronze landing zone — one row per call, exact Greylabs export column names (string-typed). Greylabs refreshes this every 5 min. Created empty here only when it does not already exist. |
# MAGIC | `ticket_status` | Placeholder — `(ticket_id, status, assigned_agent, created_at, updated_at)`. Used by inference to keep only `status='open'` tickets. Pipeline degrades gracefully when empty. |
# MAGIC | `ticket_escalation_predictions` | Inference output — upserted every 5 min by `06_batch_inference`. |
# MAGIC | `ticket_escalation_feedback` | Phase-2 CX-app feedback loop — human labels on predictions. |
# MAGIC
# MAGIC The bronze `greylabs_raw` schema uses the **exact** source column names (spaces, casing preserved) so the
# MAGIC live Greylabs writer lands here unchanged; `02_silver_clean` does all renaming / typing downstream.
# MAGIC
# MAGIC > All object names come from the `catalog` / `schema` widgets — see `escalation/MANIFESTO.md`. Defaults:
# MAGIC > `CONFIGURE(catalog)`=`main`, `CONFIGURE(schema)`=`ml_escalation`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Config widgets

# COMMAND ----------

# CONFIGURE(catalog) — UC catalog for ALL escalation tables. Default 'main'.
dbutils.widgets.text("catalog", "main", "Unity Catalog catalog (CONFIGURE(catalog)).")
# CONFIGURE(schema) — UC schema for ALL escalation tables. Default 'ml_escalation'.
dbutils.widgets.text("schema", "ml_escalation", "Unity Catalog schema (CONFIGURE(schema)).")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
if not CATALOG or not SCHEMA:
    raise ValueError("Both `catalog` and `schema` widgets are required (see MANIFESTO.md).")

# Fully-qualified names — built once, never hardcoded independently downstream.
RAW_TABLE = f"{CATALOG}.{SCHEMA}.greylabs_raw"
STATUS_TABLE = f"{CATALOG}.{SCHEMA}.ticket_status"
PRED_TABLE = f"{CATALOG}.{SCHEMA}.ticket_escalation_predictions"
FEEDBACK_TABLE = f"{CATALOG}.{SCHEMA}.ticket_escalation_feedback"

print(f"catalog.schema : {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create catalog + schema (idempotent)

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS `{CATALOG}`")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")
print(f"Schema ready: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Bronze `greylabs_raw`
# MAGIC Exact Greylabs export column names, ALL string-typed (the raw feed sends everything as text). Only created
# MAGIC when absent — if Greylabs is already writing here live, this is a no-op and the live data is untouched.
# MAGIC
# MAGIC **`Transcript` is a REQUIRED source field** (`CONFIGURE`-visible in MANIFESTO). The export carries NO
# MAGIC native per-call id, so `02_silver_clean` builds `call_id` as a per-ticket-scoped COMPOSITE hash
# MAGIC `SHA256(ticket_id || call_datetime || transcript)`. This is (a) the per-call **dedup key** — distinct
# MAGIC calls sharing a `DateTime` differ in transcript, and identity cannot collide ACROSS tickets — and (b) the
# MAGIC deterministic tie-break for the pre-escalation slice's total order (BLOCKING-2). `02_silver_clean` also
# MAGIC **quarantines rows with a null/empty transcript** (logged count) before dropping the transcript text.
# MAGIC If Greylabs later adds a native per-call id, set `NATIVE_CALL_ID_COL` in `02_silver_clean` to key on it
# MAGIC directly — do NOT fall back to `(ticket_id, DateTime)`, which silently collapses distinct same-timestamp
# MAGIC calls.

# COMMAND ----------

# Backticked column names preserve the exact Greylabs export headers (spaces + casing).
spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {RAW_TABLE} (
        `Ticket Id`                                       STRING,
        `DateTime`                                        STRING,
        `Transcript`                                      STRING,
        `Total Call Duration (In Seconds)`                STRING,
        `Total Non-Speech Duration (In Seconds)`          STRING,
        `Customer Talk Duration (In Seconds)`             STRING,
        `Agent Talk Duration (In Seconds)`                STRING,
        `Weighted Average (Normalised Score)`             STRING,
        `Scapia Categorization (Category)`                STRING,
        `Scapia Categorization (Sub-Category)`            STRING,
        `QRC (Category)`                                  STRING,
        `QRC (Sub-Category)`                              STRING,
        `Empathy (Answer)`                                STRING,
        `Agent Sentiment`                                 STRING,
        `Customer Sentiment`                              STRING,
        `Real Time Alert (Answer)`                        STRING,
        `Real Time Alert (Justification)`                 STRING,
        `Repeat Calls`                                    STRING,
        `Accurate and Complete Resolution provided (Answer)` STRING,
        `Effective Communication (Answer)`                STRING,
        `Correct TAT shared (Answer)`                     STRING,
        `Scapia User Id`                                  STRING
    ) USING DELTA
    TBLPROPERTIES (
        delta.columnMapping.mode = 'name',
        delta.minReaderVersion = '2',
        delta.minWriterVersion = '5',
        comment = 'Greylabs raw call-QA export (bronze landing). One row per call. Refreshed ~every 5 min.'
    )
    """
)
print(f"Ready: {RAW_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. `ticket_status` (placeholder)
# MAGIC Left join target for inference. When empty/missing the inference notebook logs a warning and scores ALL
# MAGIC active tickets (graceful degrade), so shipping this empty is expected in Phase 1.

# COMMAND ----------

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {STATUS_TABLE} (
        ticket_id          STRING,
        status             STRING,
        assigned_agent     STRING,
        created_at         TIMESTAMP,
        updated_at         TIMESTAMP
    ) USING DELTA
    TBLPROPERTIES (comment = 'Ticket lifecycle status (open/closed/...). Placeholder — populate from CX system.')
    """
)
print(f"Ready: {STATUS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. `ticket_escalation_predictions` (inference output)
# MAGIC Upserted (MERGE on `ticket_id`) every 5 min by `06_batch_inference`. `top_features` is a JSON array of the
# MAGIC top-5 SHAP contributions per ticket.

# COMMAND ----------

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {PRED_TABLE} (
        ticket_id               STRING,
        escalation_probability  DOUBLE,
        risk_tier               STRING,
        top_features            STRING,
        shap_status             STRING,
        scored_at               TIMESTAMP,
        model_version           STRING,
        model_alias             STRING
    ) USING DELTA
    TBLPROPERTIES (comment = 'Per-ticket escalation risk scores. Upserted (MERGE on ticket_id) every 5 min. shap_status: OK|FAILED (explanations present?).')
    """
)
print(f"Ready: {PRED_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. `ticket_escalation_feedback` (Phase-2 CX feedback loop)

# COMMAND ----------

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {FEEDBACK_TABLE} (
        ticket_id       STRING,
        predicted_flag  INT,
        actual_outcome  STRING,
        feedback_by     STRING,
        feedback_at     TIMESTAMP
    ) USING DELTA
    TBLPROPERTIES (comment = 'Human feedback on escalation predictions (CX supervisor app). Phase 2.')
    """
)
print(f"Ready: {FEEDBACK_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validation — confirm all four tables exist

# COMMAND ----------

_expected = [RAW_TABLE, STATUS_TABLE, PRED_TABLE, FEEDBACK_TABLE]
for _t in _expected:
    _n = spark.table(_t).count()
    print(f"  {_t:<60} rows={_n:,}")
print(f"\nAll {len(_expected)} escalation tables present in {CATALOG}.{SCHEMA}.")
