# Databricks notebook source
# MAGIC %md
# MAGIC # Scapia — Ticket Escalation — 06 · Batch Inference (every 5 min)
# MAGIC
# MAGIC Scores active, still-open, not-yet-escalated tickets with the `@champion` escalation model so CX
# MAGIC supervisors can intervene **before** an escalation. Runs every 5 minutes (see `inference_job.json`) after
# MAGIC `02_silver_clean` (incremental) and `03_gold_features` (mode=inference) have refreshed the gold table.
# MAGIC
# MAGIC ## Steps (from spec)
# MAGIC 1. Load the `@champion` pyfunc from Unity Catalog (strictly champion-pinned — never 'latest').
# MAGIC 2. Read `ticket_features` where `feature_computed_at >= now() - CONFIGURE(active_ticket_window_hours)`.
# MAGIC 3. Left join `ticket_status`, keep `status='open'`. **Graceful degrade**: if the status table is
# MAGIC    missing/empty, log a warning and score ALL active tickets.
# MAGIC 4. Score → `escalation_probability`.
# MAGIC 5. Risk tier: `high` (≥ `CONFIGURE(risk_tier_high)`), `medium` (≥ `CONFIGURE(risk_tier_medium)`), else `low`.
# MAGIC 6. Compute top-5 SHAP contributions per ticket (TreeExplainer on the UNWRAPPED XGBoost booster).
# MAGIC 7. **MERGE** (upsert) into `ticket_escalation_predictions` on `ticket_id`.
# MAGIC
# MAGIC > Config via widgets — see `escalation/MANIFESTO.md`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Dependencies

# COMMAND ----------

# MAGIC %pip install -q shap
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Config widgets

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Unity Catalog catalog (CONFIGURE(catalog)).")
dbutils.widgets.text("schema", "ml_escalation", "Unity Catalog schema (CONFIGURE(schema)).")
# CONFIGURE(active_ticket_window_hours) — default 8h
dbutils.widgets.text(
    "active_ticket_window_hours", "8",
    "Score tickets whose LAST CALL is within N hours (CONFIGURE(active_ticket_window_hours)).",
)
# CONFIGURE(risk_tier_high) / CONFIGURE(risk_tier_medium)
dbutils.widgets.text("risk_tier_high", "0.7", "prob >= this -> 'high' (CONFIGURE(risk_tier_high)).")
dbutils.widgets.text("risk_tier_medium", "0.4", "prob >= this -> 'medium' (CONFIGURE(risk_tier_medium)).")
# CONFIGURE(shap_failure_mode) — how a SHAP failure is handled (item 11). Default 'fail' (loud).
dbutils.widgets.dropdown(
    "shap_failure_mode", "fail", ["fail", "flag"],
    "SHAP failure handling: 'fail' raises (default, scheduled-job-safe) | 'flag' writes shap_status='FAILED'.",
)

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
ACTIVE_TICKET_WINDOW_HOURS = int(dbutils.widgets.get("active_ticket_window_hours"))
RISK_TIER_HIGH = float(dbutils.widgets.get("risk_tier_high"))
RISK_TIER_MEDIUM = float(dbutils.widgets.get("risk_tier_medium"))
SHAP_FAILURE_MODE = dbutils.widgets.get("shap_failure_mode").strip()

GOLD_TABLE = f"{CATALOG}.{SCHEMA}.ticket_features"
STATUS_TABLE = f"{CATALOG}.{SCHEMA}.ticket_status"
PRED_TABLE = f"{CATALOG}.{SCHEMA}.ticket_escalation_predictions"
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.ticket_escalation_model"
CHAMPION_ALIAS = "champion"

print(f"gold  : {GOLD_TABLE}")
print(f"model : {REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
print(f"risk tiers: high>={RISK_TIER_HIGH}, medium>={RISK_TIER_MEDIUM}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Import shared module + load the @champion model (strictly pinned)

# COMMAND ----------

import os
import sys


def _add_notebook_dir_to_path():
    candidates = []
    try:
        nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        candidates.append("/Workspace" + os.path.dirname(nb_path))
        candidates.append(os.path.dirname(nb_path))
    except Exception:
        pass
    candidates.append(os.getcwd())
    for c in candidates:
        if c and c not in sys.path and os.path.isdir(c):
            sys.path.insert(0, c)


_add_notebook_dir_to_path()

import json

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

import escalation_features as ef

mlflow.set_registry_uri("databricks-uc")
_uc_client = MlflowClient(registry_uri="databricks-uc")

# STRICT champion pin: resolve the version behind @champion. If there is no champion, FAIL FAST — inference
# must never silently fall back to 'latest' / a challenger.
from mlflow.exceptions import RestException

try:
    _champ_mv = _uc_client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
except RestException as exc:
    raise RuntimeError(
        f"No @{CHAMPION_ALIAS} alias on {REGISTERED_MODEL_NAME}. Run 05_train_register to register + promote a "
        f"champion before scoring. Refusing to score with an unpinned model. FAILING FAST. ({exc})"
    )

MODEL_VERSION = str(_champ_mv.version)
_model_uri = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
champion_pyfunc = mlflow.pyfunc.load_model(_model_uri)
print(f"Loaded {REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS} = v{MODEL_VERSION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Read active tickets (recency window)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# UTC session tz so current_timestamp() matches the UTC instants stored in gold (item 7).
spark.conf.set("spark.sql.session.timeZone", "UTC")

if not spark.catalog.tableExists(GOLD_TABLE):
    raise RuntimeError(f"{GOLD_TABLE} missing. Run 03_gold_features (mode=inference) first. FAILING FAST.")

# "Active" keys on the ticket's LAST-CALL timestamp (a customer called recently), not on when features were
# computed. last_call_datetime is a UTC instant; compared under a UTC session so both sides agree.
cutoff = F.current_timestamp() - F.expr(f"INTERVAL {ACTIVE_TICKET_WINDOW_HOURS} HOURS")
active = spark.table(GOLD_TABLE).filter(F.col("last_call_datetime") >= cutoff)

# Defense-in-depth (BLOCKING-3): never score an ALREADY-ESCALATED ticket. 03_gold_features (mode=inference)
# already emits no record for is_escalated=1, but gold is a shared table that could have last been written in
# training mode — so we also filter here. Guarded for when the column is absent.
if "is_escalated" in active.columns:
    _before_esc = active.count()
    active = active.filter((F.col("is_escalated").isNull()) | (F.col("is_escalated") == 0))
    _dropped = _before_esc - active.count()
    if _dropped:
        print(f"excluded {_dropped:,} already-escalated tickets from scoring (BLOCKING-3).")

_n_active = active.count()
print(f"active tickets (last call within {ACTIVE_TICKET_WINDOW_HOURS}h, UTC, not-yet-escalated): {_n_active:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Left join ticket_status → keep open (graceful degrade)
# MAGIC `ticket_status` may carry multiple rows per ticket (status history). We collapse it to ONE row per
# MAGIC ticket — the latest by `updated_at` then `created_at` — BEFORE the join (item 8), so the MERGE source
# MAGIC has exactly one row per ticket and cannot produce duplicate scores or Delta MERGE ambiguity.

# COMMAND ----------

_status_has_data = False
if spark.catalog.tableExists(STATUS_TABLE):
    _status_has_data = spark.table(STATUS_TABLE).limit(1).count() > 0

if _status_has_data:
    _status_raw = spark.table(STATUS_TABLE)
    # Deduplicate to one row per ticket: latest by updated_at, then created_at (nulls last), then status text
    # as a final deterministic tie-break.
    _dedup_w = Window.partitionBy("ticket_id").orderBy(
        F.col("updated_at").desc_nulls_last(),
        F.col("created_at").desc_nulls_last(),
        F.lower(F.trim(F.col("status"))).asc_nulls_last(),
    )
    status = (
        _status_raw.withColumn("_rn", F.row_number().over(_dedup_w))
        .filter(F.col("_rn") == 1)
        .select(
            F.col("ticket_id").alias("_s_ticket_id"),
            F.lower(F.trim(F.col("status"))).alias("_s_status"),
        )
    )
    # Keep tickets that are explicitly 'open'. Unknown-status tickets (null after the left join — the status
    # table only tracks a subset) are kept too so a partial status table never silently drops a live ticket.
    scored_src = (
        active.join(status, active.ticket_id == status._s_ticket_id, "left")
        .filter((F.col("_s_status") == "open") | F.col("_s_status").isNull())
        .drop("_s_ticket_id", "_s_status")
    )
    print(f"ticket_status present (deduped 1 row/ticket) -> open/unknown tickets: {scored_src.count():,}")
else:
    print(
        f"WARNING: {STATUS_TABLE} is empty or missing — scoring ALL active tickets (graceful degrade, spec)."
    )
    scored_src = active

_pdf = scored_src.toPandas()
if len(_pdf) == 0:
    print("No tickets to score this cycle — writing nothing (idempotent). Exiting.")
    dbutils.notebook.exit(json.dumps({"scored": 0, "model_version": MODEL_VERSION}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Score with the champion pyfunc

# COMMAND ----------

X_raw = _pdf[ef.FEATURE_NAMES].copy()
preds = champion_pyfunc.predict(X_raw)
_pdf["escalation_probability"] = preds["escalation_probability"].to_numpy()

# Risk tier from thresholds.
def _tier(p):
    if p >= RISK_TIER_HIGH:
        return "high"
    if p >= RISK_TIER_MEDIUM:
        return "medium"
    return "low"


_pdf["risk_tier"] = _pdf["escalation_probability"].map(_tier)
print(_pdf["risk_tier"].value_counts().to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Top-5 SHAP contributions (TreeExplainer on the UNWRAPPED booster)
# MAGIC The pyfunc wraps an XGBoost `Booster`; SHAP's `TreeExplainer` needs that raw booster, so we unwrap it
# MAGIC from the loaded pyfunc and rebuild the SAME frozen model-matrix the model scores on.
# MAGIC
# MAGIC **SHAP failures are NOT silently swallowed (item 11).** The contract requires top-5 explanations, so a
# MAGIC scheduled run must not quietly emit empty arrays that look valid. Behaviour is governed by
# MAGIC `CONFIGURE(shap_failure_mode)`:
# MAGIC * `fail` (default) — a SHAP failure RAISES and fails the job loudly (preferred for a scheduled job).
# MAGIC * `flag` — write predictions with `shap_status='FAILED'` (and empty `top_features`) so downstream can
# MAGIC   tell explanations are missing; successful rows get `shap_status='OK'`. Chosen only when you would
# MAGIC   rather keep scoring than page on an explainer outage — the status column makes the gap explicit.

# COMMAND ----------

_shap_status = "OK"
top_features_json = None
try:
    import shap

    # Unwrap the raw XGBoost booster + frozen preprocessing from the loaded pyfunc.
    _impl = champion_pyfunc._model_impl.python_model
    _booster = _impl._booster
    _pp = _impl._pp

    X_mat = ef.build_model_matrix(
        X_raw, _pp["medians"], _pp["encode_maps"], float(_pp["global_mean"]), feature_names=_pp["feature_names"]
    )
    explainer = shap.TreeExplainer(_booster)
    sv = explainer.shap_values(X_mat)
    if isinstance(sv, list):  # some shap versions return [neg, pos] for binary
        sv = sv[-1]
    sv = np.asarray(sv)

    feat_names = list(X_mat.columns)
    top_features_json = []
    for i in range(sv.shape[0]):
        row = sv[i]
        order = np.argsort(-np.abs(row))[:5]  # top 5 by |contribution|
        top = [
            {
                "feature": feat_names[j],
                "shap_value": round(float(row[j]), 6),
                "feature_value": (
                    None if pd.isna(X_mat.iloc[i, j]) else round(float(X_mat.iloc[i, j]), 6)
                ),
            }
            for j in order
        ]
        top_features_json.append(json.dumps(top))
    print(f"SHAP top-5 computed for {len(top_features_json):,} tickets.")
except Exception as _exc:
    if SHAP_FAILURE_MODE == "flag":
        print(f"ERROR: SHAP attribution FAILED ({_exc}); writing shap_status='FAILED' (flag mode).")
        _shap_status = "FAILED"
        top_features_json = ["[]"] * len(_pdf)
    else:
        # Default: fail loudly — do NOT emit empty arrays that masquerade as valid explanations.
        raise RuntimeError(
            f"SHAP attribution failed and shap_failure_mode='fail'. Contract requires top-5 explanations for "
            f"every scored ticket; refusing to write predictions without them. Set shap_failure_mode='flag' "
            f"to write shap_status='FAILED' instead. Underlying error: {_exc}"
        )

_pdf["top_features"] = top_features_json
_pdf["shap_status"] = _shap_status

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Build the predictions frame + MERGE (upsert on ticket_id)

# COMMAND ----------

from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType,
)

out_pdf = _pdf[["ticket_id", "escalation_probability", "risk_tier", "top_features", "shap_status"]].copy()
out_pdf["model_version"] = MODEL_VERSION
out_pdf["model_alias"] = CHAMPION_ALIAS

_OUT_SCHEMA = StructType(
    [
        StructField("ticket_id", StringType()),
        StructField("escalation_probability", DoubleType()),
        StructField("risk_tier", StringType()),
        StructField("top_features", StringType()),
        StructField("shap_status", StringType()),
        StructField("model_version", StringType()),
        StructField("model_alias", StringType()),
    ]
)
out_sdf = spark.createDataFrame(out_pdf, schema=_OUT_SCHEMA).withColumn(
    "scored_at", F.current_timestamp()
)

if not spark.catalog.tableExists(PRED_TABLE):
    raise RuntimeError(f"{PRED_TABLE} missing. Run 00_setup_tables first. FAILING FAST.")

# The MERGE source has exactly one row per ticket_id (gold is one row/ticket and ticket_status was deduped),
# so this cannot raise Delta's multiple-source-rows-per-key ambiguity error (item 8).
out_sdf.createOrReplaceTempView("_escalation_scores")
spark.sql(
    f"""
    MERGE INTO {PRED_TABLE} t
    USING _escalation_scores s
    ON t.ticket_id = s.ticket_id
    WHEN MATCHED THEN UPDATE SET
        t.escalation_probability = s.escalation_probability,
        t.risk_tier              = s.risk_tier,
        t.top_features           = s.top_features,
        t.shap_status            = s.shap_status,
        t.scored_at              = s.scored_at,
        t.model_version          = s.model_version,
        t.model_alias            = s.model_alias
    WHEN NOT MATCHED THEN INSERT (
        ticket_id, escalation_probability, risk_tier, top_features, shap_status, scored_at, model_version, model_alias
    ) VALUES (
        s.ticket_id, s.escalation_probability, s.risk_tier, s.top_features, s.shap_status, s.scored_at, s.model_version, s.model_alias
    )
    """
)
print(f"Upserted {out_pdf.shape[0]:,} predictions into {PRED_TABLE} (shap_status={_shap_status})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Validation counts

# COMMAND ----------

_res = (
    spark.table(PRED_TABLE)
    .groupBy("risk_tier")
    .count()
    .orderBy("risk_tier")
)
_res.show(truncate=False)
print(f"total rows in {PRED_TABLE}: {spark.table(PRED_TABLE).count():,}")

dbutils.notebook.exit(
    json.dumps({"scored": int(out_pdf.shape[0]), "model_version": MODEL_VERSION, "model_alias": CHAMPION_ALIAS})
)
