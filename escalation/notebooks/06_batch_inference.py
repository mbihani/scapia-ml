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
    "Score tickets whose features were computed within N hours (CONFIGURE(active_ticket_window_hours)).",
)
# CONFIGURE(risk_tier_high) / CONFIGURE(risk_tier_medium)
dbutils.widgets.text("risk_tier_high", "0.7", "prob >= this -> 'high' (CONFIGURE(risk_tier_high)).")
dbutils.widgets.text("risk_tier_medium", "0.4", "prob >= this -> 'medium' (CONFIGURE(risk_tier_medium)).")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
ACTIVE_TICKET_WINDOW_HOURS = int(dbutils.widgets.get("active_ticket_window_hours"))
RISK_TIER_HIGH = float(dbutils.widgets.get("risk_tier_high"))
RISK_TIER_MEDIUM = float(dbutils.widgets.get("risk_tier_medium"))

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

if not spark.catalog.tableExists(GOLD_TABLE):
    raise RuntimeError(f"{GOLD_TABLE} missing. Run 03_gold_features (mode=inference) first. FAILING FAST.")

cutoff = F.current_timestamp() - F.expr(f"INTERVAL {ACTIVE_TICKET_WINDOW_HOURS} HOURS")
active = spark.table(GOLD_TABLE).filter(F.col("feature_computed_at") >= cutoff)
_n_active = active.count()
print(f"active tickets (features within {ACTIVE_TICKET_WINDOW_HOURS}h): {_n_active:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Left join ticket_status → keep open (graceful degrade)

# COMMAND ----------

_status_has_data = False
if spark.catalog.tableExists(STATUS_TABLE):
    _status_has_data = spark.table(STATUS_TABLE).limit(1).count() > 0

if _status_has_data:
    status = spark.table(STATUS_TABLE).select(
        F.col("ticket_id").alias("_s_ticket_id"),
        F.lower(F.trim(F.col("status"))).alias("_s_status"),
    )
    # Keep tickets that are explicitly 'open'. Unknown-status tickets (null after the left join — the status
    # table only tracks a subset) are kept too so a partial status table never silently drops a live ticket.
    scored_src = (
        active.join(status, active.ticket_id == status._s_ticket_id, "left")
        .filter((F.col("_s_status") == "open") | F.col("_s_status").isNull())
        .drop("_s_ticket_id", "_s_status")
    )
    print(f"ticket_status present -> filtered to open/unknown tickets: {scored_src.count():,}")
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

# COMMAND ----------

top_features_json = ["[]"] * len(_pdf)  # default: empty list per row if SHAP is unavailable
try:
    import shap

    # Unwrap the raw XGBoost booster + frozen preprocessing from the loaded pyfunc.
    _impl = champion_pyfunc._model_impl.python_model
    # load_context is invoked by MLflow at load; the booster/pp are attributes on the impl.
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
        # Top 5 by |contribution|.
        order = np.argsort(-np.abs(row))[:5]
        top = [
            {
                "feature": feat_names[j],
                "shap_value": round(float(row[j]), 6),
                "feature_value": (
                    None
                    if pd.isna(X_mat.iloc[i, j])
                    else round(float(X_mat.iloc[i, j]), 6)
                ),
            }
            for j in order
        ]
        top_features_json.append(json.dumps(top))
    print(f"SHAP top-5 computed for {len(top_features_json):,} tickets.")
except Exception as _exc:
    print(f"WARNING: SHAP attribution skipped ({_exc}); top_features will be empty arrays.")

_pdf["top_features"] = top_features_json

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Build the predictions frame + MERGE (upsert on ticket_id)

# COMMAND ----------

from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType,
)

out_pdf = _pdf[["ticket_id", "escalation_probability", "risk_tier", "top_features"]].copy()
out_pdf["model_version"] = MODEL_VERSION
out_pdf["model_alias"] = CHAMPION_ALIAS

_OUT_SCHEMA = StructType(
    [
        StructField("ticket_id", StringType()),
        StructField("escalation_probability", DoubleType()),
        StructField("risk_tier", StringType()),
        StructField("top_features", StringType()),
        StructField("model_version", StringType()),
        StructField("model_alias", StringType()),
    ]
)
out_sdf = spark.createDataFrame(out_pdf, schema=_OUT_SCHEMA).withColumn(
    "scored_at", F.current_timestamp()
)

if not spark.catalog.tableExists(PRED_TABLE):
    raise RuntimeError(f"{PRED_TABLE} missing. Run 00_setup_tables first. FAILING FAST.")

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
        t.scored_at              = s.scored_at,
        t.model_version          = s.model_version,
        t.model_alias            = s.model_alias
    WHEN NOT MATCHED THEN INSERT (
        ticket_id, escalation_probability, risk_tier, top_features, scored_at, model_version, model_alias
    ) VALUES (
        s.ticket_id, s.escalation_probability, s.risk_tier, s.top_features, s.scored_at, s.model_version, s.model_alias
    )
    """
)
print(f"Upserted {out_pdf.shape[0]:,} predictions into {PRED_TABLE}")

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
