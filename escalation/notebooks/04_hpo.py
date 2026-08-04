# Databricks notebook source
# MAGIC %md
# MAGIC # Scapia — Ticket Escalation — 04 · Hyperparameter Optimization
# MAGIC
# MAGIC Tunes the XGBoost escalation classifier with **Optuna**, distributed across Spark executors via
# MAGIC `MlflowSparkStudy` + `MlflowStorage` (the same house HPO mechanism the FOP pipeline uses). Every trial is
# MAGIC persisted as an MLflow run; the winning params are logged to a single `escalation_hpo_champion` run that
# MAGIC `05_train_register` consumes.
# MAGIC
# MAGIC ## Data discipline (no leakage)
# MAGIC * Read gold `ticket_features`, build the raw 20-feature matrix + `is_escalated` label.
# MAGIC * **80/20 stratified** train/test split (seed 42). **TEST is reserved** for `05_train_register`'s honest
# MAGIC   evaluation and is never seen here.
# MAGIC * From TRAIN carve **VAL = 15%** (stratified). Preprocessing (median impute, K-Fold target encoding, GL
# MAGIC   ordinal) is **fit on the HPO-fit portion only** and applied to VAL — the trial objective is scored on VAL.
# MAGIC * `scale_pos_weight` is **fixed** = neg/pos of the HPO-fit split (per spec — not tuned).
# MAGIC
# MAGIC ## Search space (Optuna)
# MAGIC `n_estimators` 200–2000 (log) · `max_depth` 2–6 · `learning_rate` 0.01–0.3 (log) · `subsample` 0.6–1.0 ·
# MAGIC `colsample_bytree` 0.6–1.0 · `reg_lambda` 0.1–10 (log) · `reg_alpha` 0.0–1.0 · `min_child_weight` 1–20.
# MAGIC Fixed: `eval_metric='logloss'`, `seed=42`, `device='cpu'`.
# MAGIC
# MAGIC ## Objective
# MAGIC Default `pr_auc` (average precision) — a **threshold-free** ranking metric well-suited to this imbalanced,
# MAGIC recall-first problem: HPO optimizes how well the model *ranks* would-be escalations; the operating **F2
# MAGIC threshold** is chosen separately on VAL in `05`. `roc_auc` and `f2_max` are also selectable.
# MAGIC
# MAGIC > `MlflowSparkStudy` **minimizes** (no `direction` arg), so the objective returns the **negated** metric.
# MAGIC > Config via widgets — see `escalation/MANIFESTO.md`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Dependencies
# MAGIC `xgboost`, `optuna`, `scikit-learn` ship with the Databricks ML Runtime. The pin below is a safety net on
# MAGIC non-ML runtimes; safe to skip on a current ML Runtime.

# COMMAND ----------

# MAGIC %pip install -q optuna
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Config widgets

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Unity Catalog catalog (CONFIGURE(catalog)).")
dbutils.widgets.text("schema", "ml_escalation", "Unity Catalog schema (CONFIGURE(schema)).")
# CONFIGURE(mlflow_experiment_path)
dbutils.widgets.text(
    "mlflow_experiment_path", "/Shared/scapia_ticket_escalation",
    "MLflow experiment path (CONFIGURE(mlflow_experiment_path)).",
)
# CONFIGURE(hpo_n_trials) — default 50
dbutils.widgets.text("hpo_n_trials", "50", "Optuna trials (CONFIGURE(hpo_n_trials)).")
# CONFIGURE(n_folds) — default 5
dbutils.widgets.text("n_folds", "5", "K-Fold target-encoding folds (CONFIGURE(n_folds)).")
dbutils.widgets.text("n_jobs", "-1", "Parallel trials across Spark executors (-1 = match Spark tasks).")
dbutils.widgets.dropdown(
    "hpo_objective", "pr_auc", ["pr_auc", "roc_auc", "f2_max"],
    "Validation metric to maximize. pr_auc (default, threshold-free) suits the imbalanced recall-first goal.",
)

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
EXPERIMENT_PATH = dbutils.widgets.get("mlflow_experiment_path").strip()
N_TRIALS = int(dbutils.widgets.get("hpo_n_trials"))
N_FOLDS = int(dbutils.widgets.get("n_folds"))
N_JOBS = int(dbutils.widgets.get("n_jobs"))
HPO_OBJECTIVE = dbutils.widgets.get("hpo_objective").strip()

GOLD_TABLE = f"{CATALOG}.{SCHEMA}.ticket_features"
RANDOM_STATE = 42

print(f"gold table  : {GOLD_TABLE}")
print(f"experiment  : {EXPERIMENT_PATH}")
print(f"n_trials    : {N_TRIALS}  |  n_folds: {N_FOLDS}  |  objective: {HPO_OBJECTIVE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Import shared feature module + set experiment

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
import escalation_features as ef  # noqa: E402
import mlflow  # noqa: E402

mlflow.set_experiment(EXPERIMENT_PATH)
print("Experiment set. Model features:", ef.FEATURE_NAMES)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load gold + split (test held out, then carve VAL from train)

# COMMAND ----------

import numpy as np
import pandas as pd

pdf = spark.table(GOLD_TABLE).toPandas()
if ef.LABEL_COL not in pdf.columns or pdf[ef.LABEL_COL].isna().all():
    raise RuntimeError(
        f"Gold table {GOLD_TABLE} has no usable `{ef.LABEL_COL}` — run 03_gold_features with mode=training first."
    )

# DETERMINISTIC split BY TICKET ID (item 3). A ticket's split is a pure function of its id + seed, so HPO
# (here) and 05_train_register compute the IDENTICAL fit/val/test partition regardless of Spark row order.
pdf = pdf.reset_index(drop=True)
_split = ef.assign_split_by_ticket(
    pdf[ef.ID_COL], test_frac=0.20, val_frac_of_train=0.15, seed=RANDOM_STATE
)
X_all = pdf[ef.FEATURE_NAMES].copy()
y_all = pdf[ef.LABEL_COL].astype(int).copy()

X_fit = X_all[_split.values == "train"].reset_index(drop=True)
y_fit = y_all[_split.values == "train"].reset_index(drop=True)
X_val = X_all[_split.values == "val"].reset_index(drop=True)
y_val = y_all[_split.values == "val"].reset_index(drop=True)
X_test = X_all[_split.values == "test"].reset_index(drop=True)  # reserved (never seen by HPO)
print(f"gold rows: {len(pdf):,}  |  positives: {int(y_all.sum()):,} ({y_all.mean():.3%})")
print(f"fit: {len(X_fit):,}  |  val: {len(X_val):,}  |  test(reserved): {len(X_test):,}")

# scale_pos_weight FIXED from the fit split (spec).
spw = float((y_fit == 0).sum()) / float(max((y_fit == 1).sum(), 1))
print(f"scale_pos_weight (fixed, fit neg/pos) = {spw:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Preprocess (fit on HPO-fit only) → numeric matrices for XGBoost
# MAGIC No val labels enter any encoder: numeric medians + GL ordinals + the target-encoding maps are all fit on
# MAGIC the FIT portion only. The booster trains on OUT-OF-FOLD target encodings (leakage-free); VAL — the fold
# MAGIC each trial is scored on — is encoded with the fit-only maps and never contributes its labels (item 2).

# COMMAND ----------

# Median impute (frozen medians from FIT only) on numeric features.
medians = ef.compute_medians(X_fit, ef.NUMERIC_FEATURES)
X_fit_enc = ef.apply_medians(X_fit, medians, ef.NUMERIC_FEATURES)
X_val_enc = ef.apply_medians(X_val, medians, ef.NUMERIC_FEATURES)

# GL ordinal encoding (label-free map).
for c in ef.CATEGORICAL_GL_FEATURES:
    X_fit_enc[c] = ef.encode_gl_ordinal(X_fit[c])
    X_val_enc[c] = ef.encode_gl_ordinal(X_val[c])

# High-card target encoding:
#   * TRAIN (fit) rows -> OUT-OF-FOLD encoding (leakage-free; a row's own label never enters its value).
#     apply_medians already carried the raw high-card string columns through, so kfold_oof_encode reads them.
#   * VAL rows         -> fit-only FULL maps (val labels never enter any map).
X_fit_enc = ef.kfold_oof_encode(
    X_fit_enc, y_fit, ef.CATEGORICAL_HIGH_CARD_FEATURES,
    n_splits=N_FOLDS, smoothing=10.0, random_state=RANDOM_STATE,
)
_fit_maps, _fit_gm = ef.compute_target_encode_maps(
    X_fit, y_fit, ef.CATEGORICAL_HIGH_CARD_FEATURES, smoothing=10.0
)
for c in ef.CATEGORICAL_HIGH_CARD_FEATURES:
    X_val_enc[c] = ef.apply_target_encode(X_val[c], _fit_maps.get(c, {}), _fit_gm)

# Frozen column order.
X_fit_mat = X_fit_enc[ef.FEATURE_NAMES].astype("float32")
X_val_mat = X_val_enc[ef.FEATURE_NAMES].astype("float32")
print(f"encoded fit matrix (OOF): {X_fit_mat.shape}  |  val matrix (fit-maps): {X_val_mat.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Broadcast the encoded data + define the objective
# MAGIC The (small) encoded matrices are broadcast so executors running distributed trials share one copy.

# COMMAND ----------

sc = spark.sparkContext
_HPO_DATA = sc.broadcast(
    {
        "X_fit": X_fit_mat.to_numpy(dtype="float32"),
        "y_fit": y_fit.to_numpy(dtype="int32"),
        "X_val": X_val_mat.to_numpy(dtype="float32"),
        "y_val": y_val.to_numpy(dtype="int32"),
        "feature_names": list(ef.FEATURE_NAMES),
        "spw": spw,
        "objective_metric": HPO_OBJECTIVE,
    }
)

# Fixed XGBoost params (spec).
XGB_FIXED = {"eval_metric": "logloss", "seed": RANDOM_STATE, "device": "cpu"}


def _business_metric(y_true, proba, metric):
    """Validation metric to MAXIMIZE. All higher-is-better."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if metric == "roc_auc":
        return float(roc_auc_score(y_true, proba))
    if metric == "f2_max":
        # best F2 achievable over the grid (uses shared module's threshold sweep).
        _, best = ef.best_f2_threshold(y_true, proba)
        return float(best)
    # default pr_auc
    return float(average_precision_score(y_true, proba))


def objective(trial):
    """One distributed trial: fit XGBoost with tuned n_estimators, score the (negated) VAL metric."""
    import xgboost as xgb

    data = _HPO_DATA.value
    params = {
        **XGB_FIXED,
        "objective": "binary:logistic",
        "tree_method": "hist",
        "scale_pos_weight": data["spw"],  # FIXED (not tuned)
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    }
    n_estimators = trial.suggest_int("n_estimators", 200, 2000, log=True)

    dtrain = xgb.DMatrix(data["X_fit"], label=data["y_fit"], feature_names=data["feature_names"])
    dval = xgb.DMatrix(data["X_val"], label=data["y_val"], feature_names=data["feature_names"])
    booster = xgb.train(params, dtrain, num_boost_round=n_estimators, verbose_eval=False)

    val_proba = booster.predict(dval)
    score = _business_metric(data["y_val"], val_proba, data["objective_metric"])
    trial.set_user_attr("n_estimators", n_estimators)
    return -float(score)  # study MINIMIZES -> negate to maximize the metric


print("Objective defined. Fixed params:", XGB_FIXED, "| fixed spw:", round(spw, 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Run the distributed study

# COMMAND ----------

import hashlib

import optuna
from mlflow.optuna.storage import MlflowStorage
from mlflow.pyspark.optuna.study import MlflowSparkStudy
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id
mlflow_storage = MlflowStorage(experiment_id=experiment_id)

# Study name = fingerprint of the config that defines trial comparability. Any change starts a fresh study.
_fp_payload = "|".join(
    [
        f"gold={GOLD_TABLE}",
        "features=" + ",".join(ef.FEATURE_NAMES),
        f"objective={HPO_OBJECTIVE}",
        f"nfolds={N_FOLDS}",
        f"seed={RANDOM_STATE}",
    ]
)
STUDY_NAME = "escalation_hpo_" + hashlib.sha1(_fp_payload.encode()).hexdigest()[:12]
print(f"study: {STUDY_NAME}")

mlflow_study = MlflowSparkStudy(
    study_name=STUDY_NAME,
    storage=mlflow_storage,
    sampler=TPESampler(seed=RANDOM_STATE),
    pruner=MedianPruner(),
)
print(f"Launching {N_TRIALS} trials (n_jobs={N_JOBS})…")
mlflow_study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)
print("Study complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Log the champion run (consumed by 05_train_register)

# COMMAND ----------

import json
import tempfile

best_params = dict(mlflow_study.best_params)
best_metric = -float(mlflow_study.best_value)  # undo minimization negation
print(f"Best {HPO_OBJECTIVE} (val): {best_metric:.4f}")
print("Best params:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

with mlflow.start_run(run_name="escalation_hpo_champion") as run:
    mlflow.set_tags(
        {
            "project": "ticket_escalation",
            "stage": "hpo",
            "gold_table": GOLD_TABLE,
            "study_name": STUDY_NAME,
        }
    )
    mlflow.log_params(
        {
            "n_trials": N_TRIALS,
            "n_folds": N_FOLDS,
            "hpo_objective": HPO_OBJECTIVE,
            "scale_pos_weight": spw,
            "random_state": RANDOM_STATE,
            **{f"best_{k}": v for k, v in best_params.items()},
        }
    )
    mlflow.log_metrics({f"best_val_{HPO_OBJECTIVE}": best_metric})

    # Persist best params + the FIXED spw so 05 refits deterministically.
    _artifact_dir = tempfile.mkdtemp()
    _bp_path = os.path.join(_artifact_dir, "best_params.json")
    with open(_bp_path, "w") as fh:
        json.dump(
            {
                "best_params": best_params,
                "xgb_fixed": XGB_FIXED,
                "scale_pos_weight": spw,
                "hpo_objective": HPO_OBJECTIVE,
                "best_val_metric": best_metric,
                "n_folds": N_FOLDS,
                "feature_names": list(ef.FEATURE_NAMES),
            },
            fh,
            indent=2,
        )
    mlflow.log_artifact(_bp_path)
    HPO_CHAMPION_RUN_ID = run.info.run_id
    print(f"\nHPO champion run_id: {HPO_CHAMPION_RUN_ID}")
    print("Pass this run_id to 05_train_register via the `hpo_champion_run_id` widget.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Surface the run_id for downstream tasks

# COMMAND ----------

# Return the run_id so a Jobs task can wire it into 05 via {{tasks.hpo.values...}} or a manual paste.
dbutils.jobs.taskValues.set(key="hpo_champion_run_id", value=HPO_CHAMPION_RUN_ID) if hasattr(
    dbutils, "jobs"
) else None
dbutils.notebook.exit(HPO_CHAMPION_RUN_ID)
