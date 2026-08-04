# Databricks notebook source
# MAGIC %md
# MAGIC # Scapia — Ticket Escalation — 05 · Train + Register (UC pyfunc, champion gate)
# MAGIC
# MAGIC Consumes the HPO champion params, trains the final XGBoost escalation model, **freezes ALL preprocessing
# MAGIC into a self-contained `mlflow.pyfunc`** (K-Fold target-encoding maps, median imputer values, GL ordinal
# MAGIC maps, global mean, F2 threshold, feature order), registers it to Unity Catalog, and applies the
# MAGIC **fail-closed champion/challenger alias gate** (ported from `fop/04`).
# MAGIC
# MAGIC ## Discipline
# MAGIC * Rebuild the **exact same** 80/20 stratified split (seed 42) as HPO. From train carve **VAL = 15%** for
# MAGIC   the F2 threshold + honest reporting. Preprocessing is fit on TRAIN only.
# MAGIC * **F2 threshold** chosen on VAL over 0.1–0.9 step 0.05 (recall-first: missing an escalation is worse
# MAGIC   than a false alarm). Frozen into the pyfunc.
# MAGIC * **TEST touched once** for the honest headline metrics (ROC-AUC, PR-AUC, F2 @ threshold).
# MAGIC * **Fail-closed gate**: a new version takes `@champion` ONLY if no champion exists. Overwriting a live
# MAGIC   champion requires the explicit `force_champion_override` widget (human decision) — otherwise it lands
# MAGIC   as `@challenger`.
# MAGIC
# MAGIC ## The frozen pyfunc
# MAGIC `predict()` runs the whole chain — median-impute → GL ordinal → target-encode → XGBoost booster — so
# MAGIC preprocessing always travels with the model. Output columns: `escalation_probability`,
# MAGIC `predicted_flag` (prob ≥ frozen F2 threshold). All frozen state ships via `artifacts`, and the transform
# MAGIC is the SAME `escalation_features.build_model_matrix` the training path uses.
# MAGIC
# MAGIC > Config via widgets — see `escalation/MANIFESTO.md`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Config widgets

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Unity Catalog catalog (CONFIGURE(catalog)).")
dbutils.widgets.text("schema", "ml_escalation", "Unity Catalog schema (CONFIGURE(schema)).")
dbutils.widgets.text(
    "mlflow_experiment_path", "/Shared/scapia_ticket_escalation",
    "MLflow experiment path (CONFIGURE(mlflow_experiment_path)).",
)
dbutils.widgets.text(
    "hpo_champion_run_id", "",
    "REQUIRED: MLflow run_id of the escalation_hpo_champion run (source of best_params).",
)
dbutils.widgets.text("n_folds", "5", "K-Fold target-encoding folds (CONFIGURE(n_folds)).")
dbutils.widgets.dropdown(
    "alias_mode", "auto", ["auto", "champion", "challenger", "none"],
    "auto = @champion if none exists else @challenger. 'champion' forces promotion (gated below).",
)
dbutils.widgets.dropdown(
    "force_champion_override", "no", ["no", "yes"],
    "Human override: 'yes' allows moving @champion off a live version. Default 'no' = fail closed.",
)

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
EXPERIMENT_PATH = dbutils.widgets.get("mlflow_experiment_path").strip()
HPO_CHAMPION_RUN_ID = dbutils.widgets.get("hpo_champion_run_id").strip()
N_FOLDS = int(dbutils.widgets.get("n_folds"))
ALIAS_MODE = dbutils.widgets.get("alias_mode").strip()
FORCE_CHAMPION_OVERRIDE = dbutils.widgets.get("force_champion_override").strip() == "yes"

GOLD_TABLE = f"{CATALOG}.{SCHEMA}.ticket_features"
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.ticket_escalation_model"
RANDOM_STATE = 42

if not HPO_CHAMPION_RUN_ID:
    raise ValueError("`hpo_champion_run_id` is REQUIRED — run 04_hpo and paste its run_id.")

print(f"gold table       : {GOLD_TABLE}")
print(f"registered model : {REGISTERED_MODEL_NAME}")
print(f"hpo champion run : {HPO_CHAMPION_RUN_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports + experiment + shared feature module

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
import pickle
import tempfile

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

import escalation_features as ef

mlflow.set_experiment(EXPERIMENT_PATH)
print("Experiment set. Features:", ef.FEATURE_NAMES)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load HPO champion params

# COMMAND ----------

_client = MlflowClient()
_local = mlflow.artifacts.download_artifacts(
    run_id=HPO_CHAMPION_RUN_ID, artifact_path="best_params.json"
)
with open(_local) as fh:
    _hpo = json.load(fh)

BEST_PARAMS = _hpo["best_params"]
XGB_FIXED = _hpo.get("xgb_fixed", {"eval_metric": "logloss", "seed": RANDOM_STATE, "device": "cpu"})
SPW = float(_hpo["scale_pos_weight"])
N_ESTIMATORS = int(BEST_PARAMS.pop("n_estimators")) if "n_estimators" in BEST_PARAMS else int(
    _hpo.get("best_n_estimators", 500)
)
print(f"best_params: {BEST_PARAMS}")
print(f"n_estimators: {N_ESTIMATORS}  |  scale_pos_weight: {SPW:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load gold + reproduce the HPO split exactly

# COMMAND ----------

from sklearn.model_selection import train_test_split

pdf = spark.table(GOLD_TABLE).toPandas()
X_all = pdf[ef.FEATURE_NAMES].copy()
y_all = pdf[ef.LABEL_COL].astype(int).copy()

# Identical split to HPO (same seed / stratify) so train/test never mix across the two notebooks.
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=RANDOM_STATE, stratify=y_all
)
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
)
print(f"fit: {len(X_fit):,}  val: {len(X_val):,}  test: {len(X_test):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Fit FROZEN preprocessing on TRAIN, encode all splits

# COMMAND ----------

# Frozen medians from the FULL train (fit+val) so the served imputer matches what the final model trains on.
MEDIANS = ef.compute_medians(X_train, ef.NUMERIC_FEATURES)


def _encode(X_ref_tr, y_ref_tr, X_target, encode_maps=None, global_mean=None):
    """Median-impute + GL-ordinal + target-encode a split. When encode_maps is None, FIT the target encoder
    on (X_ref_tr, y_ref_tr) and return the maps; otherwise apply the provided (frozen) maps."""
    out = ef.apply_medians(X_target, MEDIANS, ef.NUMERIC_FEATURES)
    for c in ef.CATEGORICAL_GL_FEATURES:
        out[c] = ef.encode_gl_ordinal(X_target[c])
    if encode_maps is None:
        # Fit target encoding on the reference train; return maps (train encoded out-of-fold via helper).
        ref = ef.apply_medians(X_ref_tr, MEDIANS, ef.NUMERIC_FEATURES)
        for c in ef.CATEGORICAL_GL_FEATURES:
            ref[c] = ef.encode_gl_ordinal(X_ref_tr[c])
        ref_enc, out_enc, encode_maps, global_mean = ef.kfold_target_encode(
            ref, y_ref_tr, out, ef.CATEGORICAL_HIGH_CARD_FEATURES,
            n_splits=N_FOLDS, smoothing=10.0, random_state=RANDOM_STATE,
        )
        return out_enc[ef.FEATURE_NAMES].astype("float32"), encode_maps, global_mean, ref_enc
    for c in ef.CATEGORICAL_HIGH_CARD_FEATURES:
        out[c] = ef.apply_target_encode(X_target[c], encode_maps.get(c, {}), global_mean)
    return out[ef.FEATURE_NAMES].astype("float32")


# Fit target encoder on the FULL train (fit+val together) → the maps frozen into the model.
X_train_mat, ENCODE_MAPS, GLOBAL_MEAN, _ = _encode(X_train, y_train, X_train)
X_val_mat = _encode(X_train, y_train, X_val, ENCODE_MAPS, GLOBAL_MEAN)
X_test_mat = _encode(X_train, y_train, X_test, ENCODE_MAPS, GLOBAL_MEAN)
print(f"encoded train: {X_train_mat.shape}  val: {X_val_mat.shape}  test: {X_test_mat.shape}")
print(f"global_mean (target-encode fallback): {GLOBAL_MEAN:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Fit the final booster (on TRAIN) + choose F2 threshold on VAL

# COMMAND ----------

CHAMP_PARAMS = {**XGB_FIXED, "objective": "binary:logistic", "tree_method": "hist", "scale_pos_weight": SPW}
CHAMP_PARAMS.update(BEST_PARAMS)

# Train on fit portion, pick threshold on VAL (VAL never trains the booster).
X_fit_mat = _encode(X_train, y_train, X_fit, ENCODE_MAPS, GLOBAL_MEAN)
dfit = xgb.DMatrix(X_fit_mat.to_numpy(dtype="float32"), label=y_fit.to_numpy(dtype="int32"), feature_names=list(ef.FEATURE_NAMES))
probe = xgb.train(CHAMP_PARAMS, dfit, num_boost_round=N_ESTIMATORS, verbose_eval=False)

dval = xgb.DMatrix(X_val_mat.to_numpy(dtype="float32"), feature_names=list(ef.FEATURE_NAMES))
val_proba = probe.predict(dval)
F2_THRESHOLD, val_f2 = ef.best_f2_threshold(y_val.to_numpy(), val_proba)  # 0.1..0.9 step 0.05
from sklearn.metrics import roc_auc_score

val_roc_auc = float(roc_auc_score(y_val.to_numpy(), val_proba))
print(f"VAL F2 threshold: {F2_THRESHOLD:.3f}  (val F2={val_f2:.4f}, val ROC-AUC={val_roc_auc:.4f})")

# Final booster refit on the FULL train (fit+val) for the frozen n_estimators.
dtrain = xgb.DMatrix(X_train_mat.to_numpy(dtype="float32"), label=y_train.to_numpy(dtype="int32"), feature_names=list(ef.FEATURE_NAMES))
champion = xgb.train(CHAMP_PARAMS, dtrain, num_boost_round=N_ESTIMATORS, verbose_eval=False)
print(f"Champion refit on full train ({len(y_train):,} rows).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Honest TEST evaluation (touched once)

# COMMAND ----------

from sklearn.metrics import average_precision_score, fbeta_score

dtest = xgb.DMatrix(X_test_mat.to_numpy(dtype="float32"), feature_names=list(ef.FEATURE_NAMES))
test_proba = champion.predict(dtest)
_y_test = y_test.to_numpy()

test_base_rate = float(_y_test.mean())
test_roc_auc = float(roc_auc_score(_y_test, test_proba))
test_pr_auc = float(average_precision_score(_y_test, test_proba))
test_f2 = float(fbeta_score(_y_test, (test_proba >= F2_THRESHOLD).astype(int), beta=2, zero_division=0))
print("=== Honest TEST metrics ===")
print(f"base rate : {test_base_rate:.3%}  ({int(_y_test.sum())}/{len(_y_test)})")
print(f"ROC-AUC   : {test_roc_auc:.4f}")
print(f"PR-AUC    : {test_pr_auc:.4f}")
print(f"F2 @ {F2_THRESHOLD:.2f}: {test_f2:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Assemble the frozen pyfunc
# MAGIC All state travels via `artifacts`: `booster` (XGBoost) + `preprocessing` (JSON: medians, encode_maps,
# MAGIC global_mean, ordinal maps, feature order, F2 threshold). `predict()` reuses
# MAGIC `escalation_features.build_model_matrix` — the identical transform used above.

# COMMAND ----------


class TicketEscalationModel(mlflow.pyfunc.PythonModel):
    """Self-contained escalation model: frozen preprocessing -> XGBoost booster.

    All state travels via `artifacts`:
      * ``preprocessing`` — JSON: numeric medians, GL ordinal map, high-card target-encode maps, global_mean,
        feature_names (frozen order), n_estimators, f2_threshold.
      * ``booster``       — XGBoost booster (Booster.save_model).

    predict() output columns:
      * ``escalation_probability`` — P(escalate).
      * ``predicted_flag``         — 1 when probability >= frozen F2 threshold, else 0.
    """

    def load_context(self, context):
        import json

        import xgboost as xgb

        with open(context.artifacts["preprocessing"]) as fh:
            self._pp = json.load(fh)
        self._booster = xgb.Booster()
        self._booster.load_model(context.artifacts["booster"])

    def predict(self, context, model_input, params=None):
        import numpy as np
        import pandas as pd
        import xgboost as xgb

        # Shared, unit-tested frozen transform (import here so it travels via code_paths at serve time).
        import escalation_features as _ef

        pdf_in = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        X = _ef.build_model_matrix(
            pdf_in,
            self._pp["medians"],
            self._pp["encode_maps"],
            float(self._pp["global_mean"]),
            feature_names=self._pp["feature_names"],
        )
        dm = xgb.DMatrix(X.to_numpy(dtype="float32"), feature_names=list(X.columns))
        proba = self._booster.predict(dm)
        thr = float(self._pp["f2_threshold"])
        flag = (proba >= thr).astype("int64")
        return pd.DataFrame(
            {"escalation_probability": np.asarray(proba, dtype="float64"), "predicted_flag": flag},
            index=X.index,
        )


# Frozen preprocessing state.
PREPROCESSING = {
    "numeric_features": list(ef.NUMERIC_FEATURES),
    "categorical_gl_features": list(ef.CATEGORICAL_GL_FEATURES),
    "categorical_high_card_features": list(ef.CATEGORICAL_HIGH_CARD_FEATURES),
    "feature_names": list(ef.FEATURE_NAMES),
    "medians": MEDIANS,
    "gl_ordinal_map": ef.GL_ORDINAL_MAP,
    "encode_maps": ENCODE_MAPS,
    "global_mean": float(GLOBAL_MEAN),
    "n_estimators": int(N_ESTIMATORS),
    "f2_threshold": float(F2_THRESHOLD),
    "scale_pos_weight": float(SPW),
}

_art_dir = tempfile.mkdtemp()
_booster_path = os.path.join(_art_dir, "booster.json")
champion.save_model(_booster_path)
_pp_path = os.path.join(_art_dir, "preprocessing.json")
with open(_pp_path, "w") as fh:
    json.dump(PREPROCESSING, fh, indent=2)
MODEL_ARTIFACTS = {"booster": _booster_path, "preprocessing": _pp_path}

# Ship the shared feature module INTO the model so predict() can import it at serve time.
_CODE_PATH = os.path.join(sys.path[0], "escalation_features.py") if sys.path and os.path.exists(
    os.path.join(sys.path[0], "escalation_features.py")
) else None
CODE_PATHS = [_CODE_PATH] if _CODE_PATH else None
print(f"artifacts: {list(MODEL_ARTIFACTS)}  |  code_paths: {CODE_PATHS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Signature + input example + pip pins

# COMMAND ----------

# input_example = RAW gold feature columns (the model does all encoding internally).
input_example = X_test[ef.FEATURE_NAMES].head(5).reset_index(drop=True).copy()
for c in ef.NUMERIC_FEATURES:
    input_example[c] = input_example[c].astype("float64")

# Local reproduction of predict() output for a faithful signature.
_ex_X = ef.build_model_matrix(input_example, MEDIANS, ENCODE_MAPS, GLOBAL_MEAN)
_ex_proba = champion.predict(xgb.DMatrix(_ex_X.to_numpy(dtype="float32"), feature_names=list(_ex_X.columns)))
output_example = pd.DataFrame(
    {
        "escalation_probability": np.asarray(_ex_proba, dtype="float64"),
        "predicted_flag": (_ex_proba >= F2_THRESHOLD).astype("int64"),
    }
)
signature = infer_signature(input_example, output_example)

import cloudpickle
import sklearn

PIP_REQUIREMENTS = [
    f"mlflow=={mlflow.__version__}",
    f"xgboost=={xgb.__version__}",
    f"scikit-learn=={sklearn.__version__}",
    f"pandas=={pd.__version__}",
    f"numpy=={np.__version__}",
    f"cloudpickle=={cloudpickle.__version__}",
]
print("signature:", signature)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Log + register to Unity Catalog

# COMMAND ----------

import time

mlflow.set_registry_uri("databricks-uc")
_uc_client = MlflowClient(registry_uri="databricks-uc")


def _resolve_registered_version(client, model_name, run_id, max_attempts=6, base_delay=0.5):
    """Resolve the just-registered UC version by EXACT run_id match ONLY — never 'latest' (a concurrent
    registration could otherwise alias an unrelated version). Poll with bounded backoff; RAISE if not found."""

    def _matches(mv):
        if getattr(mv, "run_id", None) == run_id:
            return True
        src = getattr(mv, "source", "") or ""
        return f"/{run_id}/" in src or src.endswith(f"/{run_id}")

    last_err = None
    for attempt in range(max_attempts):
        try:
            mvs = client.search_model_versions(f"name='{model_name}'")
            exact = [mv for mv in mvs if _matches(mv)]
            if exact:
                return str(sorted(exact, key=lambda m: int(m.version))[-1].version)
        except Exception as exc:
            last_err = exc
        if attempt < max_attempts - 1:
            time.sleep(min(base_delay * (2 ** attempt), 8.0))
    raise RuntimeError(
        f"Could not resolve a registered version of {model_name} for run {run_id} after {max_attempts} "
        f"attempts. Refusing to fall back to 'latest'. Last error: {last_err}."
    )


with mlflow.start_run(run_name="escalation_model_registration") as run:
    mlflow.set_tags(
        {
            "project": "ticket_escalation",
            "stage": "model_registration",
            "gold_table": GOLD_TABLE,
            "hpo_champion_run_id": HPO_CHAMPION_RUN_ID,
        }
    )
    mlflow.log_params(
        {
            "n_estimators": N_ESTIMATORS,
            "scale_pos_weight": SPW,
            "n_features": len(ef.FEATURE_NAMES),
            "n_folds": N_FOLDS,
            "f2_threshold": F2_THRESHOLD,
            **{f"best_{k}": v for k, v in BEST_PARAMS.items()},
        }
    )
    mlflow.log_metrics(
        {
            "val_roc_auc": val_roc_auc,
            "val_f2_threshold": F2_THRESHOLD,
            "val_f2_at_threshold": val_f2,
            "test_base_rate": test_base_rate,
            "test_roc_auc": test_roc_auc,
            "test_pr_auc": test_pr_auc,
            "test_f2_at_val_threshold": test_f2,
        }
    )
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=TicketEscalationModel(),
        artifacts=MODEL_ARTIFACTS,
        code_paths=CODE_PATHS,
        signature=signature,
        input_example=input_example,
        registered_model_name=REGISTERED_MODEL_NAME,
        pip_requirements=PIP_REQUIREMENTS,
    )
    run_id = run.info.run_id
    _model_uri = f"runs:/{run_id}/model"
    print(f"Logged + registered pyfunc in run {run_id}")

registered_version = _resolve_registered_version(_uc_client, REGISTERED_MODEL_NAME, run_id)
print(f"Registered {REGISTERED_MODEL_NAME} version {registered_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Round-trip check (best-effort, never fails the run)

# COMMAND ----------

try:
    _loaded = mlflow.pyfunc.load_model(_model_uri)
    _rt = _loaded.predict(input_example)
    assert list(_rt.columns) == ["escalation_probability", "predicted_flag"], _rt.columns
    print(f"Round-trip OK — columns {list(_rt.columns)} for {_rt.shape[0]} rows.")
except Exception as _exc:
    print(f"WARNING: round-trip load/predict did not complete: {_exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Fail-closed champion/challenger gate
# MAGIC A new version takes `@champion` ONLY when the model has no champion yet. Overwriting a live champion is a
# MAGIC human decision — it REQUIRES `force_champion_override=yes` AND `alias_mode=champion`; otherwise the new
# MAGIC version lands as `@challenger`. The gate fails CLOSED: only the precise "alias not found" error is read as
# MAGIC "no champion"; any other registry error re-raises (never blindly overwrites).

# COMMAND ----------

from mlflow.exceptions import RestException

_ALIAS_NOT_FOUND_CODE = "RESOURCE_DOES_NOT_EXIST"


def _current_alias_version(client, model_name, alias):
    try:
        return client.get_model_version_by_alias(model_name, alias).version
    except RestException as exc:
        if getattr(exc, "error_code", None) == _ALIAS_NOT_FOUND_CODE:
            return None  # alias genuinely absent -> no current champion
        raise  # auth / throttle / internal -> fail LOUD, never overwrite blind


existing_champion = _current_alias_version(_uc_client, REGISTERED_MODEL_NAME, "champion")

if ALIAS_MODE == "none":
    chosen_alias = None
elif ALIAS_MODE == "auto":
    chosen_alias = "champion" if existing_champion is None else "challenger"
else:
    chosen_alias = ALIAS_MODE  # explicit 'champion' or 'challenger'

# HARD gate: promoting over a live champion requires the human override widget.
if (
    chosen_alias == "champion"
    and existing_champion is not None
    and str(existing_champion) != str(registered_version)
):
    if not FORCE_CHAMPION_OVERRIDE:
        raise RuntimeError(
            f"Refusing to move @champion from live version v{existing_champion} to v{registered_version}. "
            f"Promoting over a live champion is a human-gated decision. This version is registered and "
            f"available as a challenger candidate; compare it on the same split, and if it wins re-run with "
            f"alias_mode=champion AND force_champion_override=yes, or run explicitly:\n"
            f"    MlflowClient(registry_uri='databricks-uc').set_registered_model_alias("
            f"'{REGISTERED_MODEL_NAME}', 'champion', '{registered_version}')"
        )
    print(f"force_champion_override=yes — moving @champion v{existing_champion} -> v{registered_version}.")

if chosen_alias is None:
    print(f"alias_mode=none -> v{registered_version} registered without an alias.")
elif chosen_alias == "champion" and existing_champion is not None and str(existing_champion) == str(
    registered_version
):
    print(f"@champion already points to v{registered_version} — idempotent no-op.")
else:
    _uc_client.set_registered_model_alias(REGISTERED_MODEL_NAME, chosen_alias, registered_version)
    print(
        f"Set @{chosen_alias} -> {REGISTERED_MODEL_NAME} v{registered_version} "
        f"(existing champion before this run: {existing_champion})."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Exit

# COMMAND ----------

dbutils.notebook.exit(
    json.dumps(
        {
            "registered_model": REGISTERED_MODEL_NAME,
            "version": registered_version,
            "alias": chosen_alias,
            "f2_threshold": F2_THRESHOLD,
            "test_roc_auc": test_roc_auc,
        }
    )
)
