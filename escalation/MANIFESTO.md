# Escalation Pipeline — Deployment Manifesto

This module ships with **`CONFIGURE(<slug>)` markers** wherever a value must (or may) be set for a specific
Scapia workspace. Every marker below is greppable:

```bash
grep -rn "CONFIGURE(" escalation/
```

Set the **Required** markers before the first run. Review the **Customize** markers before production. The
**Optional** markers are for advanced / classic-compute setups. Notebooks read these via `dbutils.widgets`; the
job JSONs carry them as literal `CONFIGURE(...)` placeholders you replace at deploy time.

> Notebook defaults are sensible for a dev workspace (`main.ml_escalation`). The job JSONs deliberately keep
> the placeholders un-substituted so a deploy tool (or a human) fills them per environment.

---

## Required — must set before running

| Marker | Where | Default | What it is |
|---|---|---|---|
| `CONFIGURE(catalog)` | every notebook widget; both job JSONs | `main` | Unity Catalog **catalog** for all escalation tables + the registered model. |
| `CONFIGURE(schema)` | every notebook widget; both job JSONs | `ml_escalation` | Unity Catalog **schema** for all escalation tables + the registered model. |
| `CONFIGURE(mlflow_experiment_path)` | `04_hpo`, `05_train_register`; training job | `/Shared/scapia_ticket_escalation` | MLflow experiment path for HPO trials + the registration run. |
| `CONFIGURE(workspace_base_path)` | both job JSONs (notebook paths) | *(none)* | Workspace path where this repo's `escalation/` folder is imported, e.g. `/Workspace/Repos/<user>/scapia-ml/feature_store` or `/Workspace/Users/<you>/scapia-ml/feature_store`. Notebook paths are `CONFIGURE(workspace_base_path)/escalation/notebooks/<name>`. |

---

## Customize — review before production

| Marker | Where | Default | What it controls |
|---|---|---|---|
| `CONFIGURE(active_ticket_window_hours)` | `03_gold_features` (inference), `06_batch_inference`; inference job | `8` | How recently a ticket's last call / features must have been computed for it to be scored. Shorter = only very active tickets. |
| `CONFIGURE(risk_tier_high)` | `06_batch_inference`; inference job | `0.7` | Probability ≥ this ⇒ `risk_tier = 'high'`. |
| `CONFIGURE(risk_tier_medium)` | `06_batch_inference`; inference job | `0.4` | Probability ≥ this (and < high) ⇒ `risk_tier = 'medium'`; else `'low'`. |
| `CONFIGURE(hpo_n_trials)` | `04_hpo`; training job | `50` | Number of Optuna HPO trials. More trials = better tuning, longer runtime. |
| `CONFIGURE(n_folds)` | `04_hpo`, `05_train_register`; training job | `5` | K-Fold folds for target encoding of the high-cardinality categoricals. |

---

## Optional — advanced

| Marker | Where | Default | When to set |
|---|---|---|---|
| `CONFIGURE(cluster_id)` | both job JSONs (compute comments) | *(serverless)* | Set only to run on a **classic** cluster instead of serverless — add `existing_cluster_id: <id>` to each task. |
| `CONFIGURE(inference_lookback_hours)` | `02_silver_clean` (incremental); inference job | `8` | Incremental silver window: only re-clean calls newer than `now() − N h`. Keep ≥ `active_ticket_window_hours` so no active ticket's calls are missed. |
| `CONFIGURE(bronze_source_path)` | `01_bronze_ingest` (`ingest_mode=load_file` only) | *(empty)* | Path to a one-off Greylabs export to **backfill** into `greylabs_raw`. Unused in the live pipeline (Greylabs writes bronze directly); leave empty in `verify` mode. |

---

## Non-`CONFIGURE` knobs worth knowing (sensible defaults, rarely changed)

These are real widgets/params but are not tagged `CONFIGURE()` because the shipped default is almost always
right:

- `run_mode` (`02_silver_clean`): `full` (training job) vs `incremental` (inference job).
- `mode` (`03_gold_features`): `training` (labelled) vs `inference` (active, unlabelled).
- `ingest_mode` (`01_bronze_ingest`): `verify` (live default) vs `load_file` (backfill).
- `hpo_objective` (`04_hpo`): `pr_auc` (default) / `roc_auc` / `f2_max`.
- `alias_mode` + `force_champion_override` (`05_train_register`): the champion-promotion gate. Default
  `auto` + `no` = fail-closed (never overwrite a live champion without an explicit human override).
- Split seed `42`, target-encode `smoothing=10`, F2 threshold grid `0.1–0.9 step 0.05`, XGBoost fixed params
  (`eval_metric=logloss`, `seed=42`, `device=cpu`) — hard-coded for reproducibility; change in
  `escalation_features.py` / `04_hpo.py` only with intent.

---

## Tables created (in `CONFIGURE(catalog).CONFIGURE(schema)`)

| Table | Created by | Role |
|---|---|---|
| `greylabs_raw` | `00_setup_tables` | Bronze landing (Greylabs writes here every ~5 min). |
| `greylabs_calls_clean` | `02_silver_clean` | Typed, deduped per-call silver. |
| `ticket_features` | `03_gold_features` | Ticket-grain gold feature table (19 model features + metadata). |
| `ticket_status` | `00_setup_tables` | Placeholder — ticket lifecycle status (graceful degrade if empty). |
| `ticket_escalation_predictions` | `00_setup_tables` (written by `06`) | Per-ticket risk scores, upserted every 5 min. |
| `ticket_escalation_feedback` | `00_setup_tables` | Phase-2 CX-app feedback loop. |
| `ticket_escalation_model` | `05_train_register` | Registered UC pyfunc (champion/challenger aliases). |
