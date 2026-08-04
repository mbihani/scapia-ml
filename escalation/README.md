# Scapia — Ticket Escalation ML Pipeline

Predict whether a support **ticket will escalate** — a call flagged by Greylabs as
`Real Time Alert (Answer) = Yes` (customer threatened social media, mentioned RBI / Ombudsman / CEO) — so CX
supervisors can intervene **before** it happens. Active open tickets are scored every 5 minutes.

This module is a Databricks-native port of the reference notebook `greylabs_lte_v3.ipynb`: bronze → silver →
gold feature engineering, distributed Optuna HPO, a self-contained UC-registered `pyfunc`, and a 5-minute batch
scorer. It follows the same conventions as the sibling **First-Order-Propensity** pipeline in `../fop/`.

---

## Module map

```
escalation/
├── MANIFESTO.md                  # tiered CONFIGURE() deployment catalogue — read this first
├── README.md                     # this file
├── notebooks/
│   ├── escalation_features.py    # shared, unit-tested feature logic (imported by 03/05 + tests)
│   ├── 00_setup_tables.py        # create the 4 Delta tables (idempotent)
│   ├── 01_bronze_ingest.py       # verify bronze freshness (no-op live) / optional backfill
│   ├── 02_silver_clean.py        # typed, deduped per-call silver (full or incremental)
│   ├── 03_gold_features.py       # ticket-grain 20-feature gold table (training or inference)
│   ├── 04_hpo.py                 # distributed Optuna HPO (MlflowSparkStudy) -> champion params
│   ├── 05_train_register.py      # final fit, frozen pyfunc, UC register + fail-closed champion gate
│   └── 06_batch_inference.py     # 5-min scorer: @champion pyfunc, SHAP top-5, MERGE upsert
├── jobs/
│   ├── training_job.json         # 01->02->03->04->05 (Jobs API format)
│   └── inference_job.json        # 02->03->06, every 5 min (cron), 1 retry / 2-min backoff
└── tests/
    └── test_features.py          # pure-Python unit tests (no Spark) for the feature logic
```

---

## Data flow

```
Greylabs (every ~5 min)
        │  writes raw call-QA rows
        ▼
greylabs_raw ──02──▶ greylabs_calls_clean ──03──▶ ticket_features ──┬──04──▶ HPO champion params
 (bronze)            (silver, 1 row/call)         (gold, 1 row/ticket)│
                                                                     └──05──▶ ticket_escalation_model
                                                                              (UC pyfunc @champion)
                                                                                   │
ticket_features (inference) + ticket_status ──06 (every 5 min)───────────────────┘
        │  score @champion, SHAP top-5, risk tier
        ▼
ticket_escalation_predictions   (MERGE upsert on ticket_id)
```

---

## The 20 model features (ticket-grain, pre-escalation window only)

Only calls **before** the first `real_time_alert='yes'` call feed the features. Tickets whose *first* call
already escalated are excluded. Label `is_escalated = 1` if **any** call escalated.

- **Numeric (12)** — `num_calls_considered`, `time_diff_hours`, `calls_per_day`, `agent_speech_ratio`,
  `customer_speech_ratio`, `customer_sentiment_trend`, `total_call_duration`, `total_nonspeech_duration`,
  `agent_talk_duration`, `customer_talk_duration`, `weighted_average`, **`neg_sentiment_ratio`** (proportion of
  `customer_sentiment='bad'` calls — new feature agreed Aug 3 2026). Median-imputed with frozen medians.
- **GL worst-of (4)** — `gl_empathy`, `gl_agent_sentiment`, `gl_customer_sentiment`, `gl_repeat_calls`. The
  most-negative value across calls (`bad` always wins). Ordinal-encoded (`bad=0/neutral=1/good=2`, unseen=-1).
- **High-cardinality (4)** — `scapia_category`, `scapia_sub_category`, `qrc_category`, `qrc_sub_category`.
  Latest non-null value; **K-Fold target-encoded** at training (unseen → global mean at inference).

---

## Running it

### 1. One-time setup
Run `00_setup_tables` (creates the 4 Delta tables). If Greylabs is not yet writing `greylabs_raw` directly,
backfill history once with `01_bronze_ingest` (`ingest_mode=load_file`, `CONFIGURE(bronze_source_path)`).

### 2. Train
Deploy `jobs/training_job.json` (after substituting `CONFIGURE(...)`), or run the notebooks in order:
`02 (full) → 03 (training) → 04 → 05`. `05` registers the pyfunc and, on the **first** run, promotes it to
`@champion` (fail-closed thereafter — see below).

### 3. Score every 5 minutes
Deploy `jobs/inference_job.json`. It runs `02 (incremental) → 03 (inference) → 06` on a `0 */5 * * * ?` cron
and upserts `ticket_escalation_predictions`.

---

## Key design decisions

- **Pre-escalation slicing.** Features are built from calls strictly *before* the first escalation so the model
  learns leading indicators, not the escalation itself. First-call-escalation tickets are dropped (no
  pre-window). *(spec + reference)*
- **Worst-of aggregation.** GL sentiment/quality is summarised by its most-negative call (`bad < neutral <
  good`) — one bad call is the signal, not the average.
- **K-Fold target encoding — OOF for training, full maps for serving (no leakage).** High-cardinality
  categories (302 Scapia sub-categories, etc.) use smoothed target means (Bayesian α=10). The **booster trains
  on OUT-OF-FOLD encodings** (`kfold_oof_encode`) so a row's own label never enters its own feature. The
  **full-training maps** (`compute_target_encode_maps`) are what get **frozen into the pyfunc** for inference
  and are also used to encode held-out folds (val/test), whose labels are not in the maps. Unseen categories
  fall back to the global mean. This distinction is explicit in code and is guarded by a regression test that
  fails if training ever uses the full maps.
- **Deterministic, Spark-order-independent splits + slicing.** The train/val/test split is assigned **by
  ticket id via a stable hash** (`assign_split_by_ticket`), so 04_hpo and 05_train_register compute the
  identical partition regardless of Spark row order, and no ticket's rows straddle splits. The pre-escalation
  slice sorts by `(call_datetime, call_id)` so tied timestamps order reproducibly. All time math is on **UTC
  instants** (session tz pinned to UTC; IST kept only for display).
- **True per-call key.** `call_id = SHA256(transcript)` (as in the reference) is the silver **dedup key** — so
  two genuinely distinct calls sharing a timestamp are never collapsed — and the deterministic tie-break sort
  key. **Transcript handling:** the Greylabs feed carries a `Transcript` column (bronze), so silver
  **quarantines rows with a null/empty transcript** (logged with a count) *before* dropping the transcript
  text. If a future export lacks `Transcript`, repoint `CALL_ID_SOURCE` to another per-call unique column.
- **Single-source feature logic.** All ticket-grain maths lives in `escalation_features.py` and is invoked in
  Spark via `applyInPandas` (gold) and baked into the pyfunc (inference). The unit tests exercise the **exact**
  production code, not a copy.
- **Fail-closed champion gate + round-trip guard.** A newly trained version takes `@champion` **only** if none
  exists; otherwise it lands as `@challenger`. Moving `@champion` off a live version requires an explicit human
  override (`alias_mode=champion` + `force_champion_override=yes`). "No champion" is inferred **only** from a
  not-found error that positively names the alias (the model's existence is confirmed first); any other error
  re-raises. Before any alias is assigned, the registered pyfunc is **loaded back and must reproduce its own
  predictions** — a broken frozen-preprocessing package aborts registration rather than becoming champion.
  *(ported + hardened from `../fop/04`)*
- **F2 operating threshold.** Chosen on a **genuinely independent VAL holdout** (probe trained on the fit
  portion only, val encoded with fit-only maps) over 0.1–0.9 (step 0.05), maximizing F2 (β=2, favours recall —
  missing an escalation costs more than a false alarm). Frozen into the pyfunc as `predicted_flag`.
- **Sentiment magnitude.** The silver layer collapses raw `Good/Positive` into a single top bucket and
  `Bad/Negative/Profane` into `bad`; the model treats `good == positive` (rank 2). `neg_sentiment_ratio` adds
  an explicit magnitude signal for how *many* calls turned negative.
- **SHAP explanations — fail loud, never silent-empty.** `06` unwraps the raw XGBoost booster from the pyfunc
  and runs `TreeExplainer` to store the top-5 feature contributions per ticket as JSON. A SHAP failure does
  **not** emit empty arrays that masquerade as valid: `CONFIGURE(shap_failure_mode)` defaults to `fail` (the
  scheduled job raises), or `flag` to write `shap_status='FAILED'` so downstream can tell explanations are
  missing.

---

## Configuration
Every deploy-time value is a greppable `CONFIGURE(<slug>)` marker catalogued (tiered Required / Customize /
Optional) in **[MANIFESTO.md](./MANIFESTO.md)**. Defaults target a dev workspace at `main.ml_escalation`.

## Tests
```bash
pytest escalation/tests/test_features.py -v
```
Pure Python (pandas/numpy only) — no Spark, no Databricks runtime. Covers worst-of, pre-escalation slicing,
first-call exclusion, `neg_sentiment_ratio`, speech ratios, sentiment trend, divide-by-zero guards, QRC casing
normalization, plus the frozen encoders (target encoding, GL ordinal, model matrix) and F2 threshold selection.
**Regression tests** (from the cross-vendor review) additionally assert: OOF target encoding excludes each
row's own label (would fail if training used the full maps — real target leakage), pre-escalation slicing is
deterministic under tied timestamps (shuffled input → identical slice), and the by-ticket split is stable
across reorderings with no ticket straddling splits.

## Status
⚠️ **Not yet run live on a Scapia workspace.** Logic is unit-tested locally and every notebook is
syntax-validated, but the end-to-end Spark / MLflow / UC path has not been executed against Scapia's tables.
Fill the `CONFIGURE()` markers and run `00`→`05` on the target workspace to validate before production.
