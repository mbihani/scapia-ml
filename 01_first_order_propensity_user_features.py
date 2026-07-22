# Databricks notebook source
# MAGIC %md
# MAGIC # Scapia — First-Order-Propensity — Per-User Feature Store table
# MAGIC
# MAGIC Materializes Scapia's **First Order Propensity** per-user features into a Unity Catalog
# MAGIC **time-series Feature Engineering table** using the UC-native `FeatureEngineeringClient`
# MAGIC (`from databricks.feature_engineering import FeatureEngineeringClient`).
# MAGIC
# MAGIC ## What is stored
# MAGIC One row per `internal_user_id` **as of `feature_ts`** (the cutoff / as-of date). Columns are the
# MAGIC per-user behavioural features from the reference notebook: card-transaction counts, coin balance,
# MAGIC per-vertical search counts (flight / hotel / bus / train) over 7 / 15 / 30-day windows plus recency,
# MAGIC app-open engagement, lounge usage, onboarding date and activation latency.
# MAGIC
# MAGIC ## Raw-vs-encoded rule
# MAGIC The table stores **raw values only** — raw counts, balances, recency-in-days, timestamps. It does
# MAGIC **not** bucket, one-hot, or otherwise encode anything. All bucketing / dummy-encoding stays in the
# MAGIC *downstream model pipeline*. Storing raw values removes the train/serve-skew the audit flagged, where
# MAGIC encoders (buckets, dummies) were re-derived on every run and could differ between training and serving.
# MAGIC
# MAGIC ## Spine-vs-features split
# MAGIC The **label** (`output` — made a first qualifying order in the performance window), the
# MAGIC **eligibility / population filter** (carded before cutoff, never-ordered as of cutoff, engaged) and any
# MAGIC other task-specific logic live in a per-model **spine query**, *not* here. This feature table is
# MAGIC task-agnostic and shared across models. See the final "Spine + training set" cell for the pattern.
# MAGIC
# MAGIC ## Feature selection happens downstream
# MAGIC The store holds the full raw **superset** of features. Each model chooses its own subset at training
# MAGIC time via `FeatureLookup(feature_names=[...])` on the training set — never by trimming this table.
# MAGIC
# MAGIC ## Point-in-time correctness
# MAGIC This is a **time-series** feature table (`primary_keys=['internal_user_id', 'feature_ts']`,
# MAGIC `timeseries_columns=['feature_ts']`). `create_training_set` then performs an *as-of* join keyed on the
# MAGIC spine's `feature_ts`, structurally preventing the target-leakage the audit flagged.
# MAGIC
# MAGIC ## TODO — `segment` is intentionally OMITTED (audit finding #1)
# MAGIC The reference query joined `simple.crud.user_segment_mapping`. That table is a **CRUD current-state
# MAGIC snapshot with no effective-date / SCD2 column**, so it cannot be joined as-of `feature_ts` — doing so
# MAGIC leaks the user's *future* (post-cutoff) segment into a historical feature row. `segment` is therefore
# MAGIC excluded from this table. **To add it back safely:** source segment from an effective-dated / SCD2
# MAGIC history (with a valid-from / valid-to), and LEFT JOIN it as-of `feature_ts`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Dependencies
# MAGIC `databricks-feature-engineering` ships with the Databricks ML Runtime. The install below just pins a
# MAGIC recent version so `timeseries_columns` and the point-in-time `create_training_set` join are available.
# MAGIC Safe to skip on a current ML Runtime.

# COMMAND ----------

# MAGIC %pip install -U databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. CONFIG
# MAGIC Everything you would routinely change lives in this one block.

# COMMAND ----------

# ---------------------------------------------------------------------------
# CONFIG — edit here
# ---------------------------------------------------------------------------

# Target Unity Catalog feature table (catalog / schema / table kept separate so
# they are trivial to repoint at another catalog or environment).
CATALOG = "mlops_data_science"
SCHEMA = "features"
TABLE = "first_order_propensity_user_features"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"

# Entity key and feature-timestamp column names.
ENTITY_KEY = "internal_user_id"   # real entity-key column from the reference notebook
FEATURE_TS = "feature_ts"          # point-in-time key, derived from the as-of / cutoff date

# Primary key for the time-series feature table. A Databricks time-series table
# REQUIRES the timeseries column to be part of the primary key, so the PK is the
# (entity, feature_ts) composite — this is also what lets multiple per-user
# snapshots (one per as-of date) coexist and merge correctly. Declared once here
# so the create_table call never hardcodes it independently.
PRIMARY_KEYS = [ENTITY_KEY, FEATURE_TS]

# --- Source Unity Catalog tables (read-only inputs) -------------------------
ONBOARDED_USERS_FACT = "simple.crud.onboarded_users_fact"          # card issuance -> onboarding date
DATEWISE_CARD_USER_AGG = "simple.custom.datewise_card_user_agg_v2" # daily per-user card/search agg
BUS_SEARCH_REQUESTS = "rds_main.scapiadb.bus_search_requests"       # bus search events
TRAIN_SEARCH_REQUESTS = "rds_main.scapiadb.train_search_requests"   # train search events
PERCEPT_EVENT_AGG = "simple.crud.percept_event_agg"                 # app engagement events
LOUNGE_KEYPASS_USAGE = "rds_reward.scapiadb.lounge_keypass_usage"   # airport-lounge redemptions

# Orders table is NOT a feature source. It is read ONLY to derive a sensible
# default as-of date (mirrors the reference cutoff). The label and eligibility
# filter that use it belong in the per-model SPINE, not in this feature table.
ORDERS_TABLE = "rds_main.scapiadb.orders"

# IST offset. Timestamp columns in the RDS-sourced tables are stored in UTC; the
# reference notebook anchors the cutoff in IST and subtracts this offset to
# compare against the UTC values. (Date-keyed tables need no offset.)
IST_OFFSET_MINUTES = 330

# Used ONLY to compute the default as-of date when the widget is left blank:
# default as-of = (latest IST qualifying-order time) - PERFORMANCE_DAYS.
PERFORMANCE_DAYS = 90
QUALIFYING_STATUSES = ["COMPLETE", "CANCELLED"]
QUALIFYING_PRODUCT_CATEGORIES = [
    "FLIGHT", "BUS", "TRAIN", "HOTEL_STAY",
    "ECOMMERCE", "EXPERIENCE", "VISA", "HOLIDAY",
]

# ---------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Parameters — as-of date
# MAGIC The as-of date (a.k.a. the cutoff) is the point-in-time at which every feature is computed. It becomes
# MAGIC `feature_ts`. Parameterising it lets the job be **scheduled** (run for "today") and **backfilled**
# MAGIC (re-run for any historical date). Leave the widget blank to auto-derive the reference cutoff.

# COMMAND ----------

from datetime import timedelta

dbutils.widgets.text(
    "as_of_date",
    "",
    "As-of / cutoff (YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS', IST). Blank = auto-derive reference cutoff.",
)


def _sql_in_list(values):
    """Render a Python list as a SQL IN-list of single-quoted literals."""
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def resolve_as_of_ts(widget_value: str) -> str:
    """Return the as-of timestamp string used to anchor every feature.

    If the widget is set, use it verbatim (a bare date is read as IST midnight).
    Otherwise reproduce the reference cutoff: the latest qualifying-order time
    (in IST) minus PERFORMANCE_DAYS. Orders is touched ONLY for this default.

    RECOMMENDATION: always pass an explicit `as_of_date` for training runs. The
    auto-derived default depends on the global latest qualifying-order timestamp,
    which drifts as new orders arrive — so leaving it blank makes the resulting
    training set non-reproducible across runs.
    """
    widget_value = (widget_value or "").strip()
    if widget_value:
        return widget_value

    probe = spark.sql(
        f"""
        SELECT CAST(MAX(CAST(created_at AS timestamp)) + INTERVAL {IST_OFFSET_MINUTES} MINUTE AS timestamp) AS max_date
        FROM {ORDERS_TABLE}
        WHERE status IN ({_sql_in_list(QUALIFYING_STATUSES)})
          AND product_category IN ({_sql_in_list(QUALIFYING_PRODUCT_CATEGORIES)})
        """
    ).first()
    cutoff = probe["max_date"] - timedelta(days=PERFORMANCE_DAYS)
    return cutoff.strftime("%Y-%m-%d %H:%M:%S")


as_of_ts = resolve_as_of_ts(dbutils.widgets.get("as_of_date"))
print(f"Target feature table : {FEATURE_TABLE}")
print(f"as_of / feature_ts   : {as_of_ts}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build the RAW feature DataFrame (Spark, read from UC)
# MAGIC All sources are read from Unity Catalog via Spark SQL — no `databricks-sql-connector`, no local pandas
# MAGIC full-table pull, no PATs/secrets (ambient workspace auth only). Every window is anchored at the single
# MAGIC as-of timestamp in CTE `d`, so the whole table is one consistent point-in-time snapshot.
# MAGIC
# MAGIC Note on nulls (kept **raw**, on purpose):
# MAGIC * **counts** (searches / txns / app-opens) coalesce to `0` — genuinely zero activity.
# MAGIC * **recency-in-days** and **activation_days** stay `NULL` when the event never occurred — absence is
# MAGIC   `NULL`, not a magic sentinel. The downstream model owns imputation (the reference pipeline maps
# MAGIC   `NULL` -> the `never` / `inactive` bucket, identical to how it treats its old `9999` sentinel).
# MAGIC
# MAGIC **NULL contract (read before consuming):** for the recency (`days_since_last_*`) and `activation_days`
# MAGIC columns, `NULL` means *"never / not activated"* and is intended to map to the v7 `never` / `inactive`
# MAGIC buckets in the model pipeline's `bucket_recency` / `bucket_ad` functions. Any downstream consumer that
# MAGIC expects a literal `9999` sentinel (rather than `NULL`) must impute it itself — the store never writes it.

# COMMAND ----------


def build_feature_sql(as_of_ts: str, ist: int) -> str:
    """Feature-only SQL (label / eligibility deliberately excluded). Anchored at `as_of_ts`."""
    return f"""
WITH d AS (
    -- single source of truth for the point-in-time anchor
    SELECT timestamp '{as_of_ts}' AS as_of
),

-- Entity universe: every carded user, with their onboarding (first card-issue) date.
ob AS (
    SELECT user_id AS internal_user_id,
           CAST(MIN(card_issue_time) AS timestamp) AS onboarding_completion_date
    FROM {ONBOARDED_USERS_FACT}
    WHERE card_issue_time IS NOT NULL
    GROUP BY 1
),

-- First day of any travel/card activity, as of the cutoff (feeds activation_days).
tactv AS (
    SELECT internal_user_id,
           MIN(CAST(date AS date)) FILTER (
               WHERE flights_orders + stays_orders + card_coins_earned_instances > 0
           ) AS first_tactv_date
    FROM {DATEWISE_CARD_USER_AGG}
    WHERE CAST(date AS date) <= date((SELECT as_of FROM d))
    GROUP BY 1
),

-- Card-transaction & coin features (lifetime up to cutoff + last-30-day window).
duca AS (
    SELECT u.internal_user_id,
           SUM(u.card_coins_earned_instances) AS card_txn_lifetime,
           SUM(u.coins_earned)                AS coins_bal_overall,
           SUM(CASE WHEN date_diff(DAY, CAST(u.date AS date), date((SELECT as_of FROM d))) BETWEEN 0 AND 30
                    THEN u.card_coins_earned_instances ELSE 0 END) AS t_30_txn
    FROM {DATEWISE_CARD_USER_AGG} u
    WHERE CAST(u.date AS date) <= date((SELECT as_of FROM d))
    GROUP BY 1
),

-- Card transactions in the user's FIRST 30 days after onboarding.
since_ob AS (
    SELECT u.internal_user_id,
           SUM(CASE WHEN date_diff(DAY, date(ob.onboarding_completion_date), CAST(u.date AS date)) BETWEEN 0 AND 30
                    THEN u.card_coins_earned_instances ELSE 0 END) AS first_30_txn
    FROM {DATEWISE_CARD_USER_AGG} u
    JOIN ob ON ob.internal_user_id = u.internal_user_id
    WHERE CAST(u.date AS date) <= date((SELECT as_of FROM d))
      AND date_diff(DAY, date(ob.onboarding_completion_date), CAST(u.date AS date)) BETWEEN 0 AND 30
    GROUP BY 1
),

-- Flight search intensity (7/15/30d) + recency. HAVING keeps LEFT-JOIN semantics.
flight_s AS (
    SELECT u.internal_user_id,
           SUM(CASE WHEN date_diff(DAY, CAST(u.date AS date), date((SELECT as_of FROM d))) BETWEEN 1 AND 7  THEN u.flight_searches ELSE 0 END) AS flight_searches_7d,
           SUM(CASE WHEN date_diff(DAY, CAST(u.date AS date), date((SELECT as_of FROM d))) BETWEEN 1 AND 15 THEN u.flight_searches ELSE 0 END) AS flight_searches_15d,
           SUM(CASE WHEN date_diff(DAY, CAST(u.date AS date), date((SELECT as_of FROM d))) BETWEEN 1 AND 30 THEN u.flight_searches ELSE 0 END) AS flight_searches_30d,
           date_diff(DAY, MAX(CASE WHEN u.flight_searches > 0 THEN CAST(u.date AS date) END), date((SELECT as_of FROM d))) AS days_since_last_flight_search
    FROM {DATEWISE_CARD_USER_AGG} u
    WHERE u.internal_user_id IS NOT NULL
      AND CAST(u.date AS date) < date((SELECT as_of FROM d))
    GROUP BY 1
    HAVING SUM(u.flight_searches) > 0
),

-- Hotel search intensity (7/15/30d) + recency.
hotel_s AS (
    SELECT u.internal_user_id,
           SUM(CASE WHEN date_diff(DAY, CAST(u.date AS date), date((SELECT as_of FROM d))) BETWEEN 1 AND 7  THEN u.hotel_searches ELSE 0 END) AS hotel_searches_7d,
           SUM(CASE WHEN date_diff(DAY, CAST(u.date AS date), date((SELECT as_of FROM d))) BETWEEN 1 AND 15 THEN u.hotel_searches ELSE 0 END) AS hotel_searches_15d,
           SUM(CASE WHEN date_diff(DAY, CAST(u.date AS date), date((SELECT as_of FROM d))) BETWEEN 1 AND 30 THEN u.hotel_searches ELSE 0 END) AS hotel_searches_30d,
           date_diff(DAY, MAX(CASE WHEN u.hotel_searches > 0 THEN CAST(u.date AS date) END), date((SELECT as_of FROM d))) AS days_since_last_hotel_search
    FROM {DATEWISE_CARD_USER_AGG} u
    WHERE u.internal_user_id IS NOT NULL
      AND CAST(u.date AS date) < date((SELECT as_of FROM d))
    GROUP BY 1
    HAVING SUM(u.hotel_searches) > 0
),

-- Bus search intensity (7/15/30d) + recency. created_at is UTC -> shift cutoff by IST offset.
bus_s AS (
    SELECT bsr.internal_user_id,
           SUM(CASE WHEN date_diff(DAY, CAST(bsr.created_at AS timestamp), (SELECT as_of FROM d) - INTERVAL {ist} MINUTE) BETWEEN 1 AND 7  THEN 1 ELSE 0 END) AS bus_searches_7d,
           SUM(CASE WHEN date_diff(DAY, CAST(bsr.created_at AS timestamp), (SELECT as_of FROM d) - INTERVAL {ist} MINUTE) BETWEEN 1 AND 15 THEN 1 ELSE 0 END) AS bus_searches_15d,
           SUM(CASE WHEN date_diff(DAY, CAST(bsr.created_at AS timestamp), (SELECT as_of FROM d) - INTERVAL {ist} MINUTE) BETWEEN 1 AND 30 THEN 1 ELSE 0 END) AS bus_searches_30d,
           date_diff(DAY, CAST(MAX(CAST(bsr.created_at AS timestamp)) AS timestamp), (SELECT as_of FROM d) - INTERVAL {ist} MINUTE) AS days_since_last_bus_search
    FROM {BUS_SEARCH_REQUESTS} bsr
    WHERE bsr.internal_user_id IS NOT NULL
      AND CAST(bsr.created_at AS timestamp) < (SELECT as_of FROM d) - INTERVAL {ist} MINUTE
    GROUP BY 1
),

-- Train search intensity (7/15/30d) + recency. Source key is user_id.
train_s AS (
    SELECT tsr.user_id AS internal_user_id,
           SUM(CASE WHEN date_diff(DAY, CAST(tsr.created_at AS timestamp), (SELECT as_of FROM d) - INTERVAL {ist} MINUTE) BETWEEN 1 AND 7  THEN 1 ELSE 0 END) AS train_searches_7d,
           SUM(CASE WHEN date_diff(DAY, CAST(tsr.created_at AS timestamp), (SELECT as_of FROM d) - INTERVAL {ist} MINUTE) BETWEEN 1 AND 15 THEN 1 ELSE 0 END) AS train_searches_15d,
           SUM(CASE WHEN date_diff(DAY, CAST(tsr.created_at AS timestamp), (SELECT as_of FROM d) - INTERVAL {ist} MINUTE) BETWEEN 1 AND 30 THEN 1 ELSE 0 END) AS train_searches_30d,
           date_diff(DAY, MAX(CAST(tsr.created_at AS timestamp)), (SELECT as_of FROM d) - INTERVAL {ist} MINUTE) AS days_since_last_train_search
    FROM {TRAIN_SEARCH_REQUESTS} tsr
    WHERE tsr.user_id IS NOT NULL
      AND CAST(tsr.created_at AS timestamp) < (SELECT as_of FROM d) - INTERVAL {ist} MINUTE
    GROUP BY 1
),

-- App-open engagement (proxy: bottom_nav_tab_home_clicked): 7/30d + lifetime + recency.
app_opens AS (
    SELECT pea.user_id AS internal_user_id,
           SUM(CASE WHEN date_diff(DAY, pea.event_date, date((SELECT as_of FROM d))) BETWEEN 1 AND 7  THEN pea.event_count ELSE 0 END) AS app_opens_7d,
           SUM(CASE WHEN date_diff(DAY, pea.event_date, date((SELECT as_of FROM d))) BETWEEN 1 AND 30 THEN pea.event_count ELSE 0 END) AS app_opens_30d,
           SUM(pea.event_count) AS app_opens_lifetime,
           date_diff(DAY, MAX(pea.event_date), date((SELECT as_of FROM d))) AS days_since_last_app_open
    FROM {PERCEPT_EVENT_AGG} pea
    WHERE pea.event_name = 'bottom_nav_tab_home_clicked'
      AND pea.user_id IS NOT NULL
      AND pea.event_date < date((SELECT as_of FROM d))
    GROUP BY 1
),

-- Lounge ever redeemed as of the cutoff.
lounge AS (
    SELECT DISTINCT user_id
    FROM {LOUNGE_KEYPASS_USAGE}
    WHERE status = 'REDEEMED'
      AND CAST(created_at AS timestamp) < (SELECT as_of FROM d) - INTERVAL {ist} MINUTE
)

SELECT
    ob.internal_user_id,
    (SELECT as_of FROM d)                                    AS feature_ts,        -- point-in-time key
    ob.onboarding_completion_date,                                                 -- raw timestamp
    CASE WHEN tactv.first_tactv_date < date((SELECT as_of FROM d))
         THEN date_diff(DAY, ob.onboarding_completion_date, CAST(tactv.first_tactv_date AS timestamp))
    END                                                      AS activation_days,   -- NULL if not activated
    COALESCE(duca.t_30_txn, 0)                               AS t_30_txn,
    COALESCE(since_ob.first_30_txn, 0)                       AS first_30_txn,
    COALESCE(duca.coins_bal_overall, 0)                      AS coins_bal_overall,
    COALESCE(duca.card_txn_lifetime, 0)                      AS card_txn_lifetime,  -- raw superset (unused by v7 model)
    COALESCE(flight_s.flight_searches_7d, 0)                 AS flight_searches_7d,
    COALESCE(flight_s.flight_searches_15d, 0)                AS flight_searches_15d,
    COALESCE(flight_s.flight_searches_30d, 0)                AS flight_searches_30d,
    flight_s.days_since_last_flight_search                   AS days_since_last_flight_search,  -- NULL = never
    COALESCE(hotel_s.hotel_searches_7d, 0)                   AS hotel_searches_7d,
    COALESCE(hotel_s.hotel_searches_15d, 0)                  AS hotel_searches_15d,
    COALESCE(hotel_s.hotel_searches_30d, 0)                  AS hotel_searches_30d,
    hotel_s.days_since_last_hotel_search                     AS days_since_last_hotel_search,
    COALESCE(bus_s.bus_searches_7d, 0)                       AS bus_searches_7d,
    COALESCE(bus_s.bus_searches_15d, 0)                      AS bus_searches_15d,
    COALESCE(bus_s.bus_searches_30d, 0)                      AS bus_searches_30d,
    bus_s.days_since_last_bus_search                         AS days_since_last_bus_search,
    COALESCE(train_s.train_searches_7d, 0)                   AS train_searches_7d,
    COALESCE(train_s.train_searches_15d, 0)                  AS train_searches_15d,
    COALESCE(train_s.train_searches_30d, 0)                  AS train_searches_30d,
    train_s.days_since_last_train_search                     AS days_since_last_train_search,
    COALESCE(app_opens.app_opens_7d, 0)                      AS app_opens_7d,
    COALESCE(app_opens.app_opens_30d, 0)                     AS app_opens_30d,
    COALESCE(app_opens.app_opens_lifetime, 0)                AS app_opens_lifetime,
    app_opens.days_since_last_app_open                       AS days_since_last_app_open,
    CASE WHEN lounge.user_id IS NOT NULL THEN 1 ELSE 0 END   AS lounge_used
    -- `segment` intentionally OMITTED — see the TODO in the header (no as-of predicate on the source).
FROM ob
LEFT JOIN duca      ON duca.internal_user_id      = ob.internal_user_id
LEFT JOIN since_ob  ON since_ob.internal_user_id  = ob.internal_user_id
LEFT JOIN tactv     ON tactv.internal_user_id     = ob.internal_user_id
LEFT JOIN flight_s  ON flight_s.internal_user_id  = ob.internal_user_id
LEFT JOIN hotel_s   ON hotel_s.internal_user_id   = ob.internal_user_id
LEFT JOIN bus_s     ON bus_s.internal_user_id     = ob.internal_user_id
LEFT JOIN train_s   ON train_s.internal_user_id   = ob.internal_user_id
LEFT JOIN app_opens ON app_opens.internal_user_id = ob.internal_user_id
LEFT JOIN lounge    ON lounge.user_id             = ob.internal_user_id
-- Temporal-existence guard only (NOT task eligibility): the card must exist at feature_ts,
-- otherwise onboarding_completion_date would be in the future relative to the snapshot.
WHERE ob.onboarding_completion_date <= (SELECT as_of FROM d)
"""


features_df = spark.sql(build_feature_sql(as_of_ts, IST_OFFSET_MINUTES))
print(f"Feature columns ({len(features_df.columns)}): {features_df.columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create-if-not-exists + merge write (idempotent)
# MAGIC First run creates the time-series table (schema + PK + timeseries column). Every run — first or
# MAGIC subsequent — writes in **merge** mode, so re-running for the same `feature_ts` upserts (no duplicates)
# MAGIC and backfilling a new `feature_ts` simply appends that day's snapshot.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Schema must pre-exist (catalog creation is typically an admin action, so it is not attempted here).
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")

try:
    table_exists = spark.catalog.tableExists(FEATURE_TABLE)
except Exception:
    table_exists = False

if not table_exists:
    fe.create_table(
        name=FEATURE_TABLE,
        primary_keys=PRIMARY_KEYS,          # composite PK = [internal_user_id, feature_ts] (see CONFIG)
        timeseries_columns=[FEATURE_TS],    # feature_ts is BOTH part of the PK and the time dimension
        schema=features_df.schema,          # create empty with the right schema; data lands via write_table
        description=(
            "Scapia First-Order-Propensity per-user RAW features (counts / balances / recency / timestamps). "
            "Time-series table keyed on (internal_user_id, feature_ts) for point-in-time lookups. "
            "Label, eligibility filter and all encoding are OUT of scope here — they live in the per-model "
            "spine / training pipeline. `segment` omitted (no as-of source). See the notebook header."
        ),
        tags={
            "project": "first_order_propensity",
            "team": "data_science",
            "layer": "feature_store",
            "value_type": "raw",
        },
    )
    print(f"Created time-series feature table: {FEATURE_TABLE}")
else:
    print(f"Feature table already exists: {FEATURE_TABLE} — writing in merge mode")

fe.write_table(name=FEATURE_TABLE, df=features_df, mode="merge")
print(f"Merged {features_df.count():,} rows for feature_ts = {as_of_ts}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Spine + training set (illustrative — belongs in the per-model pipeline, not here)
# MAGIC
# MAGIC This feature table is **task-agnostic**. A model consumes it by building a **spine** that carries the
# MAGIC entity key, the `feature_ts` to look up at, and the label — then calling `create_training_set`, which
# MAGIC performs the point-in-time (as-of) join. Everything task-specific (label definition, first-order
# MAGIC eligibility, engagement filter, feature *selection*) lives here, not in the shared table.
# MAGIC
# MAGIC ### 5a. Spine query — label + eligibility (SQL, `{cutoff}` = as-of, `{performance_end}` = cutoff + 90d)
# MAGIC ```sql
# MAGIC WITH d AS (
# MAGIC     SELECT timestamp '{cutoff}'          AS cutoff,
# MAGIC            timestamp '{performance_end}' AS performance_end
# MAGIC ),
# MAGIC ob AS (
# MAGIC     SELECT user_id AS internal_user_id, CAST(MIN(card_issue_time) AS timestamp) AS onboarding_completion_date
# MAGIC     FROM simple.crud.onboarded_users_fact
# MAGIC     WHERE card_issue_time IS NOT NULL
# MAGIC     GROUP BY 1
# MAGIC ),
# MAGIC -- LABEL: a first qualifying order in (cutoff, performance_end]
# MAGIC perf AS (
# MAGIC     SELECT DISTINCT user_id AS internal_user_id
# MAGIC     FROM rds_main.scapiadb.orders
# MAGIC     WHERE status IN ('COMPLETE','CANCELLED')
# MAGIC       AND product_category IN ('FLIGHT','BUS','TRAIN','HOTEL_STAY','ECOMMERCE','EXPERIENCE','VISA','HOLIDAY')
# MAGIC       AND user_id IS NOT NULL
# MAGIC       AND CAST(created_at AS timestamp) >  (SELECT cutoff FROM d)          - INTERVAL 330 MINUTE
# MAGIC       AND CAST(created_at AS timestamp) <= (SELECT performance_end FROM d) - INTERVAL 330 MINUTE
# MAGIC ),
# MAGIC -- ELIGIBILITY: users who ALREADY ordered before the cutoff are excluded (this is FIRST-order propensity)
# MAGIC pre AS (
# MAGIC     SELECT DISTINCT user_id AS internal_user_id
# MAGIC     FROM rds_main.scapiadb.orders
# MAGIC     WHERE status IN ('COMPLETE','CANCELLED')
# MAGIC       AND product_category IN ('FLIGHT','BUS','TRAIN','HOTEL_STAY','ECOMMERCE','EXPERIENCE','VISA','HOLIDAY')
# MAGIC       AND user_id IS NOT NULL
# MAGIC       AND CAST(created_at AS timestamp) <= (SELECT cutoff FROM d) - INTERVAL 330 MINUTE
# MAGIC )
# MAGIC SELECT
# MAGIC     ob.internal_user_id,
# MAGIC     (SELECT cutoff FROM d)                                     AS feature_ts,   -- must match the store's timeseries key
# MAGIC     CASE WHEN perf.internal_user_id IS NOT NULL THEN 1 ELSE 0 END AS output    -- the label
# MAGIC FROM ob
# MAGIC LEFT JOIN perf ON perf.internal_user_id = ob.internal_user_id
# MAGIC LEFT JOIN pre  ON pre.internal_user_id  = ob.internal_user_id
# MAGIC WHERE ob.onboarding_completion_date < (SELECT cutoff FROM d)  -- carded before cutoff
# MAGIC   AND pre.internal_user_id IS NULL                            -- never ordered as of cutoff
# MAGIC   -- Plus the engagement filter from the reference notebook (active in last 90d OR any recent search)
# MAGIC   -- also belongs HERE, evaluated against the feature store via feature_lookups if desired.
# MAGIC ```
# MAGIC
# MAGIC ### 5b. Point-in-time training set (Python stub)
# MAGIC ```python
# MAGIC from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
# MAGIC
# MAGIC fe = FeatureEngineeringClient()
# MAGIC spine_df = spark.sql(SPINE_SQL)   # columns: internal_user_id, feature_ts, output
# MAGIC
# MAGIC training_set = fe.create_training_set(
# MAGIC     df=spine_df,
# MAGIC     feature_lookups=[
# MAGIC         FeatureLookup(
# MAGIC             table_name="mlops_data_science.features.first_order_propensity_user_features",
# MAGIC             lookup_key="internal_user_id",     # entity key
# MAGIC             timestamp_lookup_key="feature_ts", # as-of join -> no leakage from the future
# MAGIC             # This explicit list is exactly where each model plugs in its OWN selected
# MAGIC             # subset from the feature-selection step (do NOT pass None = every feature).
# MAGIC             # These are real raw columns produced by this notebook:
# MAGIC             feature_names=[
# MAGIC                 "t_30_txn",                       # card-txn count, last 30d
# MAGIC                 "coins_bal_overall",              # coin balance
# MAGIC                 "flight_searches_30d",            # flight-search count, last 30d
# MAGIC                 "days_since_last_flight_search",  # flight-search recency (days)
# MAGIC                 "app_opens_30d",                  # app-engagement count, last 30d
# MAGIC                 "lounge_used",                    # lounge-ever-used flag
# MAGIC             ],
# MAGIC         )
# MAGIC     ],
# MAGIC     label="output",
# MAGIC     exclude_columns=["feature_ts"],            # keep the join key out of the model matrix
# MAGIC )
# MAGIC
# MAGIC training_df = training_set.load_df()  # point-in-time-correct; now bucket / one-hot HERE, then train.
# MAGIC ```
