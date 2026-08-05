# Databricks notebook source
# MAGIC %md
# MAGIC # Scapia — Ticket Escalation — 01 · Bronze Ingest
# MAGIC
# MAGIC Ensures the bronze `greylabs_raw` table exists and reports its freshness. In the **live** Scapia setup
# MAGIC Greylabs writes to `greylabs_raw` directly every ~5 minutes, so this notebook is normally a **no-op
# MAGIC verifier** — the training job lists it as *"skip if already running live"*.
# MAGIC
# MAGIC This notebook exists to give the pipeline a single, explicit bronze contract point and to support two
# MAGIC bootstrap modes for workspaces where Greylabs is **not yet** wired to write directly:
# MAGIC
# MAGIC * `verify` (default) — assert the table exists and print row/ticket counts + latest `DateTime`. Never
# MAGIC   writes. Use this in the live pipeline.
# MAGIC * `load_file` — one-off backfill: read a file (`CONFIGURE(bronze_source_path)`) exported from Greylabs and
# MAGIC   append it into `greylabs_raw` with the raw column names preserved. Use only for initial history load.
# MAGIC
# MAGIC > Config via widgets — see `escalation/MANIFESTO.md`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Config widgets

# COMMAND ----------

# CONFIGURE(catalog) / CONFIGURE(schema)
dbutils.widgets.text("catalog", "main", "Unity Catalog catalog (CONFIGURE(catalog)).")
dbutils.widgets.text("schema", "ml_escalation", "Unity Catalog schema (CONFIGURE(schema)).")
dbutils.widgets.dropdown(
    "ingest_mode", "verify", ["verify", "load_file"],
    "verify = assert table + report freshness (live default). load_file = one-off backfill from a file.",
)
# CONFIGURE(bronze_source_path) — ONLY used in load_file mode. Path to a Greylabs export (csv/parquet/delta).
dbutils.widgets.text(
    "bronze_source_path", "",
    "load_file ONLY: path to a Greylabs export to backfill (CONFIGURE(bronze_source_path)).",
)
dbutils.widgets.dropdown(
    "bronze_source_format", "csv", ["csv", "parquet", "delta", "json"],
    "load_file ONLY: format of the source file at bronze_source_path.",
)

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
INGEST_MODE = dbutils.widgets.get("ingest_mode").strip()
BRONZE_SOURCE_PATH = dbutils.widgets.get("bronze_source_path").strip()
BRONZE_SOURCE_FORMAT = dbutils.widgets.get("bronze_source_format").strip()

RAW_TABLE = f"{CATALOG}.{SCHEMA}.greylabs_raw"
print(f"bronze table : {RAW_TABLE}")
print(f"ingest_mode  : {INGEST_MODE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Assert the bronze table exists
# MAGIC `00_setup_tables` creates it. Fail fast with a clear message if it is missing.

# COMMAND ----------

if not spark.catalog.tableExists(RAW_TABLE):
    raise RuntimeError(
        f"Bronze table {RAW_TABLE} does not exist. Run 00_setup_tables first (or point the catalog/schema "
        f"widgets at the workspace where Greylabs writes). FAILING FAST."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. `load_file` mode — one-off backfill (optional)
# MAGIC Appends a Greylabs export into `greylabs_raw`, keeping only columns that match the raw schema so a stray
# MAGIC export column can never corrupt the table. Skipped entirely in `verify` mode.

# COMMAND ----------

if INGEST_MODE == "load_file":
    if not BRONZE_SOURCE_PATH:
        raise ValueError("ingest_mode=load_file requires `bronze_source_path` (CONFIGURE(bronze_source_path)).")

    reader = spark.read.format(BRONZE_SOURCE_FORMAT)
    if BRONZE_SOURCE_FORMAT == "csv":
        reader = reader.option("header", "true").option("multiLine", "true").option("escape", '"')
    src = reader.load(BRONZE_SOURCE_PATH)

    from pyspark.sql import functions as F

    # Align the source to the FULL bronze schema (item 14): present columns are cast to string; absent columns
    # are materialized as typed NULLs. Selecting the full target column list (not just present ones) means the
    # appended DataFrame's schema matches the table exactly, so `append` with mergeSchema disabled succeeds —
    # this is what actually makes the "missing source columns become NULL" behavior hold.
    target_cols = [f.name for f in spark.table(RAW_TABLE).schema.fields]
    present = [c for c in target_cols if c in src.columns]
    missing = [c for c in target_cols if c not in src.columns]
    if not present:
        raise ValueError(
            f"Source file has no columns matching the bronze schema. Source cols: {src.columns}. "
            f"Expected some of: {target_cols}. FAILING FAST."
        )
    if missing:
        print(f"WARNING: source is missing {len(missing)} bronze columns (written as NULL): {missing}")

    select_exprs = [
        F.col(f"`{c}`").cast("string").alias(c) if c in src.columns
        else F.lit(None).cast("string").alias(c)
        for c in target_cols  # full schema, in table column order
    ]
    src_aligned = src.select(select_exprs)
    n_in = src_aligned.count()
    src_aligned.write.mode("append").option("mergeSchema", "false").saveAsTable(RAW_TABLE)
    print(f"Appended {n_in:,} rows into {RAW_TABLE} from {BRONZE_SOURCE_PATH}")
else:
    print("verify mode — no write performed (Greylabs writes to greylabs_raw live).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Validation — freshness + volume

# COMMAND ----------

from pyspark.sql import functions as F

df = spark.table(RAW_TABLE)
n_rows = df.count()
n_tickets = df.select("`Ticket Id`").distinct().count()
print(f"greylabs_raw rows     : {n_rows:,}")
print(f"distinct Ticket Id    : {n_tickets:,}")

if n_rows > 0:
    # DateTime is raw text (DD-MM-YYYY HH:MM:SS). Parse just to report freshness.
    latest = (
        df.select(F.to_timestamp(F.col("`DateTime`"), "dd-MM-yyyy HH:mm:ss").alias("dt"))
        .agg(F.max("dt").alias("max_dt"), F.min("dt").alias("min_dt"))
        .collect()[0]
    )
    print(f"DateTime range        : {latest['min_dt']}  ->  {latest['max_dt']} (IST, parsed)")
else:
    print("greylabs_raw is EMPTY — expected on a fresh workspace before Greylabs starts writing.")
