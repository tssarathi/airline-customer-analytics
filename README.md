# Airline Customer Analytics

End-to-end customer analytics and (future) ML pipeline using airline loyalty data.

## Status (WIP)

This project is **actively in progress**. Today it includes:
- Local notebooks for **pre-processing** and **feature exploration** (including **RFM-style** analysis)
- A Python entrypoint that can run an **AWS S3 + Athena** bootstrap ETL (upload raw CSV → write Parquet to S3)

Planned next: Glue/Athena modeling, SageMaker training/inference, and QuickSight dashboards.

## Tech stack

- **Python**: Pandas, PyArrow, Matplotlib
- **AWS (current)**: S3, Athena
- **AWS (planned)**: Glue, SageMaker, QuickSight

## What’s implemented

- **Local data pre-processing (notebook)**: creates Parquet datasets in `data/processed/`
  - `data/processed/clh.parquet` (Customer Loyalty History)
  - `data/processed/cfa.parquet` (Customer Flight Activity)
- **Feature exploration (notebook)**: EDA + customer segmentation style analysis (RFM)
- **AWS ETL bootstrap (`python -m src`)**:
  - Runs Athena SQL in `infra/sql/00_create_database.sql` (creates database `airline_analytics`)
  - Uploads raw CSVs from `data/raw/` to `s3://$S3_BUCKET/$RAW_PREFIX/`
  - Converts CSV → Parquet in S3 and writes to `s3://$S3_BUCKET/$PROCESSED_PREFIX/`
  - Uses idempotent checks (won’t overwrite if the S3 object already exists)

## Project structure

- `src/` – core application code (ETL + AWS helpers)
- `infra/` – infra scaffolding (SQL for Athena/Glue)
- `notebooks/` – EDA and pre-processing notebooks
- `data/` – raw inputs and processed outputs (local)
- `reports/` – dashboards/charts (planned; folder exists)

## Data

- **Raw inputs**: `data/raw/`
  - `Customer Loyalty History.csv`
  - `Customer Flight Activity.csv`
  - `Calendar.csv`
  - `Airline Loyalty Data Dictionary.csv`
- **Local outputs**: `data/processed/`
  - `clh.parquet`
  - `cfa.parquet`

## Quickstart (local notebooks)

Prereqs: **Python 3.10+**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open and run:
- `notebooks/pre-processing.ipynb` (writes `data/processed/*.parquet`)
- `notebooks/feature-exploration.ipynb` (reads `data/processed/*.parquet`)

## Quickstart (AWS S3 + Athena pipeline)

Prereqs:
- AWS credentials configured (e.g. via `aws configure`, or environment variables)
- An S3 bucket you control (for raw data, processed data, and Athena query results)

### 1) Configure environment variables

Create a `.env` file (loaded by `python-dotenv`) with:

```bash
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name
RAW_PREFIX=raw
PROCESSED_PREFIX=processed
GLUE_DB=airline_analytics
ATHENA_RESULTS_PREFIX=athena-results
```

Notes:
- `GLUE_DB` is present for future Glue/Athena expansion; the current entrypoint creates `airline_analytics` via SQL.
- Athena query results are written to `s3://$S3_BUCKET/$ATHENA_RESULTS_PREFIX/` (this location must be writable).

### 2) Run the pipeline

```bash
python -m src
```

What it does (today):
- Creates the Athena database
- Uploads the raw CSVs to S3
- Writes Parquet versions back to S3

## Known limitations / WIP notes

- `infra/sql/01_create_airline_loyalty_raw.sql` is **not wired into the entrypoint yet** (table creation is a WIP).
- `infra/sql/01_create_airline_loyalty_raw.sql` currently uses a **hyphenated database name** and a **hard-coded S3 location**; this will be parameterized and aligned with the `S3_BUCKET` / `RAW_PREFIX` config.
- Running `python -m src` will incur AWS usage and may fail without correct IAM permissions (S3 read/write + Athena query execution).

## Roadmap (short)

- Parameterize infra SQL and align with env config (`S3_BUCKET`, prefixes, DB name)
- Add Athena external tables for raw + Parquet datasets and basic curated views
- Add Glue integration (crawler / catalog) as needed
- Build first customer segmentation + retention models (SageMaker)
- Publish QuickSight dashboards for KPIs and segments