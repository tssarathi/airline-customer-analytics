# Airline Customer Analytics

End-to-end customer analytics and ML pipeline using airline loyalty data: ETL (S3, Parquet, Athena), curated customer features with RFM-style segmentation, Athena tables and views, and SageMaker Processing for XGBoost churn modeling.

## Status

The pipeline is **implemented end-to-end** (see [Enabling the pipeline](#enabling-the-pipeline) below):

- **ETL**: Raw CSV → S3 → Parquet (processed) → curated customer features
- **Athena**: Database, `customer_features` and `churn_predictions` external tables, `customer_scored` view
- **ML**: SageMaker Processing job trains an XGBoost churn model and writes artifacts and predictions to S3

## Tech stack

- **Python**: Pandas, PyArrow, scikit-learn, XGBoost
- **AWS**: S3, Athena, SageMaker (Processing with sklearn container)

## What’s implemented

### Entry point: `python -m src`

The main pipeline (`src/main.py`) runs in order when **`FLAG` is set to `True`** in `src/main.py`:

1. **Create Athena database**Runs `infra/sql/00_create_database.sql` → database `airline_analytics`.
2. **Upload raw CSVs to S3**

   - `data/raw/Customer Loyalty History.csv` → `s3://$S3_BUCKET/$RAW_PREFIX/customer_loyalty_history.csv`
   - `data/raw/Customer Flight Activity.csv` → `s3://$S3_BUCKET/$RAW_PREFIX/customer_flight_activity.csv`
     Idempotent: skips upload if object already exists.
3. **Transform CSV → Parquet in S3**

   - Reads from `$RAW_PREFIX/`, writes to `$PROCESSED_PREFIX/`.
   - CLH: type casting via `cast_clh`; CFA: standardized columns only.
   - Outputs: `customer_loyalty_history.parquet`, `customer_flight_activity.parquet`.
     Idempotent: skips if output Parquet exists.
4. **Build customer features**

   - Joins processed CFA + CLH, computes RFM (recency, frequency, monetary), segments (Champions, Loyal, At Risk, Dormant, Potential), tenure, churn-related fields.
   - Writes `s3://$S3_BUCKET/$CURATED_PREFIX/customer_features/customer_features.parquet`.
     Idempotent: skips if output exists.
5. **Create Athena table (customer features)**

   - `01_create_customer_features_table.sql`: external table over the curated Parquet.
6. **Run SageMaker XGBoost processing job**

   - Input: `customer_features.parquet`.
   - Trains churn model (label: `is_cancelled` or recency ≥ threshold), outputs `priority_score = churn_prob * clv`.
   - Writes `model.pkl`, `metrics.json` under `artifacts/` and `predictions.parquet` under `predictions/`.
   - Output: `s3://$S3_BUCKET/$CURATED_PREFIX/churn/` (with `artifacts/` and `predictions/` subdirs).
7. **Create Athena churn table and customer_scored view**

   - `02_create_churn_pred_table.sql`: external table `churn_predictions` over `.../churn/predictions/`.
   - `03_create_customer_scored_table.sql`: view `customer_scored` joining `customer_features` and `churn_predictions` (adds `churn_score`).

### Enabling the pipeline

By default, **`FLAG` is `False`** in `src/main.py`, so `python -m src` does nothing. Set `FLAG = True` in `src/main.py` to run the full pipeline (Athena, S3, SageMaker).

### Local notebooks

- `notebooks/01_pre-processing.ipynb` – pre-processing, writes `data/processed/*.parquet` (e.g. `clh.parquet`, `cfa.parquet`)
- `notebooks/02_feature-exploration.ipynb` – EDA, RFM-style exploration
- `notebooks/03_churn_modelling.ipynb` – churn modeling experiments (e.g. XGBoost, LR); local artifacts under `artifacts/churn/`

## Project structure

```
├── src/                    # Application code
│   ├── main.py             # Pipeline entry point (gated by FLAG)
│   ├── config/             # Env-based config (S3, Athena, SageMaker)
│   ├── etl/                # CSV→Parquet, customer features
│   ├── model/              # XGBoost churn training (SageMaker script)
│   ├── scripts/            # run_xgb_job (SageMaker Processor)
│   └── utils/              # Athena, S3 helpers
├── infra/sql/              # Athena DDL
│   ├── 00_create_database.sql
│   ├── 01_create_customer_features_table.sql
│   ├── 02_create_churn_pred_table.sql
│   └── 03_create_customer_scored_table.sql   # view
├── notebooks/              # EDA and pre-processing
├── data/
│   ├── raw/                # Raw CSVs
│   ├── processed/          # Local Parquet (notebooks)
│   └── curated/            # Local customer_features (notebooks/pipeline)
├── artifacts/              # Local ML artifacts (e.g. churn/XGBoost, churn/LR)
└── requirements.txt
```

## Data

### Raw inputs (`data/raw/`)

- `Customer Loyalty History.csv`
- `Customer Flight Activity.csv`
- `Calendar.csv`, `Airline Loyalty Data Dictionary.csv`

### Local outputs

- `data/processed/`: `clh.parquet`, `cfa.parquet` (from notebooks)
- `data/curated/`: `customer_features.parquet` (also produced by pipeline to S3)

### S3 layout (pipeline)

- `$RAW_PREFIX/` – raw CSVs (`customer_loyalty_history.csv`, `customer_flight_activity.csv`)
- `$PROCESSED_PREFIX/` – Parquet (`customer_loyalty_history.parquet`, `customer_flight_activity.parquet`)
- `$CURATED_PREFIX/customer_features/` – customer features Parquet
- `$CURATED_PREFIX/churn/artifacts/` – model.pkl, metrics.json
- `$CURATED_PREFIX/churn/predictions/` – predictions.parquet
- `$ATHENA_RESULTS_PREFIX/` – Athena query results

## Quickstart

### Prerequisites

- **Python 3.10+**
- **AWS**: credentials (e.g. `aws configure` or env vars), S3 bucket, IAM permissions for S3, Athena, and SageMaker Processing.

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure `.env`

Create a `.env` file (loaded via `python-dotenv`):

```bash
AWS_REGION=us-east-1
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=your-bucket-name
RAW_PREFIX=raw
PROCESSED_PREFIX=processed
CURATED_PREFIX=curated
ATHENA_RESULTS_PREFIX=athena_results
GLUE_DB=airline_analytics

# SageMaker Processing job
ROLE_ARN=arn:aws:iam::ACCOUNT:role/your-sagemaker-role
PROCESSING_INSTANCE_TYPE=ml.t3.medium
```

- `AWS_REGION` / `AWS_DEFAULT_REGION`: used for Athena, S3, SageMaker.
- Athena results: `s3://$S3_BUCKET/$ATHENA_RESULTS_PREFIX/` must be writable.
- `ROLE_ARN`: IAM role for SageMaker Processing (S3 access to input/output, plus Processing permissions).

### 3. Run the pipeline

1. Set `FLAG = True` in `src/main.py`.
2. Run:

```bash
python -m src
```

This runs the full flow: Athena DB + customer_features table, S3 upload/transform/curate, SageMaker XGBoost job, then churn_predictions table and customer_scored view.

### 4. Run notebooks only (local)

Open and run:

- `notebooks/01_pre-processing.ipynb`
- `notebooks/02_feature-exploration.ipynb`
- `notebooks/03_churn_modelling.ipynb`

## Known limitations

- **Pipeline gate**: The main pipeline runs only when `FLAG = True` in `src/main.py`; otherwise `python -m src` is a no-op.
- **Infra SQL**: `01_create_customer_features_table.sql` and `02_create_churn_pred_table.sql` use **hard-coded** S3 locations (`s3://airline-customer-analysis/...`). They are not parameterized from `S3_BUCKET` / `CURATED_PREFIX`. Use that bucket/path or edit the SQL to match your setup.
- **Schema**: `customer_features` table schema includes `country`; the Python feature build does not currently output `country`. Align schema or add `country` to the feature output if needed.
- **Athena context**: `run_sql_file` does not set `QueryExecutionContext.database`; DDL uses fully qualified names (`airline_analytics.*`).
- **Costs & permissions**: `python -m src` uses S3, Athena, and SageMaker; ensure IAM allows it and be aware of AWS usage.

## Roadmap

- Parameterize infra SQL (bucket, prefixes) via env or templating.
- QuickSight (or similar) dashboards for KPIs and segments.
- Conversational self service agent.
