# Airline Customer Analytics

End-to-end customer analytics and ML pipeline using airline loyalty data: ETL (S3, Parquet, Athena), curated customer features with RFM-style segmentation, Athena views, and SageMaker XGBoost churn modeling.

## Status

The pipeline is **implemented end-to-end**:
- **ETL**: Raw CSV → S3 → Parquet (processed) → curated customer features
- **Athena**: Database, `customer_features` external table, segmentation views
- **ML**: SageMaker Processing job trains an XGBoost churn model and writes artifacts to S3

## Tech stack

- **Python**: Pandas, PyArrow, scikit-learn, XGBoost, Matplotlib
- **AWS**: S3, Athena, SageMaker (Processing)

## What’s implemented

### Entry point: `python -m src`

The main pipeline (`src/main.py`) runs in order:

1. **Create Athena database**  
   Runs `infra/sql/00_create_database.sql` → `airline_analytics`.

2. **Upload raw CSVs to S3**  
   - `data/raw/Customer Loyalty History.csv` → `s3://$S3_BUCKET/$RAW_PREFIX/customer_loyalty_history.csv`  
   - `data/raw/Customer Flight Activity.csv` → `s3://$S3_BUCKET/$RAW_PREFIX/customer_flight_activity.csv`  
   Idempotent: skips upload if object already exists.

3. **Transform CSV → Parquet in S3**  
   - Reads from `$RAW_PREFIX/`, writes to `$PROCESSED_PREFIX/`.  
   - CLH: type casting via `cast_clh`; CFA: standardized columns only.  
   Idempotent: skips if output Parquet exists.

4. **Build customer features**  
   - Joins processed CFA + CLH, computes RFM (recency, frequency, monetary), segments (Champions, Loyal, At Risk, Dormant, Potential), tenure, churn-related fields.  
   - Writes `s3://$S3_BUCKET/$CURATED_PREFIX/customer_features/customer_features.parquet`.  
   Idempotent: skips if output exists.

5. **Create Athena table and views**  
   - `01_create_customer_features_table.sql`: external table over the curated Parquet.  
   - `02_segment_distribution.sql`: `vw_segment_counts`  
   - `03_at_high_risk.sql`: `vw_at_risk_high_value`  
   - `04_segment_by_province.sql`: `vw_segment_by_province`

6. **Run SageMaker XGBoost processing job**  
   - Input: `customer_features.parquet`.  
   - Trains churn model (label: cancelled or recency ≥ threshold), saves `model.pkl`, `metrics.json`, `predictions.parquet`.  
   - Output: `s3://$S3_BUCKET/$CURATED_PREFIX/artifacts/churn/`.

### Local notebooks

- `notebooks/01_pre-processing.ipynb` – pre-processing, writes `data/processed/*.parquet`
- `notebooks/02_feature-exploration.ipynb` – EDA, RFM-style exploration
- `notebooks/03_churn_modelling.ipynb` – churn modeling experiments

## Project structure

```
├── src/                 # Application code
│   ├── main.py          # Pipeline entry point
│   ├── config/          # Env-based config (S3, Athena, SageMaker)
│   ├── etl/             # CSV→Parquet, customer features
│   ├── model/           # XGBoost churn training (SageMaker script)
│   ├── scripts/         # run_xgb_job (SageMaker Processor)
│   └── utils/           # Athena, S3 helpers
├── infra/sql/           # Athena DDL (database, table, views)
├── notebooks/           # EDA and pre-processing
├── data/                # Raw inputs and local processed/curated outputs
│   ├── raw/
│   ├── processed/
│   └── curated/
├── artifacts/           # Local ML artifacts (e.g. churn/XGBoost, churn/LR)
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

- `$RAW_PREFIX/` – raw CSVs  
- `$PROCESSED_PREFIX/` – Parquet (CLH, CFA)  
- `$CURATED_PREFIX/customer_features/` – customer features Parquet  
- `$CURATED_PREFIX/artifacts/churn/` – model, metrics, predictions  
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

- `AWS_REGION` / `AWS_DEFAULT_REGION`: used for Athena, S3, SageMaker; `AWS_REGION` falls back to `AWS_DEFAULT_REGION` if unset.  
- Athena results: `s3://$S3_BUCKET/$ATHENA_RESULTS_PREFIX/` must be writable.  
- `ROLE_ARN`: IAM role for SageMaker Processing (S3 access to input/output, plus Processing permissions).

### 3. Run the pipeline

```bash
python -m src
```

This runs the full flow above (Athena DB + table + views, S3 upload/transform/curate, SageMaker XGBoost job).

### 4. Run notebooks only (local)

Open and run:

- `notebooks/01_pre-processing.ipynb`
- `notebooks/02_feature-exploration.ipynb`
- `notebooks/03_churn_modelling.ipynb`

## Known limitations

- **Infra SQL**: `01_create_customer_features_table.sql` uses a **hard-coded** S3 location (`s3://airline-customer-analysis/curated/customer_features/`). It is not parameterized from `S3_BUCKET` / `CURATED_PREFIX`. Use that bucket/path or edit the SQL to match your setup.
- **Athena context**: `run_sql_file` does not set `QueryExecutionContext.database`; DDL uses fully qualified names (`airline_analytics.*`).
- **Costs & permissions**: `python -m src` uses S3, Athena, and SageMaker; ensure IAM allows it and be aware of AWS usage.

## Roadmap

- Parameterize infra SQL (bucket, prefixes) via env or templating.
- Optional Glue crawler/catalog integration.
- QuickSight (or similar) dashboards for KPIs and segments.
