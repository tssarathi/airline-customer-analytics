CREATE EXTERNAL TABLE IF NOT EXISTS airline_analytics.churn_predictions (
    loyalty_number BIGINT,
    priority_score DOUBLE
)
STORED AS PARQUET
LOCATION 's3://airline-customer-analysis/curated/churn/predictions/';
