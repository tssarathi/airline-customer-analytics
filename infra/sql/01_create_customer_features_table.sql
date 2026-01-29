CREATE EXTERNAL TABLE IF NOT EXISTS airline_analytics.customer_features (
    loyalty_number BIGINT,
    country STRING,
    province STRING,
    city STRING,
    gender STRING,
    education STRING,
    loyalty_card STRING,
    clv DOUBLE,
    is_cancelled BOOLEAN,
    tenure_months INT,
    recency INT,
    r_score INT,
    frequency INT,
    f_score INT,
    monetary BIGINT,
    m_score INT,
    rfm_segment STRING
)
STORED AS PARQUET
LOCATION 's3://airline-customer-analytics/processed/customer_features/';