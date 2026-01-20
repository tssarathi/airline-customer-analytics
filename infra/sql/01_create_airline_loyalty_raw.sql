CREATE EXTERNAL TABLE IF NOT EXISTS airline-analytics.airline_loyalty_raw (
    loyalty_number BIGINT,
    country STRING,
    province STRING,
    city STRING,
    postal_code STRING,
    gender STRING,
    education STRING,
    salary DOUBLE,
    marital_status STRING,
    loyalty_card STRING,
    clv DOUBLE,
    enrollment_type STRING,
    enrollment_year INT,
    enrollment_month INT,
    cancellation_year INT,
    cancellation_month INT
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
    'separatorChar' = ','
)
STORED AS TEXTFILE
LOCATION 's3://airline-customer-analytics/raw/'
TBLPROPERTIES (
    'skip.header.line.count' = '1',
    'use.null.for.invalid.data' = 'true'
);
