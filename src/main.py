from src.utils.athena_utils import run_sql_file
from src.utils.s3_utils import upload_file_to_s3
from src.config.config import S3_BUCKET, RAW_FOLDER, PROCESSED_FOLDER, CURATED_FOLDER
from src.etl.csv_to_parquet import csv_to_parquet_s3
from src.etl.transforms import cast_clh
from src.etl.customer_features import customer_features_to_parquet_s3
from src.scripts.run_xgb_job import run_processing_job


def main():
    print("Airline Customer Analytics project booted successfully")

    # Create Athena database
    run_sql_file("infra/sql/00_create_database.sql")
    print("Athena database 'airline_analytics' created successfully")

    # Upload data to S3
    clh_path_local = "data/raw/Customer Loyalty History.csv"
    clh_path_s3 = f"s3://{S3_BUCKET}/{RAW_FOLDER}/customer_loyalty_history.csv"
    upload_file_to_s3(clh_path_local, clh_path_s3)

    cfa_path_local = "data/raw/Customer Flight Activity.csv"
    cfa_path_s3 = f"s3://{S3_BUCKET}/{RAW_FOLDER}/customer_flight_activity.csv"
    upload_file_to_s3(cfa_path_local, cfa_path_s3)

    print("Data uploaded to S3 successfully")

    # Transform CSV to Parquet
    clh_csv_path_s3 = f"s3://{S3_BUCKET}/{RAW_FOLDER}/customer_loyalty_history.csv"
    clh_parquet_path_s3 = (
        f"s3://{S3_BUCKET}/{PROCESSED_FOLDER}/customer_loyalty_history.parquet"
    )
    csv_to_parquet_s3(clh_csv_path_s3, clh_parquet_path_s3, cast_clh)

    cfa_csv_path_s3 = f"s3://{S3_BUCKET}/{RAW_FOLDER}/customer_flight_activity.csv"
    cfa_parquet_path_s3 = (
        f"s3://{S3_BUCKET}/{PROCESSED_FOLDER}/customer_flight_activity.parquet"
    )
    csv_to_parquet_s3(cfa_csv_path_s3, cfa_parquet_path_s3)

    print("Transformed CSV to Parquet successfully")

    # Build Customer Features
    customer_features_parquet_path_s3 = (
        f"s3://{S3_BUCKET}/{CURATED_FOLDER}/customer_features/customer_features.parquet"
    )
    customer_features_to_parquet_s3(
        input_cfa_parquet_s3=cfa_parquet_path_s3,
        input_clh_parquet_s3=clh_parquet_path_s3,
        output_customer_features_s3=customer_features_parquet_path_s3,
    )

    print("Built customer features successfully")

    # Create Athena Tables
    run_sql_file("infra/sql/01_create_customer_features_table.sql")

    print("Athena table 'customer_features' created successfully")

    # Create Athena Views
    run_sql_file("infra/sql/02_segment_distribution.sql")
    run_sql_file("infra/sql/03_at_high_risk.sql")
    run_sql_file("infra/sql/04_segment_by_province.sql")

    print("Athena views created successfully")

    # Run XGBoost Processing Job
    run_processing_job()

    print("XGBoost processing job completed successfully")


if __name__ == "__main__":
    main()
