from src.utils.athena_utils import run_sql_file
from src.utils.s3_utils import upload_file_to_s3
from src.config.config import S3_BUCKET, RAW_FOLDER, PROCESSED_FOLDER
from src.etl.csv_to_parquet import csv_to_parquet_s3
from src.etl.transforms import cast_clh


def main():
    print("Airline Customer Analytics project booted successfully")

    # Create Glue database
    run_sql_file("infra/sql/00_create_database.sql")
    print("Glue database 'airline_analytics' created successfully")

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

    print("Tansformed CSV to Parquet successfully")


if __name__ == "__main__":
    main()
