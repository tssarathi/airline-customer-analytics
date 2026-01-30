import time
from pathlib import Path
import boto3
from src.config.config import (
    AWS_REGION,
    ATHENA_RESULTS_FOLDER,
    S3_BUCKET,
)


def run_sql_file(sql_path: str, database: str | None = None) -> None:
    query = Path(sql_path).read_text()
    athena = boto3.client("athena", region_name=AWS_REGION)

    params = {
        "QueryString": query,
        "ResultConfiguration": {
            "OutputLocation": f"s3://{S3_BUCKET}/{ATHENA_RESULTS_FOLDER}/",
        },
    }
    if database:
        params["QueryExecutionContext"] = {"Database": database}

    response = athena.start_query_execution(**params)

    query_id = response["QueryExecutionId"]
    print(f"Query execution started with ID: {query_id}")

    while True:
        status = athena.get_query_execution(QueryExecutionId=query_id)[
            "QueryExecution"
        ]["Status"]["State"]

        if status == "SUCCEEDED":
            print("Query execution completed successfully")
            return

        if status in ["FAILED", "CANCELLED"]:
            raise RuntimeError("Query execution failed")

        time.sleep(1)
