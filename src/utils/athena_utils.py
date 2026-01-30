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
        exec_result = athena.get_query_execution(QueryExecutionId=query_id)[
            "QueryExecution"
        ]
        status = exec_result["Status"]["State"]

        if status == "SUCCEEDED":
            print("Query execution completed successfully")
            return

        if status in ["FAILED", "CANCELLED"]:
            st = exec_result["Status"]
            reason = st.get("StateChangeReason", "")
            err = st.get("AthenaError", {})
            msg = err.get("ErrorMessage") or reason or "Query execution failed"
            raise RuntimeError(f"Athena query failed: {msg}")

        time.sleep(1)
