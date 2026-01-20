import pandas as pd
from typing import Callable
from src.utils.s3_utils import s3_object_exists, parse_s3_uri


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def csv_to_parquet_s3(
    input_csv_s3: str,
    output_parquet_s3: str,
    transform_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> None:
    bucket, key = parse_s3_uri(output_parquet_s3)
    if s3_object_exists(bucket, key):
        print(f"Parquet already exists in S3 at s3://{bucket}/{key}")
        return

    df = pd.read_csv(input_csv_s3)
    df = standardize_columns(df)

    if transform_fn is not None:
        df = transform_fn(df)

    df.to_parquet(output_parquet_s3, index=False)
    print(f"Parquet saved to s3://{bucket}/{key}")
