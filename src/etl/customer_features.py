import numpy as np
import pandas as pd

from src.utils.s3_utils import s3_object_exists, parse_s3_uri


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def months_diff(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)


def assign_rfm_segment(r: int, f: int, m: int) -> str:
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r <= 2 and f >= 3:
        return "At Risk"
    if r <= 2 and f <= 2:
        return "Dormant"
    return "Potential"


def build_customer_features(
    cfa_df: pd.DataFrame,
    clh_df: pd.DataFrame,
) -> pd.DataFrame:
    cfa_df["activity_date"] = pd.to_datetime(
        dict(year=cfa_df["year"], month=cfa_df["month"], day=1)
    )
    max_activity_date = cfa_df["activity_date"].max()

    customers = cfa_df[["loyalty_number"]].drop_duplicates()

    recent_activity = (
        cfa_df[cfa_df["total_flights"] > 0]
        .sort_values("activity_date")
        .groupby("loyalty_number", as_index=False)
        .last()[["loyalty_number", "activity_date"]]
    )
    recent_activity["recency"] = months_diff(
        pd.Series(
            [max_activity_date] * len(recent_activity), index=recent_activity.index
        ),
        recent_activity["activity_date"],
    )

    customer_recency = customers.merge(
        recent_activity[["loyalty_number", "recency"]],
        on="loyalty_number",
        how="left",
    )
    max_recency = customer_recency["recency"].max()
    customer_recency["recency"] = (
        customer_recency["recency"].fillna(max_recency + 1).astype("int64")
    )

    customer_recency["r_score"] = pd.cut(
        customer_recency["recency"],
        bins=[-1, 1, 3, 6, 12, 24],
        labels=[5, 4, 3, 2, 1],
    ).astype("int64")

    customer_frequency = cfa_df.groupby("loyalty_number", as_index=False).agg(
        frequency=("total_flights", "sum")
    )
    customer_frequency["f_score"] = pd.qcut(
        customer_frequency["frequency"], q=5, labels=[1, 2, 3, 4, 5]
    ).astype("int64")

    monetary_distance = (
        cfa_df[cfa_df["total_flights"] > 0]
        .groupby("loyalty_number", as_index=False)
        .agg(monetary=("distance", "sum"))
    )
    customer_monetary = customers.merge(
        monetary_distance,
        on="loyalty_number",
        how="left",
    )
    customer_monetary["monetary"] = (
        customer_monetary["monetary"].fillna(0).astype("int64")
    )

    customer_monetary["m_score"] = pd.qcut(
        customer_monetary["monetary"], q=5, labels=[1, 2, 3, 4, 5]
    ).astype("int64")

    rfm_df = (
        customers.merge(
            customer_recency[["loyalty_number", "recency", "r_score"]],
            on="loyalty_number",
            how="left",
        )
        .merge(
            customer_frequency[["loyalty_number", "frequency", "f_score"]],
            on="loyalty_number",
            how="left",
        )
        .merge(
            customer_monetary[["loyalty_number", "monetary", "m_score"]],
            on="loyalty_number",
            how="left",
        )
    )

    rfm_df["rfm_segment"] = rfm_df.apply(
        lambda row: assign_rfm_segment(
            int(row["r_score"]), int(row["f_score"]), int(row["m_score"])
        ),
        axis=1,
    ).astype("string")

    clh_df["is_cancelled"] = clh_df["cancellation_year"].notna()

    clh_df["enrollment_date"] = pd.to_datetime(
        dict(year=clh_df["enrollment_year"], month=clh_df["enrollment_month"], day=1),
        errors="coerce",
    )

    clh_df["cancellation_date"] = pd.NaT
    mask = clh_df["is_cancelled"]
    clh_df.loc[mask, "cancellation_date"] = pd.to_datetime(
        dict(
            year=clh_df.loc[mask, "cancellation_year"].astype(int),
            month=clh_df.loc[mask, "cancellation_month"].astype(int),
            day=1,
        ),
        errors="coerce",
    )

    reference_date = max_activity_date

    clh_df["tenure_end_date"] = np.where(
        clh_df["is_cancelled"],
        clh_df["cancellation_date"],
        reference_date,
    )
    clh_df["tenure_end_date"] = pd.to_datetime(
        clh_df["tenure_end_date"], errors="coerce"
    )

    clh_df["tenure_months"] = months_diff(
        clh_df["tenure_end_date"], clh_df["enrollment_date"]
    ).clip(lower=0)
    clh_df["tenure_months"] = clh_df["tenure_months"].astype("int32")

    clh_keep = [
        "loyalty_number",
        "province",
        "city",
        "gender",
        "education",
        "loyalty_card",
        "clv",
        "is_cancelled",
        "tenure_months",
    ]
    clh_small = clh_df[clh_keep].copy()

    customer_features = clh_small.merge(rfm_df, on="loyalty_number", how="left")

    customer_features["loyalty_number"] = customer_features["loyalty_number"].astype(
        "int64"
    )

    return customer_features


def customer_features_to_parquet_s3(
    input_cfa_parquet_s3: str,
    input_clh_parquet_s3: str,
    output_customer_features_s3: str,
) -> None:
    bucket, key = parse_s3_uri(output_customer_features_s3)
    if s3_object_exists(bucket, key):
        print(f"Customer features Parquet already exists in S3 at s3://{bucket}/{key}")
        return

    cfa_df = pd.read_parquet(input_cfa_parquet_s3, engine="pyarrow")
    clh_df = pd.read_parquet(input_clh_parquet_s3, engine="pyarrow")

    customer_features = build_customer_features(cfa_df=cfa_df, clh_df=clh_df)

    customer_features.to_parquet(
        output_customer_features_s3, index=False, engine="pyarrow"
    )
    print(f"Customer features Parquet saved to s3://{bucket}/{key}")
