import pandas as pd


def cast_clh(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["loyalty_number"] = df["loyalty_number"].astype("int64")
    df["salary"] = df["salary"].astype("float64")
    df["clv"] = df["clv"].astype("float64")

    df["enrollment_year"] = df["enrollment_year"].astype("int64")
    df["enrollment_month"] = df["enrollment_month"].astype("int64")

    df["cancellation_year"] = df["cancellation_year"].astype("Int64")
    df["cancellation_month"] = df["cancellation_month"].astype("Int64")

    string_cols = [
        "country",
        "province",
        "city",
        "postal_code",
        "gender",
        "education",
        "marital_status",
        "loyalty_card",
        "enrollment_type",
    ]
    df[string_cols] = df[string_cols].astype("string")

    return df
