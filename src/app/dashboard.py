import os
import streamlit as st
import awswrangler as wr
import altair as alt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATABASE = os.getenv("GLUE_DB")

st.set_page_config(page_title="Airline Customer Analytics", layout="wide")


@st.cache_data(ttl=600, show_spinner="Loading...")
def load_customer_scored():
    sql = """
    SELECT
        loyalty_number,
        province,
        loyalty_card,
        clv,
        rfm_segment,
        churn_score,
        recency,
        tenure_months,
        is_cancelled
    FROM customer_scored;
    """
    df = wr.athena.read_sql_query(
        sql=sql,
        database=DATABASE,
        ctas_approach=False,
    )

    if "is_cancelled" in df.columns:
        df["is_cancelled"] = df["is_cancelled"].astype(bool)

    num_cols = ["clv", "churn_score", "recency", "tenure_months"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


df = load_customer_scored()

st.title("Airline Customer Analytics")
st.caption("Customer portfolio snapshot and churn risk prioritisation")

st.sidebar.header("Filters")

province_options = ["All"] + sorted(df["province"].dropna().unique().tolist())
segment_options = ["All"] + sorted(df["rfm_segment"].dropna().unique().tolist())

province = st.sidebar.selectbox("Province", province_options)
segment = st.sidebar.selectbox("RFM Segment", segment_options)

mask = pd.Series(True, index=df.index)
if province != "All":
    mask &= df["province"] == province
if segment != "All":
    mask &= df["rfm_segment"] == segment

df_f = df.loc[mask].copy()

total_customers = df_f["loyalty_number"].nunique()
avg_clv = df_f["clv"].mean()
cancelled_rate = df_f["is_cancelled"].mean()
avg_recency_days = df_f["recency"].mean() * 30
avg_tenure_months = df_f["tenure_months"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Customers", f"{int(total_customers):,}")
c2.metric("Average CLV", f"${(avg_clv or 0):,.0f}")
c3.metric("Cancellation Rate", f"{(cancelled_rate or 0):.0%}")
c4.metric("Avg Recency (days)", f"{(avg_recency_days or 0):.0f}")
c5.metric("Avg Tenure (months)", f"{(avg_tenure_months or 0):.0f}")

st.subheader("Customer Segments")

seg_base_df = df if province == "All" else df[df["province"] == province]

seg_df = (
    seg_base_df.groupby("rfm_segment", dropna=False)
    .agg(
        customers=("loyalty_number", "size"),
        avg_clv=("clv", "mean"),
        avg_churn_score=("churn_score", "mean"),
        avg_recency_days=("recency", lambda s: s.mean() * 30),
    )
    .reset_index()
    .sort_values("avg_churn_score", ascending=False)
)

base1 = (
    alt.Chart(seg_df)
    .mark_bar()
    .encode(
        x=alt.X(
            "rfm_segment:N",
            title="RFM Segment",
            axis=alt.Axis(labelAngle=-45, labelBaseline="top"),
        ),
        y=alt.Y("customers:Q", title="No of Customers"),
        tooltip=[
            alt.Tooltip("customers:Q", format=",", title="No of Customers"),
            alt.Tooltip("avg_clv:Q", format="$,.0f", title="Avg CLV"),
            alt.Tooltip("avg_churn_score:Q", format=".2f", title="Avg Churn Score"),
        ],
    )
)

chart1 = (
    base1
    if segment == "All"
    else base1.encode(
        opacity=alt.condition(
            alt.datum.rfm_segment == segment,
            alt.value(1.0),
            alt.value(0.3),
        )
    )
)

st.altair_chart(chart1, width="stretch")

st.subheader("Province Summary")

prov_base_df = df if segment == "All" else df[df["rfm_segment"] == segment]

province_df = (
    prov_base_df.groupby("province", dropna=False)
    .agg(
        customers=("loyalty_number", "size"),
        avg_clv=("clv", "mean"),
        avg_churn_score=("churn_score", "mean"),
        avg_recency_days=("recency", lambda s: s.mean() * 30),
    )
    .reset_index()
    .sort_values("avg_churn_score", ascending=False)
)

base2 = (
    alt.Chart(province_df)
    .mark_bar()
    .encode(
        x=alt.X(
            "province:N",
            title="Province",
            axis=alt.Axis(labelAngle=-45, labelBaseline="top", labelLimit=200),
        ),
        y=alt.Y("customers:Q", title="No of Customers"),
        tooltip=[
            alt.Tooltip("customers:Q", format=",", title="No of Customers"),
            alt.Tooltip("avg_clv:Q", format="$,.0f", title="Avg CLV"),
            alt.Tooltip("avg_churn_score:Q", format=".2f", title="Avg Churn Score"),
        ],
    )
)

chart2 = (
    base2
    if province == "All"
    else base2.encode(
        opacity=alt.condition(
            alt.datum.province == province,
            alt.value(1.0),
            alt.value(0.3),
        )
    )
)

st.altair_chart(chart2, width="stretch")

st.subheader("Top Risk Customers")

top_risk_df = (
    df_f.sort_values("churn_score", ascending=False)
    .head(50)[
        [
            "loyalty_number",
            "province",
            "loyalty_card",
            "clv",
            "rfm_segment",
            "churn_score",
        ]
    ]
    .rename(
        columns={
            "loyalty_number": "Loyalty Number",
            "province": "Province",
            "loyalty_card": "Loyalty Card",
            "clv": "CLV",
            "rfm_segment": "RFM Segment",
            "churn_score": "Churn Score",
        }
    )
)

top_risk_df["CLV"] = (
    top_risk_df["CLV"].round(0).map(lambda x: f"${x:,.0f}" if pd.notnull(x) else "")
)
top_risk_df["Churn Score"] = top_risk_df["Churn Score"].round(0)

st.dataframe(top_risk_df, width="stretch")
