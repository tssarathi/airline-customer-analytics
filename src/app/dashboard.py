import os
import streamlit as st
import awswrangler as wr
import altair as alt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATABASE = os.getenv("GLUE_DB")

st.set_page_config(page_title="Airline Customer Analytics", layout="wide")

RFM_DOMAIN = ["Dormant", "At Risk", "Potential", "Loyal", "Champions"]


@st.cache_data(ttl=600, show_spinner="Loading...")
def load_customer_scored():
    sql = """
    SELECT
        loyalty_number,
        gender,
        province,
        loyalty_card,
        clv,
        rfm_segment,
        churn_score,
        recency,
        frequency,
        monetary,
        tenure_months,
        is_cancelled
    FROM customer_scored;
    """
    df = wr.athena.read_sql_query(
        sql=sql,
        database=DATABASE,
        ctas_approach=False,
    )

    # Casts / types
    if "is_cancelled" in df.columns:
        df["is_cancelled"] = df["is_cancelled"].astype(bool)

    for c in [
        "clv",
        "churn_score",
        "recency",
        "tenure_months",
        "frequency",
        "monetary",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


df = load_customer_scored()

st.title("Airline Customer Analytics")
st.caption("Customer portfolio snapshot and churn risk prioritisation")

st.sidebar.header("Filters")

prov_domain = sorted(df["province"].unique().tolist())
seg_domain = [s for s in RFM_DOMAIN if s in set(df["rfm_segment"].unique().tolist())]

selected_provinces = st.sidebar.multiselect("Province", prov_domain, default=[])
selected_segments = st.sidebar.multiselect("RFM Segment", seg_domain, default=[])

mask = pd.Series(True, index=df.index)
if selected_provinces:
    mask &= df["province"].isin(selected_provinces)
if selected_segments:
    mask &= df["rfm_segment"].isin(selected_segments)

df_base = df.loc[mask].copy()

total_customers = df_base["loyalty_number"].nunique()
avg_clv = df_base["clv"].mean()
cancelled_rate = (
    df_base["is_cancelled"].mean() if "is_cancelled" in df_base.columns else 0
)
avg_recency_days = df_base["recency"].mean() * 30
avg_frequency = df_base["frequency"].mean()
avg_monetary_value = df_base["monetary"].mean()
avg_tenure_months = df_base["tenure_months"].mean()

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Total Customers", f"{int(total_customers):,}")
c2.metric("Average CLV", f"${(avg_clv or 0):,.0f}")
c3.metric("Cancellation Rate", f"{(cancelled_rate or 0):.0%}")
c4.metric("Avg Recency (days)", f"{(avg_recency_days or 0):.0f}")
c5.metric("Avg Frequency (flights)", f"{(avg_frequency or 0):.0f}")
c6.metric("Avg Monetary (kms)", f"{(avg_monetary_value or 0):,.0f}")
c7.metric("Avg Tenure (months)", f"{(avg_tenure_months or 0):.0f}")

st.divider()

seg_base_df = (
    df if not selected_provinces else df[df["province"].isin(selected_provinces)]
)

seg_df = (
    seg_base_df.groupby("rfm_segment", dropna=False)
    .agg(
        customers=("loyalty_number", "nunique"),
        avg_clv=("clv", "mean"),
        avg_churn_score=("churn_score", "mean"),
        avg_recency_days=("recency", lambda s: s.mean() * 30),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        avg_tenure_months=("tenure_months", "mean"),
        is_cancelled=("is_cancelled", "mean"),
    )
    .reset_index()
    .sort_values("avg_churn_score", ascending=False)
)

prov_base_df = (
    df if not selected_segments else df[df["rfm_segment"].isin(selected_segments)]
)

province_df = (
    prov_base_df.groupby("province", dropna=False)
    .agg(
        customers=("loyalty_number", "nunique"),
        avg_clv=("clv", "mean"),
        avg_churn_score=("churn_score", "mean"),
        avg_recency_days=("recency", lambda s: s.mean() * 30),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        avg_tenure_months=("tenure_months", "mean"),
        is_cancelled=("is_cancelled", "mean"),
    )
    .reset_index()
    .sort_values("avg_churn_score", ascending=False)
)

churn_min = float(
    min(seg_df["avg_churn_score"].min(), province_df["avg_churn_score"].min())
)
churn_max = float(
    max(seg_df["avg_churn_score"].max(), province_df["avg_churn_score"].max())
)
denom = churn_max - churn_min

seg_df["churn_norm"] = (
    0.5 if denom == 0 else (seg_df["avg_churn_score"] - churn_min) / denom
)
province_df["churn_norm"] = (
    0.5 if denom == 0 else (province_df["avg_churn_score"] - churn_min) / denom
)

base1 = (
    alt.Chart(seg_df)
    .mark_bar()
    .encode(
        x=alt.X(
            "rfm_segment:N",
            title="RFM Segment",
            axis=alt.Axis(labelAngle=-45, labelBaseline="top", labelOverlap=False),
            scale=alt.Scale(domain=RFM_DOMAIN),
        ),
        y=alt.Y("customers:Q", title="No of Customers"),
        color=alt.Color(
            "churn_norm:Q",
            scale=alt.Scale(scheme="orangered", domain=[0, 1]),
            legend=alt.Legend(
                title="Churn Risk",
                orient="right",
                values=[0, 1],
                labelExpr="datum.value == 0 ? 'Low' : 'High'",
                gradientLength=200,
            ),
        ),
        tooltip=[
            alt.Tooltip("customers:Q", format=",", title="No of Customers"),
            alt.Tooltip("avg_clv:Q", format="$,.0f", title="Avg CLV"),
            alt.Tooltip("avg_churn_score:Q", format=".0f", title="Avg Churn Score"),
            alt.Tooltip(
                "avg_recency_days:Q", format=",.0f", title="Avg Recency (days)"
            ),
            alt.Tooltip(
                "avg_frequency:Q", format=",.0f", title="Avg Frequency (flights)"
            ),
            alt.Tooltip("avg_monetary:Q", format=",.0f", title="Avg Monetary (kms)"),
            alt.Tooltip(
                "avg_tenure_months:Q", format=",.0f", title="Avg Tenure (months)"
            ),
            alt.Tooltip("is_cancelled:Q", format=".0%", title="Cancellation Rate"),
        ],
    )
)

base1_holder = (
    alt.Chart(seg_df)
    .mark_bar(opacity=0)
    .encode(
        x=alt.X(
            "rfm_segment:N",
            title="RFM Segment",
            axis=alt.Axis(labelAngle=-45, labelBaseline="top", labelOverlap=False),
            scale=alt.Scale(domain=RFM_DOMAIN),
        )
    )
)

base1_visible = (
    base1
    if not selected_segments
    else base1.transform_filter(
        alt.FieldOneOfPredicate(field="rfm_segment", oneOf=selected_segments)
    )
)

chart1 = base1_holder + base1_visible

base2 = (
    alt.Chart(province_df)
    .mark_bar()
    .encode(
        x=alt.X(
            "province:N",
            title="Province",
            axis=alt.Axis(
                labelAngle=-45, labelBaseline="top", labelLimit=200, labelOverlap=False
            ),
            scale=alt.Scale(domain=prov_domain),
        ),
        y=alt.Y("customers:Q", title="No of Customers"),
        color=alt.Color(
            "churn_norm:Q",
            scale=alt.Scale(scheme="orangered", domain=[0, 1]),
            legend=alt.Legend(
                title="Churn Risk",
                orient="right",
                values=[0, 1],
                labelExpr="datum.value == 0 ? 'Low' : 'High'",
            ),
        ),
        tooltip=[
            alt.Tooltip("customers:Q", format=",", title="No of Customers"),
            alt.Tooltip("avg_clv:Q", format="$,.0f", title="Avg CLV"),
            alt.Tooltip("avg_churn_score:Q", format=".0f", title="Avg Churn Score"),
            alt.Tooltip(
                "avg_recency_days:Q", format=",.0f", title="Avg Recency (days)"
            ),
            alt.Tooltip(
                "avg_frequency:Q", format=",.0f", title="Avg Frequency (flights)"
            ),
            alt.Tooltip("avg_monetary:Q", format=",.0f", title="Avg Monetary (kms)"),
            alt.Tooltip(
                "avg_tenure_months:Q", format=",.0f", title="Avg Tenure (months)"
            ),
            alt.Tooltip("is_cancelled:Q", format=".0%", title="Cancellation Rate"),
        ],
    )
)

base2_holder = (
    alt.Chart(province_df)
    .mark_bar(opacity=0)
    .encode(
        x=alt.X(
            "province:N",
            title="Province",
            axis=alt.Axis(
                labelAngle=-45, labelBaseline="top", labelLimit=200, labelOverlap=False
            ),
            scale=alt.Scale(domain=prov_domain),
        )
    )
)

base2_visible = (
    base2
    if not selected_provinces
    else base2.transform_filter(
        alt.FieldOneOfPredicate(field="province", oneOf=selected_provinces)
    )
)

chart2 = base2_holder + base2_visible

col1, col2 = st.columns(2)
with col1:
    st.subheader("Customer Segments")
    st.altair_chart(chart1)
with col2:
    st.subheader("Customer Provinces")
    st.altair_chart(chart2)

st.divider()

st.subheader("Top Risk Customers")

top_risk_df = (
    df_base.sort_values("churn_score", ascending=False)
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
    .reset_index(drop=True)
)

top_risk_df.index = top_risk_df.index + 1
top_risk_df["CLV"] = (
    pd.to_numeric(top_risk_df["CLV"], errors="coerce")
    .fillna(0)
    .round(0)
    .map(lambda x: f"${x:,.0f}")
)
top_risk_df["Churn Score"] = pd.to_numeric(
    top_risk_df["Churn Score"], errors="coerce"
).round(0)

st.dataframe(top_risk_df, width="stretch")
