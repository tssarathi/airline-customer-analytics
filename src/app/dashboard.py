import os
import streamlit as st
import awswrangler as wr
import altair as alt
from dotenv import load_dotenv

load_dotenv()
DATABASE = os.getenv("GLUE_DB")

st.set_page_config(page_title="Airline Customer Analytics", layout="wide")


@st.cache_data(ttl=600)
def loading(sql):
    return wr.athena.read_sql_query(sql=sql, database=DATABASE, ctas_approach=False)


st.title("Airline Customer Analytics")
st.caption("Customer portfolio snapshot and churn risk prioritisation")

st.sidebar.header("Filters")

province = st.sidebar.selectbox(
    "Province",
    ["All"]
    + sorted(
        loading("SELECT DISTINCT province FROM vw_province_summary")[
            "province"
        ].tolist()
    ),
)

segment = st.sidebar.selectbox(
    "RFM Segment",
    ["All"]
    + sorted(
        loading("SELECT DISTINCT rfm_segment FROM vw_segment_summary")[
            "rfm_segment"
        ].tolist()
    ),
)

filters = []
province_where_clause = ""
segment_where_clause = ""

if province != "All":
    filters.append(f"province = '{province}'")
    province_where_clause = f"WHERE province = '{province}'"

if segment != "All":
    filters.append(f"rfm_segment = '{segment}'")
    segment_where_clause = f"WHERE rfm_segment = '{segment}'"

where_clause = "WHERE " + " AND ".join(filters) if filters else ""

kpi_sql = f"""
SELECT
    COUNT(DISTINCT loyalty_number) AS total_customers,
    ROUND(AVG(clv)) AS avg_clv,
    ROUND(AVG(CASE WHEN is_cancelled THEN 1 ELSE 0 END), 2) AS cancelled_rate,
    ROUND(AVG(recency) * 30) AS avg_recency_days,
    ROUND(AVG(tenure_months)) AS avg_tenure_months
FROM customer_scored
{where_clause};
"""
kpis = loading(kpi_sql).iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Customers", f"{int(kpis.total_customers):,}")
c2.metric("Average CLV", f"${kpis.avg_clv:,.0f}")
c3.metric("Cancellation Rate", f"{kpis.cancelled_rate:.0%}")
c4.metric("Avg Recency (days)", f"{kpis.avg_recency_days:.0f}")
c5.metric("Avg Tenure (months)", f"{kpis.avg_tenure_months:.0f}")

st.subheader("Customer Segments")

segment_sql = f"""
SELECT
    rfm_segment,
    COUNT(*) AS customers,
    ROUND(AVG(clv)) AS avg_clv,
    ROUND(AVG(churn_score)) AS avg_churn_score,
    ROUND(AVG(recency) * 30) AS avg_recency_days
FROM airline_analytics.customer_scored
{province_where_clause}
GROUP BY rfm_segment
ORDER BY avg_churn_score DESC;
"""
seg_df = loading(segment_sql)

base1 = (
    alt.Chart(seg_df)
    .mark_bar()
    .encode(
        x=alt.X("rfm_segment:N", title="RFM Segment"),
        y=alt.Y("customers:Q", title="No of Customers"),
        tooltip=[
            alt.Tooltip("customers:Q", format=",", title="No of Customers"),
            alt.Tooltip("avg_clv:Q", format="$,.0f", title="Avg CLV"),
            alt.Tooltip("avg_churn_score:Q", format=".2f", title="Avg Churn Score"),
        ],
    )
)

if segment == "All":
    chart1 = base1
else:
    chart1 = base1.encode(
        opacity=alt.condition(
            alt.datum.rfm_segment == segment,
            alt.value(1.0),
            alt.value(0.3),
        )
    )

st.altair_chart(chart1, width="stretch")

st.subheader("Province Summary")

province_sql = f"""
SELECT
    province,
    COUNT(*) AS customers,
    ROUND(AVG(clv)) AS avg_clv,
    ROUND(AVG(churn_score)) AS avg_churn_score,
    ROUND(AVG(recency) * 30) AS avg_recency_days
FROM airline_analytics.customer_scored
{segment_where_clause}
GROUP BY province
ORDER BY avg_churn_score DESC;
"""
province_df = loading(province_sql)

base2 = (
    alt.Chart(province_df)
    .mark_bar()
    .encode(
        x=alt.X("province:N", title="Province", sort="-y"),
        y=alt.Y("customers:Q", title="No of Customers"),
        tooltip=[
            alt.Tooltip("customers:Q", format=",", title="No of Customers"),
            alt.Tooltip("avg_clv:Q", format=",.0f", title="Avg CLV"),
            alt.Tooltip("avg_churn_score:Q", format=".0f", title="Avg Churn Score"),
        ],
    )
)

if province == "All":
    chart2 = base2
else:
    chart2 = base2.encode(
        opacity=alt.condition(
            alt.datum.province == province,
            alt.value(1.0),
            alt.value(0.3),
        )
    )

st.altair_chart(chart2, width="stretch")

st.subheader("Top Risk Customers")

top_risk_sql = f"""
SELECT
    loyalty_number,
    province,
    loyalty_card,
    clv,
    rfm_segment,
    ROUND(churn_score) AS churn_score
FROM customer_scored
{where_clause}
ORDER BY churn_score DESC
LIMIT 50;
"""
top_risk_df = loading(top_risk_sql)

st.dataframe(top_risk_df, width="stretch")
