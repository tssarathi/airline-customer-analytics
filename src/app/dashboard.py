import streamlit as st
import awswrangler as wr
import altair as alt
import pandas as pd
import json
import boto3
from dotenv import load_dotenv
import os

load_dotenv()
GLUE_DATABASE = os.getenv("GLUE_DB")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
AWS_REGION = os.getenv("AWS_REGION")

st.set_page_config(page_title="Airline Customer Analytics", layout="wide")

RFM_DOMAIN = ["Dormant", "At Risk", "Potential", "Loyal", "Champions"]


@st.cache_resource
def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def invoke_claude(
    user_prompt: str,
    system_prompt: str,
    max_tokens: int = 800,
    temperature: float = 0.2,
) -> str:
    brt = get_bedrock_client()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    resp = brt.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(body).encode("utf-8"),
        contentType="application/json",
        accept="application/json",
    )

    payload = json.loads(resp["body"].read())

    parts = []
    for block in payload.get("content", []):
        if block.get("type") == "text":
            parts.append(block.get("text", ""))

    return "\n".join(parts).strip()


def get_dashboard_context(df_base, baseline, seg_df, province_df):
    return {
        "filters": {
            "provinces": st.session_state.selected_provinces,
            "segments": st.session_state.selected_segments,
            "gender": st.session_state.selected_gender,
            "cards": st.session_state.selected_cards,
        },
        "baseline_kpis": baseline,
        "slice_kpis": {
            "customers": int(df_base["loyalty_number"].nunique()),
            "avg_clv": float(df_base["clv"].mean() or 0),
            "cancelled_rate": float(df_base["is_cancelled"].mean() or 0),
            "avg_recency_days": float((df_base["recency"].mean() or 0) * 30),
            "avg_frequency": float(df_base["frequency"].mean() or 0),
            "avg_monetary": float(df_base["monetary"].mean() or 0),
            "avg_tenure_months": float(df_base["tenure_months"].mean() or 0),
            "avg_churn_score": float(df_base["churn_score"].mean() or 0),
        },
        "province_preview": province_df.head(5).to_dict("records"),
        "segment_preview": seg_df.head(5).to_dict("records"),
    }


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
        database=GLUE_DATABASE,
        ctas_approach=False,
    )

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

baseline = {
    "customers": int(df["loyalty_number"].nunique()),
    "avg_clv": float(df["clv"].mean() or 0),
    "cancelled_rate": float(df["is_cancelled"].mean() or 0),
    "avg_recency_days": float((df["recency"].mean() or 0) * 30),
    "avg_frequency": float(df["frequency"].mean() or 0),
    "avg_monetary": float(df["monetary"].mean() or 0),
    "avg_tenure_months": float(df["tenure_months"].mean() or 0),
    "avg_churn_score": float(df["churn_score"].mean() or 0),
}

st.title("Airline Customer Analytics")
st.caption("Customer portfolio snapshot and churn risk prioritisation")

st.sidebar.header("Filters")

if "selected_provinces" not in st.session_state:
    st.session_state.selected_provinces = []
if "selected_segments" not in st.session_state:
    st.session_state.selected_segments = []
if "selected_gender" not in st.session_state:
    st.session_state.selected_gender = "All"
if "selected_cards" not in st.session_state:
    st.session_state.selected_cards = []


prov_domain = sorted(df["province"].dropna().unique().tolist())
seg_domain = [
    s for s in RFM_DOMAIN if s in set(df["rfm_segment"].dropna().unique().tolist())
]
gender_domain = sorted(
    [g for g in df["gender"].dropna().unique().tolist() if str(g).strip() != ""]
)
card_domain = sorted(
    [c for c in df["loyalty_card"].dropna().unique().tolist() if str(c).strip() != ""]
)

selected_provinces = st.sidebar.multiselect(
    "Province", prov_domain, key="selected_provinces"
)
selected_segments = st.sidebar.multiselect(
    "RFM Segment", seg_domain, key="selected_segments"
)
selected_gender = st.sidebar.selectbox(
    "Gender", ["All"] + gender_domain, key="selected_gender"
)
selected_cards = st.sidebar.multiselect(
    "Loyalty Card", card_domain, key="selected_cards"
)

mask = pd.Series(True, index=df.index)

if selected_provinces:
    mask &= df["province"].isin(selected_provinces)
if selected_segments:
    mask &= df["rfm_segment"].isin(selected_segments)
if selected_gender != "All":
    mask &= df["gender"] == selected_gender
if selected_cards:
    mask &= df["loyalty_card"].isin(selected_cards)

df_base = df.loc[mask].copy()

if df_base.empty:
    st.warning(
        "No customers exist for the selected filter combination. Please adjust your filters."
    )
    st.stop()

total_customers = df_base["loyalty_number"].nunique()
avg_clv = df_base["clv"].mean()
cancelled_rate = df_base["is_cancelled"].mean()
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

seg_df = (
    df_base.groupby("rfm_segment")
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

province_df = (
    df_base.groupby("province")
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

chart1 = (
    alt.Chart(seg_df)
    .mark_bar()
    .encode(
        x=alt.X(
            "rfm_segment:N",
            axis=alt.Axis(labelAngle=-45, labelBaseline="top", labelOverlap=False),
            scale=alt.Scale(domain=RFM_DOMAIN),
        ),
        y=alt.Y("customers:Q"),
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
            alt.Tooltip("customers:Q", format=","),
            alt.Tooltip("avg_clv:Q", format="$,.0f"),
            alt.Tooltip("avg_churn_score:Q", format=".0f"),
            alt.Tooltip("avg_recency_days:Q", format=",.0f"),
            alt.Tooltip("avg_frequency:Q", format=",.0f"),
            alt.Tooltip("avg_monetary:Q", format=",.0f"),
            alt.Tooltip("avg_tenure_months:Q", format=",.0f"),
            alt.Tooltip("is_cancelled:Q", format=".0%"),
        ],
    )
)

chart2 = (
    alt.Chart(province_df)
    .mark_bar()
    .encode(
        x=alt.X(
            "province:N",
            axis=alt.Axis(
                labelAngle=-45, labelBaseline="top", labelLimit=200, labelOverlap=False
            ),
            scale=alt.Scale(domain=prov_domain),
        ),
        y=alt.Y("customers:Q"),
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
            alt.Tooltip("customers:Q", format=","),
            alt.Tooltip("avg_clv:Q", format="$,.0f"),
            alt.Tooltip("avg_churn_score:Q", format=".0f"),
            alt.Tooltip("avg_recency_days:Q", format=",.0f"),
            alt.Tooltip("avg_frequency:Q", format=",.0f"),
            alt.Tooltip("avg_monetary:Q", format=",.0f"),
            alt.Tooltip("avg_tenure_months:Q", format=",.0f"),
            alt.Tooltip("is_cancelled:Q", format=".0%"),
        ],
    )
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Customer Segments")
    st.altair_chart(chart1, width="stretch")
with col2:
    st.subheader("Customer Provinces")
    st.altair_chart(chart2, width="stretch")

st.divider()

p1, p2 = st.columns(2)

with p1:
    st.subheader("Customer Gender Distribution")
    gender_df = df_base.groupby("gender").size().reset_index(name="count")
    gender_df["percent"] = gender_df["count"] / gender_df["count"].sum()
    gender_pie = (
        alt.Chart(gender_df)
        .mark_arc()
        .encode(
            theta="count:Q",
            color="gender:N",
            tooltip=[
                alt.Tooltip("gender:N"),
                alt.Tooltip("count:Q", format=","),
                alt.Tooltip("percent:Q", format=".1%"),
            ],
        )
    )
    st.altair_chart(gender_pie, width="stretch")

with p2:
    st.subheader("Customer Loyalty Card Distribution")
    card_df = df_base.groupby("loyalty_card").size().reset_index(name="count")
    card_df["percent"] = card_df["count"] / card_df["count"].sum()
    card_pie = (
        alt.Chart(card_df)
        .mark_arc()
        .encode(
            theta="count:Q",
            color="loyalty_card:N",
            tooltip=[
                alt.Tooltip("loyalty_card:N"),
                alt.Tooltip("count:Q", format=","),
                alt.Tooltip("percent:Q", format=".1%"),
            ],
        )
    )
    st.altair_chart(card_pie, width="stretch")

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

st.divider()

st.subheader("Ask the AI Agent")

ctx = get_dashboard_context(df_base, baseline, seg_df, province_df)

if "chat" not in st.session_state:
    st.session_state.chat = [
        {
            "role": "assistant",
            "content": "Ask me about the current dashboard slice. I will compare it to the overall baseline.",
        }
    ]

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Try: Why is churn high in this slice?")
if q:
    st.session_state.chat.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    SYSTEM_PROMPT = """
You are a stakeholder-facing analytics copilot for an airline loyalty dashboard.

Rules:
- Use ONLY the numbers in the provided context JSON.
- Do not invent metrics or values.
- Explain in simple, non-technical language.
- Provide 2–4 churn conversion actions grounded in what the context shows.

Format:
1) What we're seeing
2) Why it's happening
3) What we should do next
4) Evidence (key numbers)
"""

    user_prompt = f"""
User question: {q}

Context JSON:
{json.dumps(ctx, indent=2)}
"""

    try:
        ans = invoke_claude(user_prompt=user_prompt, system_prompt=SYSTEM_PROMPT)
    except Exception as e:
        ans = f"Bedrock call failed: {e}"

    st.session_state.chat.append({"role": "assistant", "content": ans})
    with st.chat_message("assistant"):
        st.markdown(ans)
