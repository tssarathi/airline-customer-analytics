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


PLANNER_SYSTEM = """You are a data operations planner for airline customer analytics.

DATASET CONTEXT:
- ~16,700 customers in a Canadian airline loyalty program
- Each customer has demographic info, flight activity metrics, and churn predictions

COLUMNS AVAILABLE:
- loyalty_number: unique customer ID
- gender: Male/Female
- province: Canadian province (Ontario, British Columbia, Alberta, Quebec, etc.)
- loyalty_card: tier level (Star > Nova > Aurora, where Star is highest)
- clv: Customer Lifetime Value in CAD (total invoice value for all flights)
- recency: months since last flight (lower = more recently active)
- frequency: total number of flights booked
- monetary: total flight distance in km
- tenure_months: months as loyalty member
- r_score, f_score, m_score: RFM scores 1-5 (5 = best)
- rfm_segment: Champions (best) / Loyal / Potential / At Risk / Dormant (worst)
- churn_score: priority score = churn_probability * CLV (higher = more urgent to retain)
- is_cancelled: boolean, whether customer cancelled membership

AVAILABLE OPERATIONS (choose 3-6 that best answer the question):
- kpis_baseline: Overall portfolio KPIs (all customers)
- kpis_slice: KPIs for current filter selection
- summary_by_segment: Aggregate metrics grouped by rfm_segment (params: top_n, sort_by, ascending)
- summary_by_province: Aggregate metrics grouped by province (params: top_n, sort_by, ascending)
- summary_by_card: Aggregate metrics grouped by loyalty_card (params: top_n, sort_by, ascending)
- summary_by_gender: Aggregate metrics grouped by gender (params: top_n, sort_by, ascending)
- top_risk_customers: List highest churn_score customers (params: top_n)
- top_value_customers: List highest CLV customers for rewards/recognition (params: top_n, segment_filter)
- value_at_risk_by_segment: Total value at risk by rfm_segment (params: top_n, sort_by, ascending)
- value_at_risk_by_province: Total value at risk by province (params: top_n, sort_by, ascending)
- churn_by_clv_tier: Churn rates by Low/Medium/High CLV tiers
- do_nothing_scenario: Calculate impact if no retention action is taken
- single_priority_initiative: Identify the #1 priority segment and province
- segment_comparison: Compare two segments side-by-side (params: segment_a, segment_b)
- tenure_analysis: Churn rates by tenure buckets
- revenue_impact: Calculate projected annual revenue loss from churn
- correlation_drivers: Compare churned vs retained customers to understand churn causes

SORT_BY OPTIONS for summary operations:
- customers, avg_clv, avg_churn_score, avg_recency_days, avg_frequency, avg_monetary, avg_tenure_months, cancelled_rate

SORT_BY OPTIONS for value_at_risk operations:
- total_value_at_risk, total_clv, customers, avg_churn_score, avg_clv, cancelled_rate

OUTPUT FORMAT: Return ONLY valid JSON (no markdown, no explanation):
{
  "intent": "Brief 1-sentence description of your analysis approach",
  "operations": [
    {"op": "operation_name", "top_n": 5, "sort_by": "metric", "ascending": false},
    ...
  ]
}

STRATEGY:
1. Always include kpis_baseline and kpis_slice for context
2. DETECT QUESTION INTENT AND SELECT OPERATIONS ACCORDINGLY:
   - GEOGRAPHY keywords (province, region, area, location, city, where geographically):
     -> Prioritize value_at_risk_by_province, summary_by_province
   - SEGMENT keywords (segment, customer type, group, who, which customers):
     -> Prioritize value_at_risk_by_segment, summary_by_segment
   - COMPARISON keywords (compare, vs, versus, difference between):
     -> Use segment_comparison or include both segment AND province analysis
   - FINANCIAL keywords (cost, revenue, money, loss, impact, dollar):
     -> Include revenue_impact, do_nothing_scenario
   - REWARD keywords (reward, recognize, VIP, best customers, loyal, thank, appreciate, program):
     -> Prioritize top_value_customers, summary_by_segment with sort_by=avg_clv ascending=false
   - CHURN CAUSE keywords (why churn, cause, reason, driver, explain, understand churn):
     -> Include correlation_drivers, summary_by_segment
3. Use top_risk_customers when at-risk individual customers are requested
4. Use top_value_customers when high-value customers for rewards are requested
5. Use tenure_analysis for lifecycle or tenure-related questions
6. Use correlation_drivers to explain behavioral differences between churned and retained
7. Do NOT use the same operation combination for semantically different questions
8. Match your operation selection to the SPECIFIC focus of the question"""

NARRATOR_SYSTEM = """You are an executive analytics narrator for airline customer retention.

FORMATTING RULES (CRITICAL - follow exactly):
- NEVER use the $ symbol for currency - it breaks rendering
- Write currency as: "2.4M CAD" or "CAD 2,400" (not "$2.4M")
- Use **bold** with double asterisks for key numbers
- Percentages: write "18.5%" not "0.185"
- Round appropriately: currency to nearest thousand, percentages to 1 decimal

ANSWER DIFFERENTIATION (match your answer focus to the question type):
- GEOGRAPHY questions (province, region, area, location, where geographically) -> Focus answer on PROVINCES, not segments
- SEGMENT questions (customer type, segment, group, who) -> Focus answer on RFM SEGMENTS, not geography
- GENERAL questions (retention, focus, priority) -> Provide the single highest-impact finding from the data
- REWARD questions (reward, recognize, VIP, best, thank, appreciate) -> Focus on high-value LOYAL customers (Champions/Loyal segments), NOT at-risk customers
- CHURN CAUSE questions (why churn, cause, driver, reason, explain) -> Explain behavioral DIFFERENCES between churned and retained customers using the correlation_drivers data

Your role: Transform computed data summaries into clear, actionable executive insights.

RESPONSE GUIDELINES:
1. Lead with the answer/recommendation - don't make executives hunt for it
2. Quantify everything - customer counts, dollar values, percentages
3. Compare against baseline when relevant (e.g., "34% vs 18% portfolio average")
4. Highlight the single most actionable insight
5. Use business language, not technical jargon (say "cancellation rate" not "cancelled_rate")
6. Keep response concise: 3-4 short paragraphs maximum
7. Format key numbers in bold for scannability

DATA DICTIONARY (use these EXACT units when interpreting data - CRITICAL):
- recency: MONTHS since last flight (NOT days) - lower value = more recently active customer
- frequency: total number of FLIGHTS booked lifetime (NOT transactions or purchases)
- monetary: total flight DISTANCE traveled in KILOMETERS (NOT dollars - this measures travel engagement, not spend)
- clv: Customer Lifetime Value in CAD dollars (this IS the dollar spend value)
- tenure_months: months as loyalty program member
- churn_score: churn_probability x CLV in CAD (higher = more urgent to retain, dollar-weighted risk)
- cancelled_rate / is_cancelled: membership cancellation rate (0.18 = 18%)
- total_value_at_risk: sum of churn_scores across customers (dollar-weighted portfolio risk)

SEGMENT HIERARCHY:
- rfm_segment: Champions (best) > Loyal > Potential > At Risk > Dormant (worst)
- loyalty_card: Star (highest tier) > Nova > Aurora

IMPORTANT: When reporting correlation_drivers metrics:
- recency difference should be described in MONTHS (e.g., "churned customers were inactive for 17 months vs 2 months")
- monetary difference should be described in KM (e.g., "churned customers flew 15,000 km vs 50,000 km")
- frequency should be described as FLIGHTS (e.g., "churned customers took 10 flights vs 33 flights")
- clv should be described in CAD (e.g., "churned customers had CLV of CAD 2,500 vs CAD 8,000")

STRUCTURE YOUR RESPONSE:
1. **The Answer**: Direct response to the question with key numbers
2. **The Evidence**: 2-3 supporting data points that justify the answer
3. **The Recommendation**: One clear, specific action to take

Do NOT:
- Use bullet points excessively
- List raw data without interpretation
- Give vague recommendations like "consider" or "may want to"
- Repeat the question back
- Use the $ symbol anywhere in your response"""


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
    df["is_cancelled"] = df["is_cancelled"].astype(bool)
    for c in [
        "clv",
        "churn_score",
        "recency",
        "tenure_months",
        "frequency",
        "monetary",
    ]:
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
            title="RFM Segment",
            axis=alt.Axis(labelAngle=-45, labelBaseline="top", labelOverlap=False),
            scale=alt.Scale(domain=RFM_DOMAIN),
        ),
        y=alt.Y(
            "customers:Q",
            title="Number of Customers",
        ),
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
            alt.Tooltip("customers:Q", format=",", title="Customers"),
            alt.Tooltip("avg_clv:Q", format="$,.0f", title="Avg CLV"),
            alt.Tooltip("avg_churn_score:Q", format=".0f", title="Avg Churn Score"),
            alt.Tooltip(
                "avg_recency_days:Q", format=",.0f", title="Avg Recency (Days)"
            ),
            alt.Tooltip("avg_frequency:Q", format=",.0f", title="Avg Frequency"),
            alt.Tooltip("avg_monetary:Q", format=",.0f", title="Avg Monetary"),
            alt.Tooltip(
                "avg_tenure_months:Q", format=",.0f", title="Avg Tenure (Months)"
            ),
            alt.Tooltip("is_cancelled:Q", format=".0%", title="Percent Cancelled"),
        ],
    )
)

chart2 = (
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
        y=alt.Y(
            "customers:Q",
            title="Number of Customers",
        ),
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
            alt.Tooltip("customers:Q", format=",", title="Customers"),
            alt.Tooltip("avg_clv:Q", format="$,.0f", title="Avg CLV"),
            alt.Tooltip("avg_churn_score:Q", format=".0f", title="Avg Churn Score"),
            alt.Tooltip(
                "avg_recency_days:Q", format=",.0f", title="Avg Recency (Days)"
            ),
            alt.Tooltip("avg_frequency:Q", format=",.0f", title="Avg Frequency"),
            alt.Tooltip("avg_monetary:Q", format=",.0f", title="Avg Monetary"),
            alt.Tooltip(
                "avg_tenure_months:Q", format=",.0f", title="Avg Tenure (Months)"
            ),
            alt.Tooltip("is_cancelled:Q", format=".0%", title="Percent Cancelled"),
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
            color=alt.Color("gender:N", legend=alt.Legend(title="Gender")),
            tooltip=[
                alt.Tooltip("gender:N", title="Gender"),
                alt.Tooltip("count:Q", format=",", title="Customers"),
                alt.Tooltip("percent:Q", format=".1%", title="Percent"),
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
            color=alt.Color("loyalty_card:N", legend=alt.Legend(title="Loyalty Card")),
            tooltip=[
                alt.Tooltip("loyalty_card:N", title="Loyalty Card"),
                alt.Tooltip("count:Q", format=",", title="Customers"),
                alt.Tooltip("percent:Q", format=".1%", title="Percent"),
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


def _extract_json(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return text[start : end + 1]


ALLOWED_OPS = {
    "kpis_baseline",
    "kpis_slice",
    "summary_by_segment",
    "summary_by_province",
    "summary_by_card",
    "summary_by_gender",
    "top_risk_customers",
    "top_value_customers",
    "value_at_risk_by_segment",
    "value_at_risk_by_province",
    "churn_by_clv_tier",
    "do_nothing_scenario",
    "single_priority_initiative",
    "segment_comparison",
    "tenure_analysis",
    "revenue_impact",
    "correlation_drivers",
}

SUMMARY_OPS = {
    "summary_by_segment": "rfm_segment",
    "summary_by_province": "province",
    "summary_by_card": "loyalty_card",
    "summary_by_gender": "gender",
}

VALUE_AT_RISK_OPS = {"value_at_risk_by_segment", "value_at_risk_by_province"}

TOP_RISK_COLS = [
    "loyalty_number",
    "province",
    "loyalty_card",
    "clv",
    "rfm_segment",
    "churn_score",
]

TOP_VALUE_COLS = [
    "loyalty_number",
    "province",
    "loyalty_card",
    "clv",
    "rfm_segment",
    "frequency",
    "tenure_months",
    "is_cancelled",
]


def compute_operation(
    op_item: dict, df_all: pd.DataFrame, df_slice: pd.DataFrame, baseline_kpis: dict
):
    op = op_item.get("op")

    if op == "kpis_baseline":
        return baseline_kpis

    if op == "kpis_slice":
        clv_vals = pd.to_numeric(df_slice["clv"], errors="coerce").fillna(0)
        clv_75th = clv_vals.quantile(0.75) if len(clv_vals) > 0 else 0
        high_value_count = (clv_vals > clv_75th).sum() if clv_75th > 0 else 0
        cancelled_rate = float(df_slice["is_cancelled"].mean() or 0)
        return {
            "customers": int(df_slice["loyalty_number"].nunique()),
            "avg_clv": float(clv_vals.mean()),
            "median_clv": float(clv_vals.median()),
            "cancelled_rate": cancelled_rate,
            "retention_rate": 1 - cancelled_rate,
            "avg_recency_days": float((df_slice["recency"].mean() or 0) * 30),
            "avg_frequency": float(df_slice["frequency"].mean() or 0),
            "avg_monetary": float(df_slice["monetary"].mean() or 0),
            "avg_tenure_months": float(df_slice["tenure_months"].mean() or 0),
            "avg_churn_score": float(df_slice["churn_score"].mean() or 0),
            "high_value_customers": int(high_value_count),
            "high_value_pct": float(high_value_count / len(df_slice))
            if len(df_slice) > 0
            else 0,
        }

    if op in SUMMARY_OPS:
        group_col = SUMMARY_OPS[op]
        out = (
            df_slice.groupby(group_col)
            .agg(
                customers=("loyalty_number", "nunique"),
                avg_clv=("clv", "mean"),
                median_clv=("clv", "median"),
                avg_churn_score=("churn_score", "mean"),
                avg_recency_days=("recency", lambda s: (s.mean() or 0) * 30),
                avg_frequency=("frequency", "mean"),
                avg_monetary=("monetary", "mean"),
                avg_tenure_months=("tenure_months", "mean"),
                cancelled_rate=("is_cancelled", "mean"),
            )
            .reset_index()
        )

        out["retention_rate"] = 1 - out["cancelled_rate"]

        sort_by = op_item.get("sort_by", "avg_churn_score")
        if sort_by not in out.columns:
            sort_by = "avg_churn_score"
        ascending = bool(op_item.get("ascending", False))
        top_n = max(1, min(int(op_item.get("top_n", 10)), 50))

        out = out.sort_values(sort_by, ascending=ascending).head(top_n)

        for c in out.columns:
            if c != group_col:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

        return out.to_dict("records")

    if op == "top_risk_customers":
        top_n = max(1, min(int(op_item.get("top_n", 25)), 100))
        out = (
            df_slice.sort_values("churn_score", ascending=False)[TOP_RISK_COLS]
            .head(top_n)
            .copy()
        )
        for c in ["clv", "churn_score"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
        return out.to_dict("records")

    if op == "top_value_customers":
        top_n = max(1, min(int(op_item.get("top_n", 25)), 100))
        segment_filter = op_item.get("segment_filter")

        data = df_slice.copy()
        if segment_filter and segment_filter in data["rfm_segment"].values:
            data = data[data["rfm_segment"] == segment_filter]

        out = (
            data.sort_values("clv", ascending=False)[TOP_VALUE_COLS].head(top_n).copy()
        )

        for c in ["clv", "frequency", "tenure_months"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

        return out.to_dict("records")

    if op == "value_at_risk_by_segment":
        group_col = "rfm_segment"
        out = (
            df_slice.groupby(group_col)
            .agg(
                total_value_at_risk=("churn_score", "sum"),
                total_clv=("clv", "sum"),
                customers=("loyalty_number", "nunique"),
                avg_churn_score=("churn_score", "mean"),
                avg_clv=("clv", "mean"),
                median_clv=("clv", "median"),
                cancelled_rate=("is_cancelled", "mean"),
            )
            .reset_index()
        )
        out["retention_rate"] = 1 - out["cancelled_rate"]
        sort_by = op_item.get("sort_by", "total_value_at_risk")
        if sort_by not in out.columns:
            sort_by = "total_value_at_risk"
        ascending = bool(op_item.get("ascending", False))
        top_n = max(1, min(int(op_item.get("top_n", 10)), 50))

        out = out.sort_values(sort_by, ascending=ascending).head(top_n)
        for c in out.columns:
            if c != group_col:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
        return out.to_dict("records")

    if op == "value_at_risk_by_province":
        group_col = "province"
        out = (
            df_slice.groupby(group_col)
            .agg(
                total_value_at_risk=("churn_score", "sum"),
                total_clv=("clv", "sum"),
                customers=("loyalty_number", "nunique"),
                avg_churn_score=("churn_score", "mean"),
                avg_clv=("clv", "mean"),
                median_clv=("clv", "median"),
                cancelled_rate=("is_cancelled", "mean"),
            )
            .reset_index()
        )
        out["retention_rate"] = 1 - out["cancelled_rate"]
        sort_by = op_item.get("sort_by", "total_value_at_risk")
        if sort_by not in out.columns:
            sort_by = "total_value_at_risk"
        ascending = bool(op_item.get("ascending", False))
        top_n = max(1, min(int(op_item.get("top_n", 10)), 50))

        out = out.sort_values(sort_by, ascending=ascending).head(top_n)
        for c in out.columns:
            if c != group_col:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
        return out.to_dict("records")

    if op == "churn_by_clv_tier":
        slice_copy = df_slice.copy()
        clv_vals = pd.to_numeric(slice_copy["clv"], errors="coerce").fillna(0)
        try:
            slice_copy["clv_tier"] = pd.qcut(
                clv_vals, q=3, labels=["Low", "Medium", "High"], duplicates="drop"
            )
        except (ValueError, TypeError):
            slice_copy["clv_tier"] = "All"

        out = (
            slice_copy.groupby("clv_tier", observed=True)
            .agg(
                customers=("loyalty_number", "nunique"),
                cancelled_rate=("is_cancelled", "mean"),
                avg_clv=("clv", "mean"),
                avg_churn_score=("churn_score", "mean"),
                total_clv=("clv", "sum"),
            )
            .reset_index()
        )
        for c in out.columns:
            if c != "clv_tier":
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
        return out.to_dict("records")

    if op == "do_nothing_scenario":
        customers = int(df_slice["loyalty_number"].nunique())
        cancelled_rate = float(df_slice["is_cancelled"].mean() or 0)
        total_value_at_risk = float(df_slice["churn_score"].sum() or 0)
        total_clv = float(df_slice["clv"].sum() or 0)
        projected_customers_at_risk = customers * cancelled_rate
        return {
            "customers": customers,
            "cancelled_rate": cancelled_rate,
            "total_value_at_risk": total_value_at_risk,
            "total_clv": total_clv,
            "projected_customers_at_risk": projected_customers_at_risk,
        }

    if op == "single_priority_initiative":
        seg = (
            df_slice.groupby("rfm_segment")
            .agg(
                total_value_at_risk=("churn_score", "sum"),
                total_clv=("clv", "sum"),
                customers=("loyalty_number", "nunique"),
                avg_churn_score=("churn_score", "mean"),
                avg_clv=("clv", "mean"),
            )
            .reset_index()
        )
        prov = (
            df_slice.groupby("province")
            .agg(
                total_value_at_risk=("churn_score", "sum"),
                total_clv=("clv", "sum"),
                customers=("loyalty_number", "nunique"),
                avg_churn_score=("churn_score", "mean"),
                avg_clv=("clv", "mean"),
            )
            .reset_index()
        )
        top_seg = (
            seg.sort_values("total_value_at_risk", ascending=False)
            .head(1)
            .to_dict("records")
        )
        top_prov = (
            prov.sort_values("total_value_at_risk", ascending=False)
            .head(1)
            .to_dict("records")
        )
        return {
            "top_segment": top_seg[0] if top_seg else {},
            "top_province": top_prov[0] if top_prov else {},
        }

    if op == "segment_comparison":
        segment_a = op_item.get("segment_a", "Champions")
        segment_b = op_item.get("segment_b", "At Risk")

        valid_segments = set(df_slice["rfm_segment"].dropna().unique())
        if segment_a not in valid_segments:
            segment_a = (
                "Champions"
                if "Champions" in valid_segments
                else list(valid_segments)[0]
            )
        if segment_b not in valid_segments:
            segment_b = (
                "At Risk" if "At Risk" in valid_segments else list(valid_segments)[-1]
            )

        def segment_stats(seg_name):
            seg_data = df_slice[df_slice["rfm_segment"] == seg_name]
            if seg_data.empty:
                return {"segment": seg_name, "customers": 0}
            clv_vals = pd.to_numeric(seg_data["clv"], errors="coerce").fillna(0)
            return {
                "segment": seg_name,
                "customers": int(seg_data["loyalty_number"].nunique()),
                "avg_clv": float(clv_vals.mean()),
                "median_clv": float(clv_vals.median()),
                "total_clv": float(clv_vals.sum()),
                "avg_churn_score": float(seg_data["churn_score"].mean() or 0),
                "cancelled_rate": float(seg_data["is_cancelled"].mean() or 0),
                "avg_recency_days": float((seg_data["recency"].mean() or 0) * 30),
                "avg_frequency": float(seg_data["frequency"].mean() or 0),
                "avg_tenure_months": float(seg_data["tenure_months"].mean() or 0),
            }

        return {
            "segment_a": segment_stats(segment_a),
            "segment_b": segment_stats(segment_b),
        }

    if op == "tenure_analysis":
        slice_copy = df_slice.copy()
        tenure_vals = pd.to_numeric(
            slice_copy["tenure_months"], errors="coerce"
        ).fillna(0)

        bins = [0, 6, 12, 24, 36, 60, float("inf")]
        labels = [
            "0-6 months",
            "7-12 months",
            "13-24 months",
            "25-36 months",
            "37-60 months",
            "60+ months",
        ]

        try:
            slice_copy["tenure_bucket"] = pd.cut(
                tenure_vals, bins=bins, labels=labels, include_lowest=True
            )
        except (ValueError, TypeError):
            slice_copy["tenure_bucket"] = "All"

        out = (
            slice_copy.groupby("tenure_bucket", observed=True)
            .agg(
                customers=("loyalty_number", "nunique"),
                cancelled_rate=("is_cancelled", "mean"),
                avg_clv=("clv", "mean"),
                avg_churn_score=("churn_score", "mean"),
                total_value_at_risk=("churn_score", "sum"),
            )
            .reset_index()
        )

        for c in out.columns:
            if c != "tenure_bucket":
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

        out = out.sort_values("cancelled_rate", ascending=False)
        return out.to_dict("records")

    if op == "revenue_impact":
        total_customers = int(df_slice["loyalty_number"].nunique())
        cancelled_rate = float(df_slice["is_cancelled"].mean() or 0)
        avg_clv = float(df_slice["clv"].mean() or 0)
        total_clv = float(df_slice["clv"].sum() or 0)
        total_value_at_risk = float(df_slice["churn_score"].sum() or 0)

        projected_churned_customers = total_customers * cancelled_rate
        projected_clv_loss = projected_churned_customers * avg_clv

        avg_tenure_years = max(1, (df_slice["tenure_months"].mean() or 12) / 12)
        annual_revenue_per_customer = avg_clv / avg_tenure_years
        projected_annual_loss = (
            projected_churned_customers * annual_revenue_per_customer
        )

        high_risk = df_slice[
            df_slice["churn_score"] > df_slice["churn_score"].quantile(0.75)
        ]
        high_risk_value = float(high_risk["churn_score"].sum() or 0)
        high_risk_count = int(high_risk["loyalty_number"].nunique())

        return {
            "total_customers": total_customers,
            "cancelled_rate": cancelled_rate,
            "avg_clv": avg_clv,
            "total_clv": total_clv,
            "total_value_at_risk": total_value_at_risk,
            "projected_churned_customers": projected_churned_customers,
            "projected_clv_loss": projected_clv_loss,
            "projected_annual_loss": projected_annual_loss,
            "high_risk_customers": high_risk_count,
            "high_risk_value_at_risk": high_risk_value,
        }

    if op == "correlation_drivers":
        churned = df_slice[df_slice["is_cancelled"] == 1]
        retained = df_slice[df_slice["is_cancelled"] == 0]

        churned_count = int(churned["loyalty_number"].nunique())
        retained_count = int(retained["loyalty_number"].nunique())

        metrics = ["recency", "frequency", "monetary", "tenure_months", "clv"]
        comparison = {}

        for m in metrics:
            churned_avg = float(churned[m].mean() or 0)
            retained_avg = float(retained[m].mean() or 0)

            if retained_avg != 0:
                diff_pct = ((churned_avg - retained_avg) / abs(retained_avg)) * 100
            else:
                diff_pct = 0.0

            comparison[m] = {
                "churned_avg": churned_avg,
                "retained_avg": retained_avg,
                "difference_pct": round(diff_pct, 1),
            }

        churned_segments = (
            churned.groupby("rfm_segment")["loyalty_number"].nunique().to_dict()
        )
        retained_segments = (
            retained.groupby("rfm_segment")["loyalty_number"].nunique().to_dict()
        )

        return {
            "churned_count": churned_count,
            "retained_count": retained_count,
            "metrics_comparison": comparison,
            "churned_segment_distribution": churned_segments,
            "retained_segment_distribution": retained_segments,
        }

    raise ValueError(f"Unknown op: {op}")


def build_default_plan():
    return {
        "intent": "Answer using baseline vs slice, identify highest priority segment and province by value at risk, and do-nothing impact.",
        "operations": [
            {"op": "kpis_baseline"},
            {"op": "kpis_slice"},
            {"op": "do_nothing_scenario"},
            {
                "op": "value_at_risk_by_segment",
                "top_n": 5,
                "sort_by": "total_value_at_risk",
                "ascending": False,
            },
            {
                "op": "value_at_risk_by_province",
                "top_n": 5,
                "sort_by": "total_value_at_risk",
                "ascending": False,
            },
            {
                "op": "summary_by_segment",
                "top_n": 5,
                "sort_by": "avg_churn_score",
                "ascending": False,
            },
            {
                "op": "summary_by_province",
                "top_n": 5,
                "sort_by": "avg_churn_score",
                "ascending": False,
            },
            {"op": "top_risk_customers", "top_n": 15},
        ],
    }


def _sanitize_plan(plan: dict) -> dict:
    if (
        not isinstance(plan, dict)
        or "operations" not in plan
        or not isinstance(plan["operations"], list)
    ):
        return build_default_plan()

    ops = []
    seen = set()

    for item in plan["operations"]:
        op = (item or {}).get("op")
        if op not in ALLOWED_OPS or op in seen:
            continue

        clean = {"op": op}

        if op in SUMMARY_OPS:
            clean.update(
                {
                    "top_n": max(1, min(int(item.get("top_n", 10)), 50)),
                    "sort_by": item.get("sort_by", "avg_churn_score"),
                    "ascending": bool(item.get("ascending", False)),
                }
            )

        elif op == "top_risk_customers":
            clean.update({"top_n": max(1, min(int(item.get("top_n", 25)), 100))})

        elif op in VALUE_AT_RISK_OPS:
            clean.update(
                {
                    "top_n": max(1, min(int(item.get("top_n", 10)), 50)),
                    "sort_by": item.get("sort_by", "total_value_at_risk"),
                    "ascending": bool(item.get("ascending", False)),
                }
            )

        elif op == "segment_comparison":
            segment_a = str(item.get("segment_a", "Champions")).strip()
            segment_b = str(item.get("segment_b", "At Risk")).strip()
            clean.update({"segment_a": segment_a, "segment_b": segment_b})

        elif op == "top_value_customers":
            top_n = max(1, min(int(item.get("top_n", 25)), 100))
            clean.update({"top_n": top_n})
            segment_filter = item.get("segment_filter")
            if segment_filter:
                clean["segment_filter"] = str(segment_filter).strip()

        ops.append(clean)
        seen.add(op)

    if not ops:
        return build_default_plan()

    return {
        "intent": str(plan.get("intent", "")).strip()
        or "Answer the executive question using computed summaries.",
        "operations": ops,
    }


if "chat" not in st.session_state:
    st.session_state.chat = [
        {
            "role": "assistant",
            "content": "Hi! How can I help you today?",
        }
    ]

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Try: Where should we focus retention efforts?")
if q:
    st.session_state.chat.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    planner_prompt = f"""
Executive question: {q}

Current filter state:
- provinces: {selected_provinces}
- segments: {selected_segments}
- gender: {selected_gender}
- cards: {selected_cards}

Available columns:
loyalty_number, gender, province, loyalty_card, clv, rfm_segment, churn_score,
recency, frequency, monetary, tenure_months, is_cancelled
""".strip()

    planner_error = None
    try:
        plan_raw = invoke_claude(
            user_prompt=planner_prompt,
            system_prompt=PLANNER_SYSTEM,
            max_tokens=450,
            temperature=0.0,
        )
        plan = _sanitize_plan(json.loads(_extract_json(plan_raw)))
    except Exception as e:
        planner_error = str(e)
        plan = build_default_plan()

    computed = {
        "kpis_baseline": baseline,
        "kpis_slice": {
            "customers": int(df_base["loyalty_number"].nunique()),
            "avg_clv": float(df_base["clv"].mean() or 0),
            "cancelled_rate": float(df_base["is_cancelled"].mean() or 0),
            "avg_recency_days": float((df_base["recency"].mean() or 0) * 30),
            "avg_frequency": float(df_base["frequency"].mean() or 0),
            "avg_monetary": float(df_base["monetary"].mean() or 0),
            "avg_tenure_months": float(df_base["tenure_months"].mean() or 0),
            "avg_churn_score": float(df_base["churn_score"].mean() or 0),
        },
    }

    compute_errors = []
    for item in plan["operations"]:
        op = item["op"]
        if op in computed:
            continue
        try:
            computed[op] = compute_operation(item, df, df_base, baseline)
        except Exception as e:
            compute_errors.append(f"{op}: {e}")

    with st.expander("How I pulled the data (agent plan)", expanded=False):
        st.write(plan)
        if planner_error:
            st.caption(f"Planner fallback used: {planner_error}")
        if compute_errors:
            st.caption("Some summaries failed to compute:")
            st.write(compute_errors)

    narrator_prompt = f"""
Executive question: {q}

Computed summaries JSON:
{json.dumps(computed, indent=2)}
""".strip()

    try:
        ans = invoke_claude(
            user_prompt=narrator_prompt,
            system_prompt=NARRATOR_SYSTEM,
            max_tokens=900,
            temperature=0.2,
        )
    except Exception as e:
        ans = f"Agent call failed: {e}"

    st.session_state.chat.append({"role": "assistant", "content": ans})
    with st.chat_message("assistant"):
        st.markdown(ans)
