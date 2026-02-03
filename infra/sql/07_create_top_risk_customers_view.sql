CREATE OR REPLACE VIEW airline_analytics.vw_top_risk_customers AS
SELECT
    loyalty_number,
    province,
    loyalty_card,
    clv,
    rfm_segment,
    ROUND(churn_score) AS churn_score
FROM airline_analytics.customer_scored
ORDER BY churn_score DESC
LIMIT 50;
