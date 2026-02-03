CREATE OR REPLACE VIEW airline_analytics.vw_province_summary AS
SELECT
    province,
    COUNT(*) AS customers,
    ROUND(AVG(clv)) AS avg_clv,
    ROUND(AVG(churn_score)) AS avg_churn_score,
    ROUND(AVG(recency) * 30) AS avg_recency_days
FROM airline_analytics.customer_scored
GROUP BY province
ORDER BY avg_churn_score DESC;
