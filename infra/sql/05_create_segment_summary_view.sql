CREATE OR REPLACE VIEW airline_analytics.vw_segment_summary AS
SELECT
    rfm_segment,
    COUNT(*) AS customers,
    ROUND(AVG(clv)) AS avg_clv,
    ROUND(AVG(churn_score)) AS avg_churn_score,
    ROUND(AVG(recency) * 30) AS avg_recency_days
FROM airline_analytics.customer_scored
GROUP BY rfm_segment
ORDER BY avg_churn_score DESC;
