CREATE OR REPLACE VIEW airline_analytics.vw_kpi_overview AS
SELECT
    COUNT(DISTINCT loyalty_number) AS total_customers,
    ROUND(AVG(clv)) AS avg_clv,
    ROUND(AVG(CASE WHEN is_cancelled THEN 1 ELSE 0 END), 2) AS cancelled_rate,
    ROUND(AVG(recency) * 30) AS avg_recency_days,
    ROUND(AVG(tenure_months)) AS avg_tenure_months
FROM airline_analytics.customer_scored;
