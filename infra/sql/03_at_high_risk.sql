CREATE OR REPLACE VIEW airline_analytics.vw_at_risk_high_value AS
SELECT
    loyalty_number,
    clv,
    loyalty_card,
    recency,
    frequency,
    monetary
FROM airline_analytics.customer_features
WHERE rfm_segment = 'At Risk'
ORDER BY clv DESC;
