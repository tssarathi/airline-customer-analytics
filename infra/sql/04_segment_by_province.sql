CREATE OR REPLACE VIEW airline_analytics.vw_segment_by_province AS
SELECT
    province,
    rfm_segment,
    COUNT(*) AS customers
FROM airline_analytics.customer_features
GROUP BY 1, 2;
