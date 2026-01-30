CREATE OR REPLACE VIEW airline_analytics.customer_scored AS
SELECT
    f.loyalty_number,
    f.country,
    f.province,
    f.city,
    f.gender,
    f.education,
    f.loyalty_card,
    f.clv,
    f.is_cancelled,
    f.tenure_months,
    f.recency,
    f.r_score,
    f.frequency,
    f.f_score,
    f.monetary,
    f.m_score,
    f.rfm_segment,
    p.priority_score AS churn_score
FROM airline_analytics.customer_features AS f
LEFT JOIN airline_analytics.churn_predictions AS p
    ON f.loyalty_number = p.loyalty_number;
