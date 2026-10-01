/* ============================================================================
   02_bi_analysis.sql
   NEW YORK QUARTERLY ED DEMAND - ANALYST / POWER BI QUERIES

   PREREQUISITE
   ------------
   Run 01_build_county_quarter_analysis.sql first.

   PRIMARY BI SOURCE
   -----------------
   BI_COUNTY_QUARTER_METRICS

   WHY THIS FILE IS SEPARATE
   -------------------------
   01_build...sql demonstrates data preparation / transformation.
   02_bi_analysis.sql demonstrates analyst-style querying and provides
   validation queries for the Power BI dashboard.

   Run queries one at a time with Ctrl+Enter.
   ============================================================================ */


/* QUERY 1 - Statewide quarterly ED trend
   Suggested Power BI visual: line chart. */
SELECT
    year,
    quarter,
    quarter_num,
    SUM(total_ed_encounters) AS statewide_ed_encounters
FROM BI_COUNTY_QUARTER_METRICS
GROUP BY year, quarter, quarter_num
ORDER BY year, quarter_num;


/* QUERY 2 - Top 10 counties by average ED encounters per 100K
   Suggested visual: ranked horizontal bar chart. */
SELECT
    county_name,
    AVG(facility_ed_encounters_per_100k_proxy) AS avg_ed_per_100k
FROM BI_COUNTY_QUARTER_METRICS
GROUP BY county_name
ORDER BY avg_ed_per_100k DESC
FETCH FIRST 10 ROWS ONLY;


/* QUERY 3 - Region comparison
   Suggested visual: clustered bar chart or KPI table. */
SELECT
    region,
    AVG(total_ed_encounters) AS avg_quarterly_ed_encounters,
    AVG(facility_ed_encounters_per_100k_proxy) AS avg_ed_per_100k,
    AVG(acs_lag2_median_income) AS avg_median_income,
    AVG(acs_lag2_poverty_rate) AS avg_poverty_rate
FROM BI_COUNTY_QUARTER_METRICS
GROUP BY region
ORDER BY avg_quarterly_ed_encounters DESC;


/* QUERY 4 - Largest quarter-over-quarter increases
   Suggested visual: "counties to watch" table. */
SELECT
    year,
    quarter,
    county_name,
    total_ed_encounters,
    target_lag1 AS previous_quarter_ed,
    qoq_change,
    qoq_pct_change
FROM BI_COUNTY_QUARTER_METRICS
WHERE qoq_pct_change IS NOT NULL
ORDER BY qoq_pct_change DESC
FETCH FIRST 20 ROWS ONLY;


/* QUERY 5 - Largest year-over-year increases. */
SELECT
    year,
    quarter,
    county_name,
    total_ed_encounters,
    target_lag4 AS prior_year_same_quarter_ed,
    yoy_change,
    yoy_pct_change
FROM BI_COUNTY_QUARTER_METRICS
WHERE yoy_pct_change IS NOT NULL
ORDER BY yoy_pct_change DESC
FETCH FIRST 20 ROWS ONLY;


/* QUERY 6 - County ranking for one quarter
   Change the year / quarter to inspect another snapshot. */
SELECT
    year,
    quarter,
    county_rank_ed_per_100k,
    county_name,
    total_ed_encounters,
    facility_ed_encounters_per_100k_proxy
FROM BI_COUNTY_QUARTER_METRICS
WHERE year = 2024
  AND quarter = 'Q4'
ORDER BY county_rank_ed_per_100k;


/* QUERY 7 - County detail trend
   Change Albany to another county name. */
   
--SELECT DISTINCT county_name
--FROM BI_COUNTY_QUARTER_METRICS
--WHERE UPPER(county_name) LIKE '%ALBANY%';

SELECT
    year,
    quarter,
    quarter_num,
    county_name,
    total_ed_encounters,
    facility_ed_encounters_per_100k_proxy,
    qoq_pct_change,
    yoy_pct_change,
    acs_lag2_population,
    acs_lag2_median_income,
    acs_lag2_poverty_rate,
    acs_lag2_pct_65plus,
    weather_lag4_tavg_mean,
    weather_lag4_precip_total,
    weather_lag4_snowfall_total
FROM BI_COUNTY_QUARTER_METRICS
WHERE UPPER(county_name) LIKE '%ALBANY%'
ORDER BY year, quarter_num;


/* QUERY 8 - Income vs ED utilization
   Suggested Power BI visual: scatter plot.
   X = average median income
   Y = average ED encounters per 100K
   Details = county
   Tooltip = poverty rate */
SELECT
    county_name,
    AVG(acs_lag2_median_income) AS avg_median_income,
    AVG(facility_ed_encounters_per_100k_proxy) AS avg_ed_per_100k,
    AVG(acs_lag2_poverty_rate) AS avg_poverty_rate
FROM BI_COUNTY_QUARTER_METRICS
GROUP BY county_name
ORDER BY county_name;


/* QUERY 9 - Quarterly region trend
   Useful for a region-by-time visual. */
SELECT
    year,
    quarter,
    quarter_num,
    region,
    SUM(total_ed_encounters) AS total_ed_encounters,
    AVG(facility_ed_encounters_per_100k_proxy) AS avg_ed_per_100k
FROM BI_COUNTY_QUARTER_METRICS
GROUP BY year, quarter, quarter_num, region
ORDER BY year, quarter_num, region;


/* QUERY 10 - Power BI source preview. */
SELECT *
FROM BI_COUNTY_QUARTER_METRICS
ORDER BY year, quarter_num, fips
FETCH FIRST 100 ROWS ONLY;


/* ============================================================================
   RECOMMENDED POWER BI RESPONSIBILITY

   Keep stable/reusable business logic in SQL:
   - cleaned fields
   - joins / grain
   - lag features
   - per-100K metric
   - QoQ / YoY base metrics
   - region labels

   Use DAX for interactive/filter-dependent measures:
   - selected-period total ED encounters
   - selected-county ED encounters
   - slicer-aware averages
   - dynamic Top N
   - dashboard KPI cards

   Suggested pages:
   1. Executive Overview
      Total ED | ED/100K | QoQ | YoY | trend | map | Top 10 | region
   2. County Analysis
      county slicer | demand trend | population | income | poverty | age 65+
   3. Forecast & Planning
      actual vs forecast | selected forecast | WAPE | highest expected demand
   ============================================================================ */
