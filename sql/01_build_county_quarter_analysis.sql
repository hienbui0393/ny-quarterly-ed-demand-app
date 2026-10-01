/* ============================================================================
   01_build_county_quarter_analysis.sql
   NEW YORK QUARTERLY ED DEMAND - ORACLE SQL DATA PREPARATION

   PURPOSE
   -------
   This file replaces the Python/Colab cleaning, aggregation, joining, and
   feature-engineering stages from the original project.

   SQL RESPONSIBILITIES
   --------------------
   1. Clean raw SPARCS ED fields.
   2. Aggregate facility-quarter data to county-quarter.
   3. Clean / derive Census ACS variables.
   4. Aggregate daily weather to county-quarter.
   5. Join ED + geography + ACS + weather.
   6. Create exact demand lags and modeling features.
   7. Create a BI-ready metrics view.
   8. Run data-quality checks.
   9. Export FINAL_COUNTY_QUARTER_ANALYSIS for Python ML.

   PYTHON RESPONSIBILITIES AFTER THIS FILE
   ---------------------------------------
   - EDA / visualization
   - feature selection
   - Ridge / Random Forest / XGBoost
   - persistence benchmarks
   - rolling validation and final holdout evaluation
   - model artifact export

   RAW TABLES EXPECTED IN ORACLE
   -----------------------------
   RAW_SPARCS_ED_FACILITY_QUARTER
   RAW_CENSUS_ACS_COUNTY
   RAW_NY_COUNTY_GEO
   RAW_WEATHER_DAILY_BY_COUNTY

   RAW_SPARCS_PPV_BY_COUNTY may remain in Oracle, but it is not used in the
   current modeling table because the original final ML panel contains no
   PPV-derived feature.

   FINAL GRAIN
   -----------
   One row = one facility county + one calendar quarter.

   HOW TO RUN
   ----------
   PART A: Run all CREATE VIEW statements together with F5 / Run Script.
   PART B: Run validation SELECT statements one at a time with Ctrl+Enter.
   PART C: Run the export SELECT separately and export its result to:
           data/processed/county_quarter_analysis.csv
   ============================================================================ */


/* ============================================================================
   PART A - BUILD VIEWS
   Run this whole PART A with F5 / Run Script.
   ============================================================================ */


/* ============================================================================
   STEP 1 - CLEAN SPARCS ED DATA

   - Remove records without county FIPS.
   - Keep Q1-Q4 only; annual CY totals are excluded.
   - Convert numeric text to NUMBER.
   - Suppressed/non-numeric values such as "S" become NULL.
   ============================================================================ */

CREATE OR REPLACE VIEW STG_SPARCS_ED AS
SELECT
    year,
    quarter,
    facility_id,
    facility_name,

    CASE
        WHEN REGEXP_LIKE(TRIM(total_ed_encounters), '^[0-9]+$')
        THEN TO_NUMBER(total_ed_encounters)
    END AS total_ed_encounters,

    CASE
        WHEN REGEXP_LIKE(TRIM(treat_and_release_ed), '^[0-9]+$')
        THEN TO_NUMBER(treat_and_release_ed)
    END AS treat_and_release_ed,

    CASE
        WHEN REGEXP_LIKE(TRIM(ed_encounters_requiring), '^[0-9]+$')
        THEN TO_NUMBER(ed_encounters_requiring)
    END AS ed_encounters_requiring,

    CASE
        WHEN REGEXP_LIKE(TRIM(ed_encounters_admitted_as), '^[0-9]+$')
        THEN TO_NUMBER(ed_encounters_admitted_as)
    END AS ed_encounters_admitted_as,

    LPAD(TRIM(facility_county_fips), 5, '0') AS fips

FROM RAW_SPARCS_ED_FACILITY_QUARTER
WHERE facility_county_fips IS NOT NULL
  AND quarter IN ('Q1', 'Q2', 'Q3', 'Q4');


/* ============================================================================
   STEP 2 - AGGREGATE ED TO COUNTY-QUARTER

   Source grain: facility + quarter
   New grain:    county + quarter

   period_index makes exact calendar-quarter joins explicit.
   Example: 2018 Q1 = 2018 * 4 + 1 = 8073.

   QA columns retain information about suppressed/missing source totals.
   ============================================================================ */

CREATE OR REPLACE VIEW INT_ED_COUNTY_QUARTER AS
SELECT
    fips,
    year,
    quarter,

    CASE quarter
        WHEN 'Q1' THEN 1
        WHEN 'Q2' THEN 2
        WHEN 'Q3' THEN 3
        WHEN 'Q4' THEN 4
    END AS quarter_num,

    year * 4 +
    CASE quarter
        WHEN 'Q1' THEN 1
        WHEN 'Q2' THEN 2
        WHEN 'Q3' THEN 3
        WHEN 'Q4' THEN 4
    END AS period_index,

    /* The original Python pipeline dropped rows with a missing target before
       aggregating encounter components. Preserve that behavior here. */
    SUM(total_ed_encounters) AS total_ed_encounters,

    SUM(CASE
        WHEN total_ed_encounters IS NOT NULL THEN treat_and_release_ed
    END) AS treat_and_release_ed,

    SUM(CASE
        WHEN total_ed_encounters IS NOT NULL THEN ed_encounters_requiring
    END) AS ed_encounters_requiring,

    SUM(CASE
        WHEN total_ed_encounters IS NOT NULL THEN ed_encounters_admitted_as
    END) AS ed_encounters_admitted_as,

    /* Facility count was calculated before dropping suppressed target rows. */
    COUNT(DISTINCT facility_id) AS facility_count,

    COUNT(*) AS facility_row_count,
    COUNT(total_ed_encounters) AS facilities_with_ed_data,
    COUNT(*) - COUNT(total_ed_encounters) AS facilities_missing_ed_data,

    CASE
        WHEN COUNT(CASE
            WHEN total_ed_encounters IS NOT NULL
             AND (treat_and_release_ed IS NOT NULL
               OR ed_encounters_requiring IS NOT NULL
               OR ed_encounters_admitted_as IS NOT NULL)
            THEN 1
        END) > 0
        THEN
            SUM(total_ed_encounters)
            - (
                NVL(SUM(CASE
                    WHEN total_ed_encounters IS NOT NULL THEN treat_and_release_ed
                END), 0)
                + NVL(SUM(CASE
                    WHEN total_ed_encounters IS NOT NULL THEN ed_encounters_requiring
                END), 0)
                + NVL(SUM(CASE
                    WHEN total_ed_encounters IS NOT NULL THEN ed_encounters_admitted_as
                END), 0)
              )
    END AS component_gap

FROM STG_SPARCS_ED
GROUP BY
    fips,
    year,
    quarter
HAVING COUNT(total_ed_encounters) > 0;


/* ============================================================================
   STEP 3 - CLEAN / ENGINEER CENSUS ACS

   FIPS example:
   state 36 + county 001 = 36001

   Derived fields:
   - population
   - median household income
   - poverty rate
   - % under age 5
   - % age 65+
   ============================================================================ */

CREATE OR REPLACE VIEW STG_CENSUS_ACS AS
SELECT
    fips,
    year,
    population,
    median_income,

    100.0 * pov_below
        / NULLIF(pov_total, 0) AS poverty_rate,

    100.0 * (
        male_under5 + female_under5
    ) / NULLIF(age_total, 0) AS pct_under5,

    100.0 * (
        NVL(age65_1, 0) + NVL(age65_2, 0) + NVL(age65_3, 0) +
        NVL(age65_4, 0) + NVL(age65_5, 0) + NVL(age65_6, 0) +
        NVL(age65_7, 0) + NVL(age65_8, 0) + NVL(age65_9, 0) +
        NVL(age65_10, 0) + NVL(age65_11, 0) + NVL(age65_12, 0)
    ) / NULLIF(age_total, 0) AS pct_65plus

FROM (
    SELECT
        LPAD(TRIM(state), 2, '0') || LPAD(TRIM(county), 3, '0') AS fips,
        year,

        CASE WHEN B01003_001E >= 0 THEN B01003_001E END AS population,
        CASE WHEN B19013_001E >= 0 THEN B19013_001E END AS median_income,
        CASE WHEN B17001_001E >= 0 THEN B17001_001E END AS pov_total,
        CASE WHEN B17001_002E >= 0 THEN B17001_002E END AS pov_below,
        CASE WHEN B01001_001E >= 0 THEN B01001_001E END AS age_total,
        CASE WHEN B01001_003E >= 0 THEN B01001_003E END AS male_under5,
        CASE WHEN B01001_027E >= 0 THEN B01001_027E END AS female_under5,

        CASE WHEN B01001_020E >= 0 THEN B01001_020E END AS age65_1,
        CASE WHEN B01001_021E >= 0 THEN B01001_021E END AS age65_2,
        CASE WHEN B01001_022E >= 0 THEN B01001_022E END AS age65_3,
        CASE WHEN B01001_023E >= 0 THEN B01001_023E END AS age65_4,
        CASE WHEN B01001_024E >= 0 THEN B01001_024E END AS age65_5,
        CASE WHEN B01001_025E >= 0 THEN B01001_025E END AS age65_6,
        CASE WHEN B01001_044E >= 0 THEN B01001_044E END AS age65_7,
        CASE WHEN B01001_045E >= 0 THEN B01001_045E END AS age65_8,
        CASE WHEN B01001_046E >= 0 THEN B01001_046E END AS age65_9,
        CASE WHEN B01001_047E >= 0 THEN B01001_047E END AS age65_10,
        CASE WHEN B01001_048E >= 0 THEN B01001_048E END AS age65_11,
        CASE WHEN B01001_049E >= 0 THEN B01001_049E END AS age65_12

    FROM RAW_CENSUS_ACS_COUNTY
);



/* ============================================================================
   STEP 4 - CLEAN COUNTY GEOGRAPHY
   ============================================================================ */

CREATE OR REPLACE VIEW STG_COUNTY_GEO AS
SELECT
    LPAD(TRIM(fips), 5, '0') AS fips,
    county_name,
    lat,
    lon
FROM RAW_NY_COUNTY_GEO;


/* ============================================================================
   STEP 5 - AGGREGATE DAILY WEATHER TO COUNTY-QUARTER

   Raw grain: county + day
   New grain: county + quarter

   The original CSV date field was imported to Oracle as WEATHER_DATE.
   ============================================================================ */

CREATE OR REPLACE VIEW INT_WEATHER_COUNTY_QUARTER AS
SELECT
    LPAD(TRIM(fips), 5, '0') AS fips,
    EXTRACT(YEAR FROM weather_date) AS year,
    'Q' || TO_CHAR(weather_date, 'Q') AS quarter,

    COUNT(*) AS days_observed,
    AVG((tmax + tmin) / 2) AS tavg_mean,
    SUM(precip) AS precip_total,
    SUM(snowfall) AS snowfall_total,

    SUM(CASE WHEN tmax >= 90 THEN 1 ELSE 0 END) AS hot_days,
    SUM(CASE WHEN tmin <= 32 THEN 1 ELSE 0 END) AS freeze_days,
    SUM(CASE WHEN precip >= 0.01 THEN 1 ELSE 0 END) AS wet_days

FROM RAW_WEATHER_DAILY_BY_COUNTY
GROUP BY
    LPAD(TRIM(fips), 5, '0'),
    EXTRACT(YEAR FROM weather_date),
    'Q' || TO_CHAR(weather_date, 'Q');


/* ============================================================================
   STEP 6 - COMBINE ED + GEOGRAPHY + ACS + WEATHER

   Time alignment preserved from the original Python project:
   ED target year = Y
   ACS source year = Y - 2
   Weather source = same quarter from Y - 1

   ACS and weather use LEFT JOINs, matching the original Python pipeline.
   Target rows remain in the panel even when an external predictor is missing;
   the modeling notebook later drops incomplete model-ready rows explicitly.

   Example:
   ED 2018 Q1 -> ACS 2016 + Weather 2017 Q1
   ============================================================================ */

CREATE OR REPLACE VIEW INT_COUNTY_QUARTER_BASE AS
SELECT
    e.fips,
    e.year,
    e.quarter,
    e.quarter_num,
    e.period_index,

    e.total_ed_encounters,
    e.treat_and_release_ed,
    e.ed_encounters_requiring,
    e.ed_encounters_admitted_as,
    e.facility_count,
    e.facility_row_count,
    e.facilities_with_ed_data,
    e.facilities_missing_ed_data,
    e.component_gap,

    g.county_name,
    g.lat,
    g.lon,

    a.population AS acs_lag2_population,
    a.median_income AS acs_lag2_median_income,
    a.poverty_rate AS acs_lag2_poverty_rate,
    a.pct_under5 AS acs_lag2_pct_under5,
    a.pct_65plus AS acs_lag2_pct_65plus,
    a.year AS acs_source_year,
    a.population / 100000.0 AS acs_lag2_population_100k,

    w.days_observed AS weather_lag4_days_observed,
    w.tavg_mean AS weather_lag4_tavg_mean,
    w.precip_total AS weather_lag4_precip_total,
    w.snowfall_total AS weather_lag4_snowfall_total,
    w.hot_days AS weather_lag4_hot_days,
    w.freeze_days AS weather_lag4_freeze_days,
    w.wet_days AS weather_lag4_wet_days,
    w.year AS weather_source_year

FROM INT_ED_COUNTY_QUARTER e
JOIN STG_COUNTY_GEO g
    ON e.fips = g.fips
LEFT JOIN STG_CENSUS_ACS a
    ON e.fips = a.fips
   AND a.year = e.year - 2
LEFT JOIN INT_WEATHER_COUNTY_QUARTER w
    ON e.fips = w.fips
   AND w.year = e.year - 1
   AND w.quarter = e.quarter;


/* ============================================================================
   STEP 7 - FINAL ML FEATURE VIEW

   Exact period-index joins to INT_ED_COUNTY_QUARTER are retained instead of
   ordinary LAG() so a missing county-quarter cannot silently turn the previous
   AVAILABLE row into the previous CALENDAR quarter. This also matches the
   original Python order: demand-history features are created before external
   ACS/weather predictors are joined.

   Export this view to CSV for the Python notebook.
   ============================================================================ */

CREATE OR REPLACE VIEW FINAL_COUNTY_QUARTER_ANALYSIS AS
SELECT
    b.fips,
    b.year,
    b.quarter,
    b.quarter_num,
    b.period_index,

    b.total_ed_encounters,
    b.treat_and_release_ed,
    b.ed_encounters_requiring,
    b.ed_encounters_admitted_as,
    b.facility_count,
    b.facility_row_count,
    b.facilities_with_ed_data,
    b.facilities_missing_ed_data,
    b.component_gap,

    b.county_name,
    b.lat,
    b.lon,

    l1.total_ed_encounters AS target_lag1,
    l2.total_ed_encounters AS target_lag2,
    l3.total_ed_encounters AS target_lag3,
    l4.total_ed_encounters AS target_lag4,

    CASE
        WHEN l1.total_ed_encounters IS NOT NULL
         AND l2.total_ed_encounters IS NOT NULL
         AND l3.total_ed_encounters IS NOT NULL
         AND l4.total_ed_encounters IS NOT NULL
        THEN (
            l1.total_ed_encounters +
            l2.total_ed_encounters +
            l3.total_ed_encounters +
            l4.total_ed_encounters
        ) / 4.0
    END AS target_roll4,

    l1.facility_count AS facility_count_lag1,

    b.acs_lag2_population,
    b.acs_lag2_median_income,
    b.acs_lag2_poverty_rate,
    b.acs_lag2_pct_under5,
    b.acs_lag2_pct_65plus,
    b.acs_source_year,
    b.acs_lag2_population_100k,

    b.weather_lag4_days_observed,
    b.weather_lag4_tavg_mean,
    b.weather_lag4_precip_total,
    b.weather_lag4_snowfall_total,
    b.weather_lag4_hot_days,
    b.weather_lag4_freeze_days,
    b.weather_lag4_wet_days,
    b.weather_source_year,

    CASE WHEN b.year IN (2020, 2021) THEN 1 ELSE 0 END AS is_covid,
    CASE WHEN b.year >= 2022 THEN 1 ELSE 0 END AS post_covid,

    CASE WHEN b.quarter = 'Q2' THEN 1 ELSE 0 END AS quarter_Q2,
    CASE WHEN b.quarter = 'Q3' THEN 1 ELSE 0 END AS quarter_Q3,
    CASE WHEN b.quarter = 'Q4' THEN 1 ELSE 0 END AS quarter_Q4,

    CASE
        WHEN b.fips IN ('36005', '36047', '36061', '36081', '36085')
        THEN 1 ELSE 0
    END AS is_nyc,

    CASE
        WHEN b.fips IN ('36059', '36087', '36103', '36119')
        THEN 1 ELSE 0
    END AS is_downstate_non_nyc,

    CASE
        WHEN b.fips IN ('36005', '36047', '36061', '36081', '36085')
        THEN 'nyc'
        WHEN b.fips IN ('36059', '36087', '36103', '36119')
        THEN 'downstate_non_nyc'
        ELSE 'upstate'
    END AS region,

    b.total_ed_encounters
        / NULLIF(b.acs_lag2_population_100k, 0)
        AS facility_ed_encounters_per_100k_proxy

FROM INT_COUNTY_QUARTER_BASE b
LEFT JOIN INT_ED_COUNTY_QUARTER l1
    ON b.fips = l1.fips
   AND l1.period_index = b.period_index - 1
LEFT JOIN INT_ED_COUNTY_QUARTER l2
    ON b.fips = l2.fips
   AND l2.period_index = b.period_index - 2
LEFT JOIN INT_ED_COUNTY_QUARTER l3
    ON b.fips = l3.fips
   AND l3.period_index = b.period_index - 3
LEFT JOIN INT_ED_COUNTY_QUARTER l4
    ON b.fips = l4.fips
   AND l4.period_index = b.period_index - 4;


/* ============================================================================
   STEP 8 - BI-READY METRICS VIEW

   This does NOT replace FINAL_COUNTY_QUARTER_ANALYSIS.
   - FINAL_COUNTY_QUARTER_ANALYSIS -> Python ML / Flask processed panel
   - BI_COUNTY_QUARTER_METRICS     -> Power BI / analyst SQL

   Adds reusable QoQ, YoY, and quarter-level ranking fields.
   ============================================================================ */

CREATE OR REPLACE VIEW BI_COUNTY_QUARTER_METRICS AS
SELECT
    f.*,

    CASE
        WHEN f.target_lag1 IS NOT NULL
        THEN f.total_ed_encounters - f.target_lag1
    END AS qoq_change,

    CASE
        WHEN f.target_lag1 IS NOT NULL
        THEN 100.0 * (f.total_ed_encounters - f.target_lag1)
             / NULLIF(f.target_lag1, 0)
    END AS qoq_pct_change,

    CASE
        WHEN f.target_lag4 IS NOT NULL
        THEN f.total_ed_encounters - f.target_lag4
    END AS yoy_change,

    CASE
        WHEN f.target_lag4 IS NOT NULL
        THEN 100.0 * (f.total_ed_encounters - f.target_lag4)
             / NULLIF(f.target_lag4, 0)
    END AS yoy_pct_change,

    RANK() OVER (
        PARTITION BY f.year, f.quarter
        ORDER BY f.facility_ed_encounters_per_100k_proxy DESC NULLS LAST
    ) AS county_rank_ed_per_100k

FROM FINAL_COUNTY_QUARTER_ANALYSIS f;


/* ============================================================================
   EXPECTED CREATE VIEW MESSAGES
   ============================================================================
   View STG_SPARCS_ED created.
   View INT_ED_COUNTY_QUARTER created.
   View STG_CENSUS_ACS created.
   View STG_COUNTY_GEO created.
   View INT_WEATHER_COUNTY_QUARTER created.
   View INT_COUNTY_QUARTER_BASE created.
   View FINAL_COUNTY_QUARTER_ANALYSIS created.
   View BI_COUNTY_QUARTER_METRICS created.
   ============================================================================ */


/* ============================================================================
   PART B - DATA QUALITY CHECKS
   Run each query below SEPARATELY with Ctrl+Enter / Run Statement.
   ============================================================================ */

/* CHECK 1 - Final row count. Compare with the prior Python output. */
SELECT COUNT(*) AS final_row_count
FROM FINAL_COUNTY_QUARTER_ANALYSIS;

/* CHECK 1B - External predictor joins must not change the ED panel row count. */
SELECT
    (SELECT COUNT(*) FROM INT_ED_COUNTY_QUARTER) AS ed_rows,
    (SELECT COUNT(*) FROM INT_COUNTY_QUARTER_BASE) AS base_rows,
    (SELECT COUNT(*) FROM FINAL_COUNTY_QUARTER_ANALYSIS) AS final_rows
FROM dual;

/* CHECK 2 - Distinct facility counties. Prior project expected 57. */
SELECT COUNT(DISTINCT fips) AS county_count
FROM FINAL_COUNTY_QUARTER_ANALYSIS;

/* CHECK 3 - Target year range. Prior project expected 2018 through 2024. */
SELECT
    MIN(year) AS min_year,
    MAX(year) AS max_year
FROM FINAL_COUNTY_QUARTER_ANALYSIS;

/* CHECK 4 - Duplicate county-quarter keys. Expected: no rows. */
SELECT
    fips,
    year,
    quarter,
    COUNT(*) AS row_count
FROM FINAL_COUNTY_QUARTER_ANALYSIS
GROUP BY fips, year, quarter
HAVING COUNT(*) > 1;

/* CHECK 5 - Missing county geography. Expected: no rows. */
SELECT *
FROM FINAL_COUNTY_QUARTER_ANALYSIS
WHERE county_name IS NULL;

/* CHECK 6 - Negative ED totals. Expected: no rows. */
SELECT *
FROM FINAL_COUNTY_QUARTER_ANALYSIS
WHERE total_ed_encounters < 0;

/* CHECK 7 - County-quarters affected by suppressed/non-numeric facility totals. */
SELECT
    year,
    quarter,
    fips,
    county_name,
    facility_row_count,
    facilities_with_ed_data,
    facilities_missing_ed_data
FROM FINAL_COUNTY_QUARTER_ANALYSIS
WHERE facilities_missing_ed_data > 0
ORDER BY year, quarter_num, fips;

/* CHECK 8 - Yearly summary of missing/suppressed facility totals. */
SELECT
    year,
    SUM(facility_row_count) AS source_rows,
    SUM(facilities_missing_ed_data) AS missing_or_suppressed_ed_rows
FROM FINAL_COUNTY_QUARTER_ANALYSIS
GROUP BY year
ORDER BY year;

/* CHECK 9 - ED FIPS values without geography matches. Expected: no rows. */
SELECT DISTINCT e.fips
FROM INT_ED_COUNTY_QUARTER e
LEFT JOIN STG_COUNTY_GEO g
    ON e.fips = g.fips
WHERE g.fips IS NULL;

/* CHECK 10 - ED county-quarters without required two-year-lag ACS. */
SELECT
    e.fips,
    e.year,
    e.quarter
FROM INT_ED_COUNTY_QUARTER e
LEFT JOIN STG_CENSUS_ACS a
    ON e.fips = a.fips
   AND a.year = e.year - 2
WHERE a.fips IS NULL
ORDER BY e.year, e.quarter_num, e.fips;

/* CHECK 11 - ED county-quarters without required prior-year weather. */
SELECT
    e.fips,
    e.year,
    e.quarter
FROM INT_ED_COUNTY_QUARTER e
LEFT JOIN INT_WEATHER_COUNTY_QUARTER w
    ON e.fips = w.fips
   AND w.year = e.year - 1
   AND w.quarter = e.quarter
WHERE w.fips IS NULL
ORDER BY e.year, e.quarter_num, e.fips;

/* CHECK 12 - Missing important model features. Early lag rows may be expected. */
SELECT
    SUM(CASE WHEN acs_lag2_population IS NULL THEN 1 ELSE 0 END)
        AS missing_population,
    SUM(CASE WHEN acs_lag2_median_income IS NULL THEN 1 ELSE 0 END)
        AS missing_income,
    SUM(CASE WHEN acs_lag2_poverty_rate IS NULL THEN 1 ELSE 0 END)
        AS missing_poverty_rate,
    SUM(CASE WHEN weather_lag4_tavg_mean IS NULL THEN 1 ELSE 0 END)
        AS missing_weather_tavg,
    SUM(CASE WHEN target_lag1 IS NULL THEN 1 ELSE 0 END)
        AS missing_target_lag1,
    SUM(CASE WHEN target_lag4 IS NULL THEN 1 ELSE 0 END)
        AS missing_target_lag4
FROM FINAL_COUNTY_QUARTER_ANALYSIS;

/* CHECK 13 - Component-gap diagnostic; non-zero does not automatically mean error. */
SELECT
    year,
    quarter,
    fips,
    county_name,
    total_ed_encounters,
    component_gap
FROM FINAL_COUNTY_QUARTER_ANALYSIS
WHERE component_gap IS NOT NULL
  AND component_gap <> 0
ORDER BY ABS(component_gap) DESC
FETCH FIRST 50 ROWS ONLY;

/* CHECK 14 - Manual spot check. */
SELECT *
FROM FINAL_COUNTY_QUARTER_ANALYSIS
ORDER BY year, quarter_num, fips
FETCH FIRST 20 ROWS ONLY;

/* CHECK 15 - BI view should preserve the ML view row count. */
SELECT
    (SELECT COUNT(*) FROM FINAL_COUNTY_QUARTER_ANALYSIS) AS ml_view_rows,
    (SELECT COUNT(*) FROM BI_COUNTY_QUARTER_METRICS) AS bi_view_rows
FROM dual;


/* ============================================================================
   PART C - EXPORT FOR PYTHON

   Run ONLY this SELECT with Ctrl+Enter when validation is complete.

   In SQL Developer:
   1. Run the query.
   2. Right-click the result grid -> Export.
   3. Choose CSV.
   4. Save as:
      C:\Users\emily\DATA_SCIENCE\PROJECT\ed_demand_app\data\processed\county_quarter_analysis.csv

   Do not perform the same cleaning/joins again in Python.
   ============================================================================ */

SELECT *
FROM FINAL_COUNTY_QUARTER_ANALYSIS
ORDER BY year, quarter_num, fips;
