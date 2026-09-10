-- ============================================================
-- analysis_queries.sql
-- Core analytical queries against sql/ea_analytics.db (table: reviews)
-- Demonstrates: aggregations, CTEs, window functions, self-joins.
-- Run with: sqlite3 sql/ea_analytics.db < sql/analysis_queries.sql
-- ============================================================

-- 1. Average rating & review volume per game (basic aggregation)
SELECT
    game,
    COUNT(*)                AS total_reviews,
    ROUND(AVG(rating), 2)   AS avg_rating,
    ROUND(AVG(playtime_hours), 1) AS avg_playtime_hours
FROM reviews
GROUP BY game
ORDER BY avg_rating DESC;

-- 2. Daily rating trend with a 7-day rolling average (window function)
-- Useful for spotting slow drift vs sudden shocks in player sentiment.
WITH daily AS (
    SELECT
        game,
        review_date,
        COUNT(*)              AS n_reviews,
        AVG(rating)           AS avg_rating
    FROM reviews
    GROUP BY game, review_date
)
SELECT
    game,
    review_date,
    n_reviews,
    ROUND(avg_rating, 2) AS avg_rating,
    ROUND(
        AVG(avg_rating) OVER (
            PARTITION BY game
            ORDER BY review_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 2
    ) AS rolling_7d_avg_rating
FROM daily
ORDER BY game, review_date;

-- 3. Platform-level engagement ranking within each game (RANK window function)
WITH platform_stats AS (
    SELECT
        game,
        platform,
        COUNT(*)                       AS n_reviews,
        ROUND(AVG(rating), 2)          AS avg_rating,
        ROUND(AVG(playtime_hours), 1)  AS avg_playtime
    FROM reviews
    GROUP BY game, platform
)
SELECT
    game,
    platform,
    n_reviews,
    avg_rating,
    avg_playtime,
    RANK() OVER (PARTITION BY game ORDER BY avg_rating DESC) AS rating_rank_in_game
FROM platform_stats
ORDER BY game, rating_rank_in_game;

-- 4. Regional breakdown with share-of-total (window function, no self-join needed)
SELECT
    region,
    game,
    COUNT(*) AS n_reviews,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY game), 1) AS pct_of_game_reviews
FROM reviews
GROUP BY region, game
ORDER BY game, pct_of_game_reviews DESC;

-- 5. Before/after comparison around the Battlefield 2042 patch (CTE + CASE)
-- This is the SQL-side check that backs the anomaly-detection notebook section.
WITH labeled AS (
    SELECT
        rating,
        review_date,
        CASE
            WHEN is_post_patch_window = 1 THEN 'post_patch_window'
            ELSE 'baseline'
        END AS period
    FROM reviews
    WHERE game = 'Battlefield 2042'
)
SELECT
    period,
    COUNT(*)               AS n_reviews,
    ROUND(AVG(rating), 2)  AS avg_rating
FROM labeled
GROUP BY period;

-- 6. Power users vs casual players (self-referencing CTE + window function)
-- Buckets users by playtime percentile, then checks whether power users
-- rate the game differently -- a segmentation-style question a PA would ask.
WITH ranked AS (
    SELECT
        review_id,
        game,
        rating,
        playtime_hours,
        PERCENT_RANK() OVER (PARTITION BY game ORDER BY playtime_hours) AS pctile
    FROM reviews
),
bucketed AS (
    SELECT
        game,
        rating,
        CASE
            WHEN pctile >= 0.8 THEN 'power_user'
            WHEN pctile <= 0.2 THEN 'casual_user'
            ELSE 'regular_user'
        END AS user_segment
    FROM ranked
)
SELECT
    game,
    user_segment,
    COUNT(*)              AS n_reviews,
    ROUND(AVG(rating), 2) AS avg_rating
FROM bucketed
GROUP BY game, user_segment
ORDER BY game, user_segment;
