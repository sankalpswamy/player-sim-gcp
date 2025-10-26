CREATE SCHEMA IF NOT EXISTS football;


CREATE OR REPLACE TABLE football.stats_offense (
season INT64,
player_id STRING,
player_name STRING,
team STRING,
position STRING,
games INT64,
attempts INT64,
completions INT64,
passing_yards INT64,
passing_tds INT64,
interceptions INT64,
rushing_yards INT64,
rushing_tds INT64,
receptions INT64,
receiving_yards INT64,
receiving_tds INT64,
fumbles INT64
);