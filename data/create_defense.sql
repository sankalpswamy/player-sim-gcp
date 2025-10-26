CREATE OR REPLACE TABLE football.stats_defense (
season INT64,
player_id STRING,
player_name STRING,
team STRING,
position STRING,
games INT64,
tackles_total INT64,
sacks FLOAT64,
interceptions INT64,
forced_fumbles INT64,
fumbles_recovered INT64,
passes_defended INT64,
tfl INT64
);