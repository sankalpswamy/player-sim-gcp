CREATE OR REPLACE VIEW football.features_offense AS
SELECT
player_id, player_name, season, position,
SAFE_DIVIDE(passing_yards, games) AS passing_yards_pg,
SAFE_DIVIDE(passing_tds, games) AS passing_tds_pg,
SAFE_DIVIDE(interceptions, games) AS ints_pg,
SAFE_DIVIDE(rushing_yards, games) AS rushing_yards_pg,
SAFE_DIVIDE(rushing_tds, games) AS rushing_tds_pg,
SAFE_DIVIDE(receiving_yards, games) AS receiving_yards_pg
FROM football.stats_offense
WHERE games > 0;


CREATE OR REPLACE VIEW football.features_defense AS
SELECT
player_id, player_name, season, position,
SAFE_DIVIDE(tackles_total, games) AS tackles_pg,
SAFE_DIVIDE(sacks, games) AS sacks_pg,
SAFE_DIVIDE(interceptions, games) AS ints_pg,
SAFE_DIVIDE(forced_fumbles, games) AS ff_pg,
SAFE_DIVIDE(fumbles_recovered, games) AS fr_pg,
SAFE_DIVIDE(passes_defended, games) AS pd_pg,
SAFE_DIVIDE(tfl, games) AS tfl_pg
FROM football.stats_defense
WHERE games > 0;