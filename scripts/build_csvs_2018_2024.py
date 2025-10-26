#!/usr/bin/env python3
"""
Builds stats_offense.csv and stats_defense.csv for seasons 2018–2024.

- Offense source: nfl_data_py.import_weekly_data (aggregated to season totals)
- Defense source (tiered):
    1) Try nfl_data_py.import_weekly_defense (if available in your version)
    2) Fallback to play-by-play attribution via nfl_data_py.import_pbp_data
       + roster join for names/positions/teams (robust across versions)

Outputs (relative to repo):
  ../statistics/stats_offense.csv
  ../statistics/stats_defense.csv
"""

import os
import pandas as pd
import numpy as np

# ----------------- CONFIG -----------------
YEARS = list(range(2018, 2025))

OFF_POS = {
    "QB","RB","HB","FB","WR","TE",
    "OT","OG","C","LT","RT","G","T","OL",
    "K","P","LS"
}

# Defensive positions we keep in defense.csv
DEF_POS = {
    "DL", "DE", "DT", "NT", "EDGE",
    "LB", "OLB", "ILB", "MLB",
    "DB", "CB", "NB",
    "S", "SS", "FS", "SAF",
    "LCB", "RCB"
}

# Output directory: ../statistics relative to this script file
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "statistics"))
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------- HELPERS -----------------
def get_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] if present; else a zero Series aligned to df.index."""
    if col in df.columns:
        return df[col]
    return pd.Series(0, index=df.index, dtype="float64")

def coalesce_any(df: pd.DataFrame, *cols: str) -> pd.Series:
    """Return the first existing column among cols; else zeros aligned to df.index."""
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series(0, index=df.index, dtype="float64")

def make_game_key(df: pd.DataFrame) -> pd.Series:
    """Create a synthetic per-game key if 'game_id' is unavailable."""
    if "game_id" in df.columns:
        return df["game_id"].astype(str)
    for c in ["season", "week", "recent_team"]:
        if c not in df.columns:
            raise KeyError(f"Expected column '{c}' not found to synthesize game key")
    parts = [
        df["season"].astype(str),
        df["week"].astype(str),
        df["recent_team"].fillna("NA").astype(str),
    ]
    if "opponent" in df.columns:
        parts.append(df["opponent"].fillna("NA").astype(str))
    return pd.Series(["-".join(t) for t in zip(*parts)], index=df.index)

# ---- Roster loader that handles multiple nfl_data_py versions ------------
def try_load_rosters(years) -> pd.DataFrame:
    """
    Return a roster-like DataFrame with columns:
      season, player_id, player_name, position, team

    Tries several function names to cover different nfl_data_py versions and
    maps common ID columns to 'player_id'. If nothing usable is found,
    returns an empty DataFrame with the expected columns.
    """
    # Try several entry points
    candidates = []
    try:
        from nfl_data_py import import_rosters  # common
        candidates.append(("import_rosters", import_rosters))
    except Exception:
        pass
    try:
        from nfl_data_py import import_roster   # some versions
        candidates.append(("import_roster", import_roster))
    except Exception:
        pass
    try:
        from nfl_data_py import import_roster_data  # some versions
        candidates.append(("import_roster_data", import_roster_data))
    except Exception:
        pass
    try:
        from nfl_data_py import import_players  # player master (may lack season/team)
        candidates.append(("import_players", import_players))
    except Exception:
        pass

    df = None
    fn_used = None
    for name, fn in candidates:
        try:
            if name in ("import_rosters", "import_roster", "import_roster_data"):
                df = fn(years)
            else:  # import_players has no year arg
                df = fn()
            fn_used = name
            break
        except Exception:
            continue

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["season","player_id","player_name","position","team"])

    # Identify a usable player-id column, in descending preference
    id_candidates = [
        "player_id",           # gsis_id in many versions
        "gsis_id",
        "gsis_it_id",
        "nfl_player_id",
        "nfl_id",
        "pfr_id",
        "pff_id",
        "espn_id",
        "yahoo_id",
    ]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if id_col is None:
        # No usable ID; return empty to avoid KeyError downstream
        return pd.DataFrame(columns=["season","player_id","player_name","position","team"])

    # Find best-available name/position/team columns
    name_col = next((c for c in ["full_name","player_name","display_name","name"] if c in df.columns), None)
    pos_col  = next((c for c in ["position","pos"] if c in df.columns), None)
    team_col = next((c for c in ["team","recent_team","recent_team_abbr","team_abbr"] if c in df.columns), None)

    # Season column may be missing for player master tables; fill later if absent
    has_season = "season" in df.columns
    if not has_season:
        df = df.copy()
        df["season"] = np.nan

    # Build normalized output
    norm = pd.DataFrame({
        "season": df["season"],
        "player_id": df[id_col].astype(str),
        "player_name": df[name_col] if name_col else "",
        "position": df[pos_col] if pos_col else "",
        "team": df[team_col] if team_col else "",
    })
    # Drop obvious empties
    norm = norm.dropna(subset=["player_id"])
    norm = norm.drop_duplicates(subset=["season","player_id","team","position","player_name"])

    return norm

# ----------------- OFFENSE -----------------
def load_weekly_offense() -> pd.DataFrame:
    from nfl_data_py import import_weekly_data
    df = import_weekly_data(YEARS)

    if "recent_team" not in df.columns and "team" in df.columns:
        df = df.rename(columns={"team": "recent_team"})

    df["_game_key"] = make_game_key(df)
    return df

def make_offense_csv(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.assign(
        _attempts     = coalesce_any(weekly, "attempts", "passing_attempts"),
        _completions  = coalesce_any(weekly, "completions", "passing_completions"),
        _pass_yds     = get_series(weekly, "passing_yards"),
        _pass_tds     = get_series(weekly, "passing_tds"),
        _ints         = coalesce_any(weekly, "interceptions"),
        _rush_yds     = get_series(weekly, "rushing_yards"),
        _rush_tds     = get_series(weekly, "rushing_tds"),
        _rec          = get_series(weekly, "receptions"),
        _rec_yds      = get_series(weekly, "receiving_yards"),
        _rec_tds      = get_series(weekly, "receiving_tds"),
        _fumbles      = coalesce_any(weekly, "fumbles"),
    )

    grp_cols = ["season", "player_id", "player_name", "recent_team", "position"]
    agg = (weekly
           .groupby(grp_cols, dropna=False)
           .agg(
               games          = ("_game_key", pd.Series.nunique),
               attempts       = ("_attempts", "sum"),
               completions    = ("_completions", "sum"),
               passing_yards  = ("_pass_yds", "sum"),
               passing_tds    = ("_pass_tds", "sum"),
               interceptions  = ("_ints", "sum"),
               rushing_yards  = ("_rush_yds", "sum"),
               rushing_tds    = ("_rush_tds", "sum"),
               receptions     = ("_rec", "sum"),
               receiving_yards= ("_rec_yds", "sum"),
               receiving_tds  = ("_rec_tds", "sum"),
               fumbles        = ("_fumbles", "sum"),
           )
           .reset_index())

    out = agg.rename(columns={"recent_team": "team"})[
        ["season","player_id","player_name","team","position","games",
         "attempts","completions","passing_yards","passing_tds","interceptions",
         "rushing_yards","rushing_tds","receptions","receiving_yards","receiving_tds","fumbles"]
    ].fillna(0)

    int_cols = ["season","games","attempts","completions","passing_yards","passing_tds",
                "interceptions","rushing_yards","rushing_tds","receptions",
                "receiving_yards","receiving_tds","fumbles"]
    for c in int_cols:
        out[c] = out[c].astype(int)

    out.sort_values(["season","player_id","team"], inplace=True)
    out = (out
           .groupby(["season","player_id"], as_index=False, group_keys=False)
           .apply(lambda g: g.tail(1))
           .reset_index(drop=True))
    return out

# ----------------- DEFENSE -----------------
def try_import_weekly_defense():
    """Try to import a defense-specific weekly dataset; return None if unsupported."""
    try:
        from nfl_data_py import import_weekly_defense  # may not exist in all versions
        df = import_weekly_defense(YEARS)
        df["source"] = "weekly_defense"
        return df
    except Exception:
        return None

def load_defense_from_pbp() -> pd.DataFrame:
    """
    Fallback: Build weekly defensive events from play-by-play,
    derive names/teams from PBP (robust), and then best-effort fill from rosters.
    """
    from nfl_data_py import import_pbp_data
    pbp = import_pbp_data(YEARS)

    # Helper: first present col in list
    def first_present(df, cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    # IDs for each defensive stat
    int_pid = first_present(pbp, ["interception_player_id", "interceptor_player_id", "interception_player_id_1"])
    int_pnm = first_present(pbp, ["interception_player_name", "interceptor_player_name", "interception_player_name_1"])
    sack_pid = first_present(pbp, ["sack_player_id", "sack_player_id_1", "qb_hit_1_player_id"])
    sack_pnm = first_present(pbp, ["sack_player_name", "sack_player_name_1", "qb_hit_1_player_name"])
    ff_pid   = first_present(pbp, ["forced_fumble_player_1_player_id", "forced_fumble_player_id", "forced_fumble_player_id_1"])
    ff_pnm   = first_present(pbp, ["forced_fumble_player_1_player_name", "forced_fumble_player_name", "forced_fumble_player_name_1"])
    fr_pid   = first_present(pbp, ["fumble_recovery_1_player_id", "fumble_recovery_player_id", "fumble_recovery_player_id_1"])
    fr_pnm   = first_present(pbp, ["fumble_recovery_1_player_name", "fumble_recovery_player_name", "fumble_recovery_player_name_1"])
    pd_pid   = first_present(pbp, ["pass_defensed_1_player_id", "pass_defended_player_id", "pass_defended_player_id_1"])
    pd_pnm   = first_present(pbp, ["pass_defensed_1_player_name", "pass_defended_player_name", "pass_defended_player_name_1"])
    tfl_pid  = first_present(pbp, ["tackle_for_loss_1_player_id", "tfl_player_id", "tackled_for_loss_player_id"])
    tfl_pnm  = first_present(pbp, ["tackle_for_loss_1_player_name", "tfl_player_name", "tackled_for_loss_player_name"])

    # Defensive team column on each play
    defteam_col = "defteam" if "defteam" in pbp.columns else None

    frames, label_frames = [], []

    def collect(stat_flag, pid_col, pnm_col, stat_name):
        if pid_col is None or stat_flag not in pbp.columns:
            return
        sel_cols = ["season", "week", pid_col]
        if pnm_col: sel_cols.append(pnm_col)
        if defteam_col: sel_cols.append(defteam_col)

        g = pbp.loc[pbp[stat_flag] == 1, sel_cols].dropna(subset=[pid_col]).copy()
        g = g.rename(columns={pid_col: "player_id"})
        g["stat"] = stat_name
        g["val"] = 1.0

        frames.append(g[["season","week","player_id","stat","val"]])

        # labels: take whatever name/team is present for later mode()
        lab = g[["season","player_id"]].copy()
        lab["player_name"] = g[pnm_col] if pnm_col in g.columns else ""
        lab["team"] = g[defteam_col] if defteam_col in g.columns else ""
        label_frames.append(lab)

    collect("interception", int_pid, int_pnm, "interceptions")
    collect("sack",         sack_pid, sack_pnm, "sacks")
    collect("forced_fumble", ff_pid,  ff_pnm,   "forced_fumbles")
    if fr_pid:
        # fumble recoveries recorded when an id is present; not always a flag
        cols = ["season","week",fr_pid]
        if fr_pnm: cols.append(fr_pnm)
        if defteam_col: cols.append(defteam_col)
        g = pbp.loc[pbp[fr_pid].notna(), cols].copy()
        g = g.rename(columns={fr_pid: "player_id"})
        g["stat"] = "fumbles_recovered"; g["val"] = 1.0
        frames.append(g[["season","week","player_id","stat","val"]])

        lab = g[["season","player_id"]].copy()
        lab["player_name"] = g[fr_pnm] if fr_pnm in g.columns else ""
        lab["team"] = g[defteam_col] if defteam_col in g.columns else ""
        label_frames.append(lab)

    collect("pass_defended", pd_pid,  pd_pnm,  "passes_defended")
    collect("tackle_for_loss", tfl_pid, tfl_pnm, "tfl")

    if not frames:
        # Nothing attributed — return empty frame with headers
        return pd.DataFrame(columns=["season","week","player_id","interceptions","sacks",
                                     "forced_fumbles","fumbles_recovered","passes_defended","tfl",
                                     "player_name","team","position"])

    events = pd.concat(frames, ignore_index=True)

    # Pivot per defender-week to counts
    weekly_def = (events
        .groupby(["season","week","player_id","stat"], as_index=False)["val"].sum()
        .pivot_table(index=["season","week","player_id"], columns="stat", values="val", fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1))

    # ---- Derive labels (name/team) from PBP by mode per season+player_id ----
    labels = pd.concat(label_frames, ignore_index=True)
    # Clean strings
    for c in ["player_name","team"]:
        if c in labels.columns:
            labels[c] = labels[c].fillna("").astype(str).str.strip()
        else:
            labels[c] = ""
    labels["player_id"] = labels["player_id"].astype(str)

    def most_frequent(s: pd.Series) -> str:
        s = s[s.astype(str).str.len() > 0]
        if s.empty: return ""
        return s.value_counts().idxmax()

    label_mode = (labels
        .groupby(["season","player_id"], as_index=False)
        .agg(player_name=("player_name", most_frequent),
             team=("team", most_frequent)))

    weekly_def["player_id"] = weekly_def["player_id"].astype(str)
    weekly_def = weekly_def.merge(label_mode, on=["season","player_id"], how="left")

    # ---- Roster (secondary fill) ----
    roster_norm = try_load_rosters(YEARS)  # season, player_id, player_name, position, team
    if roster_norm is not None and len(roster_norm):
        roster_norm["player_id"] = roster_norm["player_id"].astype(str)
        if "season" in roster_norm.columns:
            weekly_def = weekly_def.merge(
                roster_norm[["season","player_id","player_name","position","team"]],
                on=["season","player_id"], how="left", suffixes=("", "_roster")
            )
        else:
            weekly_def = weekly_def.merge(
                roster_norm[["player_id","player_name","position","team"]],
                on=["player_id"], how="left", suffixes=("", "_roster")
            )

        # Fill blanks with roster values but prefer PBP-derived if present
        for col in ["player_name","team"]:
            weekly_def[col] = weekly_def[col].where(weekly_def[col].astype(str).str.len() > 0,
                                                    weekly_def[f"{col}_roster"])
            if f"{col}_roster" in weekly_def.columns:
                weekly_def.drop(columns=[f"{col}_roster"], inplace=True, errors="ignore")
        # position only comes from roster here
        if "position_roster" in weekly_def.columns and "position" in weekly_def.columns:
            weekly_def["position"] = weekly_def["position"].where(
                weekly_def["position"].astype(str).str.len() > 0,
                weekly_def["position_roster"]
            )
            weekly_def.drop(columns=["position_roster"], inplace=True, errors="ignore")

    # Final string cleanup
    for c in ["player_name","team","position"]:
        if c not in weekly_def.columns:
            weekly_def[c] = ""
        weekly_def[c] = weekly_def[c].fillna("").astype(str).str.strip()

    weekly_def["source"] = "pbp"
    return weekly_def

def load_weekly_defense_or_pbp() -> pd.DataFrame:
    # Try weekly defense first (if your nfl_data_py exposes it)
    df = try_import_weekly_defense()
    if df is not None and len(df):
        return df
    # Fallback to PBP
    return load_defense_from_pbp()

def make_defense_csv(def_like: pd.DataFrame) -> pd.DataFrame:
    df = def_like.copy()

    # Required columns
    for need in ["season","week","player_id"]:
        if need not in df.columns:
            raise KeyError(f"Defense frame missing required column '{need}'")

    # Ensure stat columns exist
    for stat in ["interceptions","sacks","forced_fumbles","fumbles_recovered","passes_defended","tfl","tackles_total"]:
        if stat not in df.columns:
            df[stat] = 0

    # Prefer defensive INTs label if present
    if "interceptions_def" in df.columns:
        df["interceptions"] = df["interceptions_def"]

    # Game key for counting games
    df["_game_key"] = df["season"].astype(str) + "-" + df["week"].astype(str)

    # Normalize identity columns to strings BEFORE filtering/grouping
    if "team" not in df.columns and "recent_team" in df.columns:
        df = df.rename(columns={"recent_team": "team"})
    for col in ["player_id","player_name","team","position"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # --------- FILTER: defenders only (and block offensive leakage) ----------
    # Treat these columns as defensive-only counts (from weekly_defense or PBP):
    statmask = (
        df["interceptions"].fillna(0).astype(float) > 0
        ) | (df["sacks"].fillna(0).astype(float) > 0
        ) | (df["forced_fumbles"].fillna(0).astype(float) > 0
        ) | (df["fumbles_recovered"].fillna(0).astype(float) > 0
        ) | (df["passes_defended"].fillna(0).astype(float) > 0
        ) | (df["tfl"].fillna(0).astype(float) > 0
        ) | (df["tackles_total"].fillna(0).astype(float) > 0
    )

    pos_clean = df["position"].str.upper().str.strip()
    # Keep known defensive positions OR rows with defensive stats but NOT known offense positions
    keepmask = pos_clean.isin(DEF_POS) | (statmask & ~pos_clean.isin(OFF_POS))
    df = df[keepmask].copy()

    # For rows with defensive stats but missing/odd position, tag as DEF
    import numpy as _np
    pos_clean = df["position"].str.upper().str.strip()
    df["position"] = _np.where(
        pos_clean.isin(DEF_POS), pos_clean,
        _np.where(statmask & ~pos_clean.isin(OFF_POS), "DEF", pos_clean)
    )

    # Aggregate by player-season-team
    grp_cols = ["season","player_id","player_name","team","position"]
    agg = (df
           .groupby(grp_cols, dropna=False)
           .agg(
               games            = ("_game_key", pd.Series.nunique),
               tackles_total    = ("tackles_total", "sum"),
               sacks            = ("sacks", "sum"),
               interceptions    = ("interceptions", "sum"),
               forced_fumbles   = ("forced_fumbles", "sum"),
               fumbles_recovered= ("fumbles_recovered", "sum"),
               passes_defended  = ("passes_defended", "sum"),
               tfl              = ("tfl", "sum"),
           )
           .reset_index())

    out = agg.fillna(0)

    # Types
    out["games"] = out["games"].astype(int)
    for c in ["tackles_total","interceptions","forced_fumbles","fumbles_recovered","passes_defended","tfl"]:
        out[c] = out[c].astype(int)
    out["sacks"] = out["sacks"].astype(float)

    # Last-team of season
    out.sort_values(["season","player_id","team"], inplace=True)
    out = (out
           .groupby(["season","player_id"], as_index=False, group_keys=False)
           .apply(lambda g: g.tail(1))
           .reset_index(drop=True))

    cols = ["season","player_id","player_name","team","position","games",
            "tackles_total","sacks","interceptions","forced_fumbles","fumbles_recovered","passes_defended","tfl"]
    for c in cols:
        if c not in out.columns:
            out[c] = "" if c in {"player_id","player_name","team","position"} else 0

    # Final string assurance
    for c in ["player_id","player_name","team","position"]:
        out[c] = out[c].astype(str)

    return out[cols]

def make_defense_csv_from_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    if "_game_key" not in weekly.columns:
        weekly["_game_key"] = make_game_key(weekly)

    # Base defensive-friendly columns
    tackles_total     = get_series(weekly, "tackles_solo") + get_series(weekly, "tackles_assists")
    sacks_raw         = coalesce_any(weekly, "sacks", "sacks_def", "def_sacks")
    ints_def_raw      = get_series(weekly, "interceptions_def")  # do NOT fall back yet
    forced_fumbles    = coalesce_any(weekly, "forced_fumbles", "fumbles_forced")
    fumbles_recovered = coalesce_any(weekly, "fumble_recoveries", "fumbles_rec", "fumbles_recovered")
    passes_defended   = coalesce_any(weekly, "passes_defended", "pass_defended")
    tfl               = coalesce_any(weekly, "tackles_for_loss", "tfl")

    df = weekly.copy()

    # Normalize identity strings
    if "recent_team" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"recent_team": "team"})
    for col in ["player_id","player_name","team","position"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # Position buckets (before any filtering)
    pos_clean_pre = df["position"].str.upper().str.strip()

    # INTs: prefer defense-specific; only fall back to generic for clear defenders
    generic_ints = get_series(df, "interceptions")
    ints_safe = ints_def_raw
    if ("interceptions_def" not in df.columns) or (ints_def_raw.sum(skipna=True) == 0):
        ints_safe = np.where(pos_clean_pre.isin(DEF_POS), generic_ints, 0)

    # Sacks: only for defenders (prevents QB "times sacked" leakage)
    sacks_safe = np.where(pos_clean_pre.isin(DEF_POS), sacks_raw, 0)

    df = df.assign(
        _tackles_total = tackles_total.fillna(0),
        _sacks         = pd.Series(sacks_safe, index=df.index).fillna(0),
        _ints          = pd.Series(ints_safe, index=df.index).fillna(0),
        _ff            = forced_fumbles.fillna(0),
        _fr            = fumbles_recovered.fillna(0),
        _pd            = passes_defended.fillna(0),
        _tfl           = tfl.fillna(0),
    )

    # Keep defenders (by position) OR rows with defensive-looking stats AND not an offensive position
    statmask_pre = (
        (df["_tackles_total"] > 0) | (df["_sacks"] > 0) | (df["_ints"] > 0) |
        (df["_ff"] > 0) | (df["_fr"] > 0) | (df["_pd"] > 0) | (df["_tfl"] > 0)
    )
    keepmask = pos_clean_pre.isin(DEF_POS) | (statmask_pre & ~pos_clean_pre.isin(OFF_POS))
    df = df[keepmask].copy()

    # Recompute masks on filtered frame; set DEF for unknown-but-defensive rows
    pos_clean = df["position"].str.upper().str.strip()
    statmask  = (
        (df["_tackles_total"] > 0) | (df["_sacks"] > 0) | (df["_ints"] > 0) |
        (df["_ff"] > 0) | (df["_fr"] > 0) | (df["_pd"] > 0) | (df["_tfl"] > 0)
    )
    df["position"] = np.where(
        pos_clean.isin(DEF_POS), pos_clean,
        np.where(statmask & ~pos_clean.isin(OFF_POS), "DEF", pos_clean)
    )

    # -------- Aggregate to season totals (count games via unique _game_key) --------
    grp_cols = ["season","player_id","player_name","team","position"]
    agg = (df.groupby(grp_cols, dropna=False)
             .agg(
                 games            = ("_game_key", pd.Series.nunique),
                 tackles_total    = ("_tackles_total", "sum"),
                 sacks            = ("_sacks", "sum"),
                 interceptions    = ("_ints", "sum"),
                 forced_fumbles   = ("_ff", "sum"),
                 fumbles_recovered= ("_fr", "sum"),
                 passes_defended  = ("_pd", "sum"),
                 tfl              = ("_tfl", "sum"),
             ).reset_index())

    out = agg.fillna(0)

    # Casts
    out["games"] = out["games"].astype(int)
    for c in ["tackles_total","interceptions","forced_fumbles","fumbles_recovered","passes_defended","tfl"]:
        out[c] = out[c].astype(int)
    out["sacks"] = out["sacks"].astype(float)

    # Mid-season team changes → keep last row per player-season
    out.sort_values(["season","player_id","team"], inplace=True)
    out = (out.groupby(["season","player_id"], as_index=False, group_keys=False)
              .apply(lambda g: g.tail(1))
              .reset_index(drop=True))

    # Final columns & string safety
    cols = ["season","player_id","player_name","team","position","games",
            "tackles_total","sacks","interceptions","forced_fumbles","fumbles_recovered","passes_defended","tfl"]
    for c in cols:
        if c not in out.columns:
            out[c] = "" if c in {"player_id","player_name","team","position"} else 0
    for c in ["player_id","player_name","team","position"]:
        out[c] = out[c].astype(str)

    return out[cols]

# ----------------- MAIN -----------------
def main():
    print("Loading weekly offense data… (2018–2024)")
    weekly_off = load_weekly_offense()
    weekly_off = weekly_off[weekly_off["season"].between(2018, 2024, inclusive="both")]

    print("Building offense CSV…")
    offense = make_offense_csv(weekly_off)
    offense_path = os.path.join(OUT_DIR, "stats_offense.csv")
    offense.to_csv(offense_path, index=False)
    print(f"✓ Wrote {len(offense):,} rows to {offense_path}")

    print("Loading defense data (weekly defense or PBP fallback)…")
    weekly_def = load_weekly_defense_or_pbp()
    # keep years 2018–2024 just in case
    if "season" in weekly_def.columns:
        weekly_def = weekly_def[weekly_def["season"].between(2018, 2024, inclusive="both")]

    print("Building defense CSV…")
    defense = make_defense_csv(weekly_def)
    defense_path = os.path.join(OUT_DIR, "stats_defense.csv")
    defense.to_csv(defense_path, index=False)
    print(f"✓ Wrote {len(defense):,} rows to {defense_path}")

if __name__ == "__main__":
    main()
