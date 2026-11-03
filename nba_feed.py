"""
nba_feed.py - turns free nba_api game logs into MEASURED simulation parameters
==============================================================================

What nba_api gives you:   historical player game logs (minutes, box score)
What it does NOT give:    betting lines, player props, or an injury report

So this module replaces the hand-set constants in the simulator with numbers
measured from real games, and keeps YOUR projections as the mean of every
player. Specifically it measures:

  * league-wide: blowout minute-loss for starters / bench, margin spread,
    game-level and team-level shooting swings, and real teammate / opponent
    correlation (so the sim can be checked against reality)
  * per player: recent minutes (exponentially weighted), minutes volatility
    and per-minute scoring volatility (shrunk toward role averages so small
    samples don't produce silly numbers)
  * a sanity check flagging projections far from a player's recent output

Install:  pip install nba_api pandas numpy
"""

import datetime as dt
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


# ==================================================
# HELPERS
# ==================================================

def match_key(name):
    """'Nikola Jokić Jr.' -> 'nikola jokic' (for joining slate to NBA logs)."""

    s = unicodedata.normalize("NFKD", str(name))

    s = "".join(c for c in s if not unicodedata.combining(c)).lower()

    s = re.sub(r"[.'’`]", "", s)

    s = re.sub(r"[^a-z0-9 ]", " ", s)

    return " ".join(t for t in s.split() if t not in SUFFIXES)


def current_season(today=None):

    today = today or dt.date.today()

    y = today.year if today.month >= 10 else today.year - 1

    return f"{y}-{(y + 1) % 100:02d}"


def previous_season(season):

    y = int(season[:4]) - 1

    return f"{y}-{(y + 1) % 100:02d}"


def fd_points(df):

    return (
        df["pts"] + 1.2 * df["reb"] + 1.5 * df["ast"]
        + 3 * df["stl"] + 3 * df["blk"] - df["tov"]
    )


def _minutes(x):
    """nba_api sometimes returns minutes as 'mm:ss' strings."""

    if isinstance(x, str) and ":" in x:

        m, s = x.split(":")

        return float(m) + float(s) / 60

    try:
        return float(x)

    except (TypeError, ValueError):
        return np.nan


# ==================================================
# DATA PULL (cached - be kind to stats.nba.com)
# ==================================================

class NBAFeed:

    def __init__(self, cache_dir="nba_cache", max_age_hours=6, seasons=2, log=print):

        self.cache_dir = cache_dir

        self.max_age = max_age_hours * 3600

        self.seasons = seasons

        self.log = log

        os.makedirs(cache_dir, exist_ok=True)

    def get_logs(self, force=False):
        """All player game logs for this season (+ last season early on)."""

        season = current_season()

        names = [season]

        for _ in range(self.seasons - 1):
            names.append(previous_season(names[-1]))

        frames = []

        for k, name in enumerate(names):

            path = os.path.join(self.cache_dir, f"player_logs_{name}.csv")

            fresh = (
                os.path.exists(path)
                and (k > 0 or time.time() - os.path.getmtime(path) < self.max_age)
                and not force
            )

            if fresh:

                df = pd.read_csv(path)

                self.log(f"NBA feed: {name} loaded from cache ({len(df):,} rows)")

            else:

                try:

                    df = self._fetch(name)

                    df.to_csv(path, index=False)

                    self.log(f"NBA feed: {name} downloaded ({len(df):,} rows)")

                except Exception as e:

                    if os.path.exists(path):

                        df = pd.read_csv(path)

                        self.log(f"NBA feed: {name} download failed ({e}); using stale cache")

                    elif k == 0 and len(names) > 1:

                        self.log(f"NBA feed: {name} unavailable ({e})")

                        continue

                    else:
                        raise

            frames.append(df)

        if not frames:
            raise RuntimeError("No NBA game logs available")

        logs = prepare_logs(pd.concat(frames, ignore_index=True))

        try:

            today = dt.date.today()

            age = (today - logs["game_date"].max().date()).days

            if age >= 3 and (today.month >= 11 or today.month <= 4):
                self.log(
                    f"NBA feed WARNING: newest game in the data is {age} days old. "
                    "Rest / back-to-back / recent-form features may be wrong; "
                    "click 'Refresh NBA data' (or check your internet connection)."
                )

        except Exception:
            pass

        return logs

    def _fetch(self, season):

        try:
            from nba_api.stats.endpoints import leaguegamelog

        except ImportError:
            raise RuntimeError("nba_api is not installed: pip install nba_api")

        last = None

        for attempt in range(3):

            try:

                return leaguegamelog.LeagueGameLog(
                    season=season,
                    player_or_team_abbreviation="P",
                    season_type_all_star="Regular Season",
                    timeout=60
                ).get_data_frames()[0]

            except Exception as e:

                last = e

                time.sleep(2 * (attempt + 1))

        raise RuntimeError(f"stats.nba.com request failed: {last}")


def prepare_logs(raw):
    """Normalize columns; add fantasy points and each team's final margin."""

    df = raw.copy()

    df.columns = df.columns.str.lower()

    df["min"] = df["min"].map(_minutes)

    df["game_date"] = pd.to_datetime(df["game_date"])

    for c in ("pts", "reb", "ast", "stl", "blk", "tov"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df = df[df["min"] > 0].copy()

    df["fp"] = fd_points(df)

    df["key"] = df["player_name"].map(match_key)

    tp = (
        df.groupby(["game_id", "team_abbreviation"], as_index=False)["pts"]
        .sum().rename(columns={"pts": "team_pts"})
    )

    tp["n_teams"] = tp.groupby("game_id")["team_pts"].transform("count")

    tp["game_pts"] = tp.groupby("game_id")["team_pts"].transform("sum")

    tp = tp[tp["n_teams"] == 2]

    tp["margin"] = 2 * tp["team_pts"] - tp["game_pts"]

    df = df.merge(
        tp[["game_id", "team_abbreviation", "margin"]],
        on=["game_id", "team_abbreviation"], how="inner"
    )

    return df.sort_values("game_date").reset_index(drop=True)


# ==================================================
# LEAGUE-WIDE PARAMETERS
# ==================================================

class LeagueParams:

    def __init__(self, overrides, extra):

        self.sim_overrides = overrides

        self.extra = extra

    def summary(self):

        o = self.sim_overrides

        e = self.extra

        return "\n".join([
            "NBA FEED - MEASURED LEAGUE PARAMETERS",
            f"  Games used: {e['games']:,}   player-games: {e['rows']:,}",
            f"  Margin sd (given spread, est.): {o['margin_sd']:.1f}",
            f"  Starter minute loss in blowout: winners {o['starter_loss_win']:.2f}"
            f" / losers {o['starter_loss_lose']:.2f}",
            f"  Bench minute gain in blowout:   rotation {o['bench_gain_rot']:.2f}"
            f" / deep {o['bench_gain_deep']:.2f}",
            f"  Game-level scoring swing sd: {e['game_sd']:.3f}   team-level: {o['team_sd']:.3f}",
            f"  Empirical corr, top-2 teammates: {e['corr_team']:+.3f}   "
            f"top-1 opponents: {e['corr_opp']:+.3f}",
        ])


def _role(median_min):

    return np.where(median_min >= 26, "starter",
                    np.where(median_min >= 12, "rotation", "deep"))


def _fit_ratio(b, ratio):
    """ratio ~ a + c*b  ->  relative change per unit blowout = c / a."""

    if len(b) < 50 or np.ptp(b) == 0:
        return None

    c, a = np.polyfit(b, ratio, 1)

    return c / a if a > 0 else None


def compute_league_params(logs, defaults):
    """
    defaults: dict with the sim's current constants (used as fallbacks and for
    the blowout curve shape: blow_start, blow_span).
    """

    df = logs.copy()

    out = dict(defaults)

    # ---- margin spread
    uncond = float(df.drop_duplicates(["game_id", "team_abbreviation"])["margin"].std())

    out["margin_sd"] = float(np.sqrt(max(uncond ** 2 - 6.0 ** 2, 100.0)))

    # ---- blowout minutes
    med = df.groupby("key")["min"].transform("median")

    n_games = df.groupby("key")["min"].transform("count")

    df = df[n_games >= 10].copy()

    df["med"] = med[df.index]

    df["role"] = _role(df["med"])

    df["ratio"] = df["min"] / df["med"]

    df["b"] = np.clip(
        (df["margin"].abs() - out["blow_start"]) / out["blow_span"], 0, 1
    )

    def fit(mask, sign):

        sub = df[mask]

        v = _fit_ratio(sub["b"].values, sub["ratio"].values)

        return None if v is None else float(np.clip(sign * v, 0.0, 0.9))

    for key, mask, sign in [
        ("starter_loss_win", (df.role == "starter") & (df.margin > 0), -1),
        ("starter_loss_lose", (df.role == "starter") & (df.margin <= 0), -1),
        ("bench_gain_rot", df.role == "rotation", 1),
        ("bench_gain_deep", df.role == "deep", 1),
    ]:

        v = fit(mask, sign)

        if v is not None:
            out[key] = min(v, 0.6) if "starter" in key else v

    # ---- game / team scoring swings (log fantasy points per minute)
    df = df[df["min"] >= 10].copy()

    df["lfpm"] = np.log(np.maximum(df["fp"], 1.0) / df["min"])

    df["resid"] = df["lfpm"] - df.groupby("key")["lfpm"].transform("mean")

    def agg(g):

        w = g["min"].values

        r = g["resid"].values

        avg = np.average(r, weights=w)

        var = np.average((r - avg) ** 2, weights=w)

        return pd.Series({"avg": avg, "wvar": var, "n": len(g)})

    tg = df.groupby(["game_id", "team_abbreviation"]).apply(agg).reset_index()

    tg = tg[tg["n"] >= 5]

    tg["rank"] = tg.groupby("game_id").cumcount()

    pair = tg.pivot(index="game_id", columns="rank", values="avg").dropna()

    var_game = max(float(np.mean(pair[0] * pair[1])), 1e-4) if len(pair) > 20 else 4e-4

    noise = float(np.mean(tg["wvar"] / tg["n"]))

    var_team = max(float(tg["avg"].var()) - var_game - noise, 1e-4)

    out["pace_sd"] = float(0.75 * np.sqrt(var_game))

    out["shoot_sd"] = float(0.66 * np.sqrt(var_game))

    out["team_sd"] = float(np.sqrt(var_team))

    # ---- empirical correlation of fantasy points (top-2 teammates, top-1 opponents)
    d = df.copy()

    d["z"] = (d["fp"] - d.groupby("key")["fp"].transform("mean")) / d.groupby("key")["fp"].transform("std")

    d = d.dropna(subset=["z"])

    # rank by the player's TYPICAL minutes, not minutes played that night
    # (ranking by the night's minutes would filter out the blowout effect)
    d["typ"] = d.groupby("key")["min"].transform("median")

    d["mrank"] = d.groupby(["game_id", "team_abbreviation"])["typ"].rank(ascending=False, method="first")

    top = d[d["mrank"] <= 2]

    wide = top.pivot_table(index=["game_id", "team_abbreviation"], columns="mrank", values="z").dropna()

    corr_team = float(wide[1.0].corr(wide[2.0])) if len(wide) > 30 else float("nan")

    one = top[top["mrank"] == 1].copy()

    one["rank"] = one.groupby("game_id").cumcount()

    opp = one.pivot(index="game_id", columns="rank", values="z").dropna()

    corr_opp = float(opp[0].corr(opp[1])) if len(opp) > 30 else float("nan")

    extra = {
        "games": int(logs["game_id"].nunique()),
        "rows": int(len(logs)),
        "game_sd": float(np.sqrt(var_game)),
        "corr_team": corr_team,
        "corr_opp": corr_opp,
        "var_game": var_game,
        "var_team": var_team,
    }

    keep = {k: out[k] for k in defaults}

    return LeagueParams(keep, extra)


# ==================================================
# PER-PLAYER PROFILES
# ==================================================

def _wstd(x, w):

    m = np.average(x, weights=w)

    return float(np.sqrt(np.average((x - m) ** 2, weights=w)))


def compute_profiles(logs, league, halflife=12, window=25, min_games=4):
    """
    One row per player: recent minutes, plus minutes / efficiency volatility
    already converted into the sim's terms (blowout, game and team effects,
    usage noise and hot-game mixture removed so nothing is double counted).
    """

    o = league.sim_overrides

    e = league.extra

    rows = []

    hot_var = o["hot_rate"] * 0.0558 - (o["hot_rate"] * 0.225) ** 2

    for key, g in logs.groupby("key"):

        g = g.sort_values("game_date", ascending=False).head(window)

        n = len(g)

        if n < min_games:
            continue

        w = 0.5 ** (np.arange(n) / halflife)

        m = g["min"].values

        fp = g["fp"].values

        med = float(np.median(m))

        # expected minute multiplier from blowouts, so we measure only the rest
        b = np.clip((np.abs(g["margin"].values) - o["blow_start"]) / o["blow_span"], 0, 1)

        win = g["margin"].values > 0

        if med >= 26:
            mult = 1 - np.where(win, o["starter_loss_win"], o["starter_loss_lose"]) * b
            role = "starter"

        else:
            gain = o["bench_gain_rot"] if med >= 12 else o["bench_gain_deep"]
            mult = 1 + gain * b
            role = "rotation" if med >= 12 else "deep"

        lm = np.log(np.maximum(m, 1) / (med * mult))

        keep = m >= 8

        lfpm = np.log(np.maximum(fp, 1.0) / m)

        rows.append({
            "key": key,
            "name": g["player_name"].iloc[0],
            "team": g["team_abbreviation"].iloc[0],
            "n": n,
            "role": role,
            "min_mean": float(np.average(m, weights=w)),
            "fp_mean": float(np.average(fp, weights=w)),
            "sd_lm": _wstd(lm, w),
            "sd_lfpm": _wstd(lfpm[keep], w[keep]) if keep.sum() >= 3 else np.nan,
        })

    prof = pd.DataFrame(rows).set_index("key")

    # shrink toward the role average (more games -> trust the player more)
    K = 8.0

    for col in ("sd_lm", "sd_lfpm"):

        prior = prof.groupby("role")[col].transform("mean")

        prior = prior.fillna(prof[col].mean())

        s = prof[col].fillna(prior)

        n_eff = np.minimum(prof["n"], window)

        prof[col] = np.sqrt((n_eff * s ** 2 + K * prior ** 2) / (n_eff + K))

    prof["sigma_m"] = np.clip(prof["sd_lm"], 0.04, 0.45)

    idio = prof["sd_lfpm"] ** 2 - e["var_game"] - e["var_team"]

    floor = (0.45 * prof["sd_lfpm"]) ** 2

    sig_e2 = np.maximum(idio - o["usage_sd"] ** 2 - hot_var, floor)

    prof["sigma_e"] = np.clip(np.sqrt(sig_e2), 0.10, 0.60)

    return prof


def apply_profiles(pool, prof, log=print, flag_pct=0.35):
    """Write measured minutes / volatility onto slate players; flag odd projections."""

    matched = 0

    missing = []

    flags = []

    for p in pool:

        k = match_key(p.name)

        if k not in prof.index:
            missing.append(p.name)
            continue

        r = prof.loc[k]

        matched += 1

        if p.minutes <= 0:
            p.minutes = float(np.clip(r["min_mean"], 6, 42))

        p.sigma_m = float(r["sigma_m"])

        p.sigma_e = float(r["sigma_e"])

        if r["n"] >= 8 and r["fp_mean"] > 5:

            dev = p.projection / r["fp_mean"] - 1

            if abs(dev) > flag_pct:
                flags.append((abs(dev), p.name, p.projection, r["fp_mean"], int(r["n"])))

    log(f"NBA feed: matched {matched}/{len(pool)} slate players to game logs")

    if missing:
        log("  no logs for: " + ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""))

    if flags:

        log("  Projection vs recent output (check these for news / role changes):")

        for dev, name, proj, base, n in sorted(flags, reverse=True)[:10]:
            log(f"    {name:24s} proj {proj:5.1f}  recent avg {base:5.1f}  ({n} games)")

    return matched


# ==================================================
# TEAM MAPPING
# ==================================================

# FanDuel / DFS-site team codes -> nba_api codes
_TEAM_ALIAS = {
    "GS": "GSW", "NO": "NOP", "NY": "NYK", "PHO": "PHX", "SA": "SAS",
    "BRK": "BKN", "CHO": "CHA", "UTH": "UTA", "WSH": "WAS",
}


def build_team_map(pool):
    """{slate team code: nba_api team code} for every team in the pool."""

    out = {}

    for p in pool:

        t = str(p.team).strip().upper()

        out[p.team] = _TEAM_ALIAS.get(t, t)

    return out


def _team_games(logs):
    """One row per team-game: points, fantasy points, points/fantasy allowed."""

    tg = (
        logs.groupby(["team_abbreviation", "game_id"])
        .agg(gdate=("game_date", "first"), pts=("pts", "sum"), fp=("fp", "sum"))
        .reset_index()
    )

    tg["game_pts"] = tg.groupby("game_id")["pts"].transform("sum")

    tg["game_fp"] = tg.groupby("game_id")["fp"].transform("sum")

    tg["fp_allowed"] = tg["game_fp"] - tg["fp"]

    return tg


# ==================================================
# TEAM PROFILES (blowout habits, scoring volatility, opponent variance)
# ==================================================

def compute_team_profiles(logs, league, prof, window=40, k=25.0):
    """
    Per NBA team, relative to the league (1.0 = average):
      blow      how hard the team's starters lose minutes in blowouts
      game_vol  volatility of the game's scoring environment
      opp_var   how variable fantasy output is against this defense
    Each is shrunk toward 1.0 when the team has few games.
    """

    o = league.sim_overrides

    df = logs.copy()

    df["med"] = df.groupby("key")["min"].transform("median")

    cnt = df.groupby("key")["min"].transform("count")

    df = df[cnt >= 10].copy()

    df["role"] = _role(df["med"])

    df["ratio"] = df["min"] / df["med"]

    df["b"] = np.clip(
        (df["margin"].abs() - o["blow_start"]) / o["blow_span"], 0, 1
    )

    league_loss = max(0.5 * (o["starter_loss_win"] + o["starter_loss_lose"]), 0.05)

    tg = _team_games(logs)

    league_pts_sd = float(tg["game_pts"].std())

    tg["fpa_dev"] = tg["fp_allowed"] - tg.groupby("team_abbreviation")["fp_allowed"].transform("mean")

    league_fpa_sd = float(tg["fpa_dev"].std())

    rows = []

    for team, g in tg.sort_values("gdate").groupby("team_abbreviation"):

        g = g.tail(window)

        n = len(g)

        w = n / (n + k)

        # ---- blowout sensitivity of this team's starters
        sub = df[(df["team_abbreviation"] == team) & (df["role"] == "starter")]

        v = _fit_ratio(sub["b"].values, sub["ratio"].values) if len(sub) else None

        raw_blow = 1.0 if v is None else float(np.clip(-v, 0.0, 0.9)) / league_loss

        blow = float(np.clip(1 + w * (raw_blow - 1), 0.6, 1.5))

        # ---- scoring-environment volatility
        raw_vol = float(g["game_pts"].std() / league_pts_sd) if n >= 5 else 1.0

        game_vol = float(np.clip(1 + w * (raw_vol - 1), 0.85, 1.20))

        # ---- variance of fantasy output allowed (as a VARIANCE multiplier)
        raw_opp = float((g["fpa_dev"].std() / league_fpa_sd) ** 2) if n >= 5 else 1.0

        opp_var = float(np.clip(1 + w * (raw_opp - 1), 0.85, 1.25))

        rows.append({
            "team": team, "n": n,
            "blow": blow, "game_vol": game_vol, "opp_var": opp_var,
        })

    return pd.DataFrame(rows).set_index("team")


def apply_team_profiles(pool, tp, team_map, log=print):
    """Write per-team / per-game / per-player multipliers onto the pool."""

    def get(team, col):

        t = team_map.get(team, str(team).upper())

        return float(tp.at[t, col]) if t in tp.index else 1.0

    players = pool.all_players()

    pool.team_blow = np.array([get(t, "blow") for t in pool.team_names], dtype=float)

    pool.game_sd_mult = np.array([
        np.mean([get(t, "game_vol") for t in members]) for members in pool.game_teams
    ], dtype=float)

    pool.opp_var = np.array([get(p.opponent, "opp_var") for p in players], dtype=float)

    hi = sorted(zip(pool.team_blow, pool.team_names), reverse=True)

    log(
        "NBA feed: team profiles applied "
        f"(blowout minute-loss x{pool.team_blow.min():.2f}-{pool.team_blow.max():.2f}; "
        f"most blowout-prone: {', '.join(t for _, t in hi[:3])})"
    )


# ==================================================
# TEAMMATE PAIR CORRELATIONS
# ==================================================

def compute_pair_excess(logs, pool, team_map, min_shared=15, k=20.0):
    """
    Correlation of fantasy-point z-scores between teammates over games they
    shared, minus the league-wide teammate baseline. Returns
    ({(i, j): excess}, baseline) with i, j = pool player indexes.
    """

    players = pool.all_players()

    keys = [match_key(p.name) for p in players]

    d = logs[logs["min"] >= 10].copy()

    g = d.groupby("key")["fp"]

    d["z"] = (d["fp"] - g.transform("mean")) / g.transform("std")

    d = d.dropna(subset=["z"])

    found = []

    by_team = {}

    for i, p in enumerate(players):
        by_team.setdefault(team_map.get(p.team, str(p.team).upper()), []).append(i)

    for team, idxs in by_team.items():

        ks = {keys[i] for i in idxs}

        sub = d[(d["team_abbreviation"] == team) & (d["key"].isin(ks))]

        if sub.empty:
            continue

        wide = sub.pivot_table(index="game_id", columns="key", values="z")

        C = wide.corr(min_periods=min_shared)

        have = wide.notna().astype(int)

        N = have.T @ have

        for a in range(len(idxs)):

            for b in range(a + 1, len(idxs)):

                ka, kb = keys[idxs[a]], keys[idxs[b]]

                if ka == kb or ka not in C.index or kb not in C.index:
                    continue

                c = C.at[ka, kb]

                if pd.isna(c):
                    continue

                found.append((idxs[a], idxs[b], float(c), int(N.at[ka, kb])))

    if not found:
        return {}, 0.0

    wts = np.array([f[3] for f in found], dtype=float)

    baseline = float(np.average([f[2] for f in found], weights=wts))

    excess = {}

    for i, j, c, n in found:

        e = (c - baseline) * n / (n + k)

        if abs(e) >= 0.04:
            excess[(i, j)] = float(np.clip(e, -0.4, 0.4))

    return excess, baseline


# ==================================================
# VEGAS LINE HISTORY (how wrong are the lines?)
# ==================================================

class LineHistory:
    """Saves each slate's spreads/totals so, once games are played, the real
    size of Vegas' misses can replace the default total/margin spread."""

    def __init__(self, path="line_history.csv"):

        self.path = path

    def load(self):

        cols = ["date", "team_a", "team_b", "margin_a", "total"]

        if not os.path.exists(self.path):
            return pd.DataFrame(columns=cols)

        return pd.read_csv(self.path)

    def record(self, pool, team_map, date=None):
        """Save this slate's lines. Returns the number of games saved."""

        date = pd.Timestamp(date or getattr(pool, "slate_date", None) or dt.date.today())

        date = date.normalize().date().isoformat()

        players = pool.all_players()

        rows = []

        for g, members in enumerate(pool.game_teams):

            if len(members) < 2:
                continue

            has_line = any(
                p.total is not None and pool.game_idx[i] == g
                for i, p in enumerate(players)
            )

            if not has_line:
                continue

            a, b = (team_map.get(t, str(t).upper()) for t in members[:2])

            rows.append({
                "date": date, "team_a": a, "team_b": b,
                "margin_a": float(pool.game_margin[g]),
                "total": float(pool.game_total[g]),
            })

        if not rows:
            return 0

        new = pd.DataFrame(rows)

        old = self.load()

        if len(old):
            keep = ~old.set_index(["date", "team_a", "team_b"]).index.isin(
                new.set_index(["date", "team_a", "team_b"]).index
            )
            old = old[keep]

        pd.concat([old, new], ignore_index=True).to_csv(self.path, index=False)

        return len(rows)

    def miss_stats(self, logs, min_games=30):
        """Actual-vs-Vegas error for saved games that have since been played."""

        h = self.load()

        if not len(h):
            return {"n": 0}

        tg = _team_games(logs)

        tg["day"] = tg["gdate"].dt.normalize().dt.date.astype(str)

        pts = {(r.day, r.team_abbreviation): r.pts for r in tg.itertuples()}

        dev_t, dev_m = [], []

        for r in h.itertuples():

            pa, pb = pts.get((r.date, r.team_a)), pts.get((r.date, r.team_b))

            if pa is None or pb is None:
                continue

            dev_t.append((pa + pb) - r.total)

            dev_m.append((pa - pb) - r.margin_a)

        n = len(dev_t)

        out = {"n": n}

        if n >= min_games:

            out.update({
                "total_sd": float(np.std(dev_t, ddof=1)),
                "total_bias": float(np.mean(dev_t)),
                "margin_sd": float(np.std(dev_m, ddof=1)),
            })

        return out


# ==================================================
# WITH / WITHOUT (usage shift when a player is out)
# ==================================================

def usage_shift_report(pool, logs, out_names, team_map, news_weight=0.0,
                       log=print, min_without=3):
    """
    For each starter/rotation player ruled OUT, measure how his teammates
    scored in past games without him versus with him (shrunk for small
    samples), and log it. If news_weight > 0, nudge projections by that
    fraction of the measured shift (capped at +/-25%).
    """

    if not out_names:
        return {}

    players = pool.all_players()

    by_team = {}

    for p in players:
        by_team.setdefault(team_map.get(p.team, str(p.team).upper()), []).append(p)

    shifts = {}

    lines = []

    for nm in out_names:

        k = match_key(nm)

        g = logs[logs["key"] == k]

        if len(g) < 8 or g["min"].median() < 12:
            continue

        team = g.sort_values("game_date")["team_abbreviation"].iloc[-1]

        mates = by_team.get(team)

        if not mates:
            continue

        since = g["game_date"].min()

        tl = logs[(logs["team_abbreviation"] == team) & (logs["game_date"] >= since)]

        played = set(g["game_id"])

        without = set(tl["game_id"]) - played

        if len(without) < min_without:
            lines.append(f"  {nm}: only {len(without)} games without him - no usable sample")
            continue

        lines.append(f"  {nm} OUT ({len(without)} games without him in sample):")

        for p in mates:

            gp = tl[tl["key"] == match_key(p.name)]

            w = gp[gp["game_id"].isin(without)]

            i = gp[gp["game_id"].isin(played)]

            if len(w) < 3 or len(i) < 5:
                continue

            delta = (w["fp"].mean() - i["fp"].mean()) * len(w) / (len(w) + 8.0)

            shifts[p.name] = shifts.get(p.name, 0.0) + float(delta)

            lines.append(
                f"    {p.name:24s} {delta:+5.1f} fp  "
                f"({w['fp'].mean():.1f} without vs {i['fp'].mean():.1f} with)"
            )

    if lines:
        log("NBA feed: with/without report" + (
            " (reporting only; News weight = 0)" if news_weight <= 0 else
            f" (news weight {news_weight:.2f})"))

        for ln in lines:
            log(ln)

    if news_weight > 0 and shifts:

        for p in players:

            s = shifts.get(p.name)

            if s:
                cap = 0.25 * p.projection
                p.projection = float(max(0.5, p.projection + np.clip(news_weight * s, -cap, cap)))

    return shifts


# ==================================================
# RECENCY HALF-LIFE TUNING
# ==================================================

def tune_halflife(logs, candidates=(3, 5, 8, 12, 16, 24, 40), window=25, min_games=12):
    """Walk-forward test: predict each game's fantasy points from the previous
    games with exponential recency weights; lower error = better half-life."""

    errs = {h: [] for h in candidates}

    for key, g in logs.groupby("key"):

        g = g.sort_values("game_date")

        fp = g["fp"].values

        m = g["min"].values

        if len(fp) < min_games:
            continue

        for t in range(8, len(fp)):

            if m[t] < 10:
                continue

            hist = fp[max(0, t - window):t][::-1]       # most recent first

            ages = np.arange(len(hist))

            for h in candidates:

                w = 0.5 ** (ages / h)

                errs[h].append(fp[t] - np.average(hist, weights=w))

    rows = []

    for h in candidates:

        e = np.array(errs[h])

        if len(e):
            rows.append({
                "halflife": h, "mae": float(np.abs(e).mean()),
                "rmse": float(np.sqrt((e ** 2).mean())), "n": len(e),
            })

    table = pd.DataFrame(rows)

    return table, int(table.loc[table["mae"].idxmin(), "halflife"])


# ==================================================
# REFRESHING THE DATA
# ==================================================

def data_status(cache_dir="nba_cache"):
    """What's in the cache right now: latest game date and how old it is."""

    path = os.path.join(cache_dir, f"player_logs_{current_season()}.csv")

    if not os.path.exists(path):
        return {"exists": False, "text": "NBA data: not downloaded yet"}

    try:

        df = pd.read_csv(path, usecols=["GAME_DATE"])

        latest = pd.to_datetime(df["GAME_DATE"]).max().date()

    except Exception:

        try:
            df = pd.read_csv(path)
            latest = pd.to_datetime(df.iloc[:, df.columns.str.lower().get_loc("game_date")]).max().date()
        except Exception:
            return {"exists": True, "text": "NBA data: cache unreadable"}

    age = (dt.date.today() - latest).days

    return {
        "exists": True, "latest": latest, "age_days": age,
        "text": f"NBA data through {latest} ({age} day{'s' if age != 1 else ''} old)",
    }


def refresh(cache_dir="nba_cache", log=print):
    """Force a fresh download of the current season's game logs."""

    feed = NBAFeed(cache_dir, log=log)

    logs = feed.get_logs(force=True)

    log(data_status(cache_dir)["text"])

    return logs


if __name__ == "__main__":

    import sys

    folder = "nba_cache"

    for a in sys.argv[1:]:
        if a.startswith("--cache="):
            folder = a.split("=", 1)[1]

    if "--status" in sys.argv:
        print(data_status(folder)["text"])

    else:
        # default action: refresh (this is what a scheduled task should run)
        refresh(folder)
