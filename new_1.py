"""
Optimizer V3 - NBA DFS (FanDuel) GPP portfolio builder
=======================================================

Pipeline
    Data Loader -> Ownership Engine -> Simulation Engine -> Tournament Metrics
    -> Field Generator -> Candidate Optimizer (MILP) -> Portfolio Selection
    -> Validator -> Portfolio Simulator -> Exporter

Requires:  pip install numpy pandas scipy
Optional:  pip install pulp   (only used if scipy's HiGHS MILP solver is missing)

FIXES APPLIED (search for "# FIX:"):
  1. Settings.max_overlap semantics documented (it means max SHARED, not max unique).
  2. Settings.construction_weight defaulted to 0.0 (the Gaussian kernel on
     near-constant features was likely hurting, not helping).
  3. _calibrate() uses the same seed every iteration (was chasing noise).
  4. PortfolioBuilder.select: int8 -> int32 for member masks (overflow safety
     and faster BLAS path).
  5. FieldGenerator.generate: added log param + explicit under-fill warning.
  6. _lineup_stats: correlation computed per-lineup instead of full (N,N).
  7. OptimalLineups.compute: relaxes max_per_team (was distorting Optimal %).
  8. OwnershipEngine.apply: renormalises total after cap/floor loop.
  9. LineupModel.solve: time_limit / mip_rel_gap are now per-call parameters,
     so candidate generation can use a loose solve and final selection a tight one.
 10. calibration_rows: DNPs are included (PIT=0, dnp=1) instead of silently
     dropped, and calibration_report splits active vs DNP.
 11. CandidateGenerator: logs locked-infeasible fallbacks.
"""

import csv
import datetime as dt
import heapq
import json
import math
import os
import queue
import re
import threading
import time
import traceback
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter.scrolledtext import ScrolledText
except ImportError:  # lets the engines run headless (tests, servers)
    tk = ttk = filedialog = messagebox = ScrolledText = None

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix
    HAVE_HIGHS = True
except ImportError:
    HAVE_HIGHS = False

try:
    import pulp
    HAVE_PULP = True
except ImportError:
    HAVE_PULP = False


# ==================================================
# HELPERS
# ==================================================

def normalize_name(name):
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(name))
        if not unicodedata.combining(c)
    ).strip()


POS_ORDER = ["PG", "SG", "SF", "PF", "C"]
VALID_POS = set(POS_ORDER)

POS_EXPAND = {
    "G": {"PG", "SG"},
    "F": {"SF", "PF"},
    "GF": {"PG", "SG", "SF", "PF"},
    "G/F": {"PG", "SG", "SF", "PF"},
    "FC": {"SF", "PF", "C"},
    "F/C": {"SF", "PF", "C"},
}


def parse_positions(*values):
    out = set()
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        s = str(v).upper().strip()
        if s in ("", "NAN", "NONE"):
            continue
        if s in POS_EXPAND:
            out |= POS_EXPAND[s]
            continue
        for tok in s.replace(",", "/").replace(" ", "").split("/"):
            if tok in VALID_POS:
                out.add(tok)
            elif tok in POS_EXPAND:
                out |= POS_EXPAND[tok]
    return sorted(out, key=POS_ORDER.index)


ALL_POS = {"PG", "SG", "SF", "PF", "C"}

ROSTER_FORMATS = {
    "PG/SG/SF/PF/C + G/F + F/C + 2 UTIL": [
        ("PG", {"PG"}),
        ("SG", {"SG"}),
        ("SF", {"SF"}),
        ("PF", {"PF"}),
        ("C", {"C"}),
        ("G/F", {"PG", "SG", "SF", "PF"}),
        ("F/C", {"SF", "PF", "C"}),
        ("UTIL", ALL_POS),
        ("UTIL", ALL_POS),
    ],
    "Classic 2PG/2SG/2SF/2PF/C": [
        ("PG", {"PG"}), ("PG", {"PG"}),
        ("SG", {"SG"}), ("SG", {"SG"}),
        ("SF", {"SF"}), ("SF", {"SF"}),
        ("PF", {"PF"}), ("PF", {"PF"}),
        ("C", {"C"}),
    ],
}

DEFAULT_FORMAT = "PG/SG/SF/PF/C + G/F + F/C + 2 UTIL"


# ==================================================
# SETTINGS
# ==================================================

@dataclass
class Settings:
    roster_format: str = DEFAULT_FORMAT

    salary_cap: int = 60000
    salary_floor: int = 59000

    lineups: int = 150

    # FIX: max_overlap = maximum SHARED players between any two lineups.
    # With a 9-slot roster: max_overlap=6 means at least 3 unique per pair;
    # max_overlap=8 means at least 1 unique. Lower value = more diverse.
    max_overlap: int = 6
    max_per_team: int = 4
    min_stack: int = 0            # min players from ONE team (0 = off)
    max_exposure: float = 0.35    # global default, 0-1

    sims: int = 5000
    field_size: int = 3000
    candidate_mult: float = 5.0   # candidates generated = lineups * this
    leverage: float = 0.10        # penalty on chalk in the optimizer objective
    randomness: float = 0.10      # noise on the optimizer objective
    min_projection: float = 5.0   # players below this are dropped
    gpp_value_multiple: float = 5.5

    # payout curve (approximation of a top-heavy GPP)
    cash_pct: float = 0.20
    min_cash: float = 1.8
    payout_alpha: float = 0.90
    rake: float = 0.15

    seed: int = None

    use_nba_feed: bool = True
    nba_cache_dir: str = "nba_cache"

    nba_halflife: float = 12.0
    use_pair_corr: bool = True
    use_team_profiles: bool = True
    calibrate_shape: bool = True
    news_weight: float = 0.0
    slate_date: str = ""

    use_news: bool = True
    news_refresh_lines: bool = False
    news_own_boost: float = 0.5
    news_overrides: dict = field(default_factory=dict)
    sim_engine: str = "game"       # "game" | "pbp" | "blend"
    pbp_blend: float = 0.5
    pbp_calib_sims: int = 300

    contest_type: str = "150-Max"

    # FIX: defaulted to 0.0. The historical ConstructionScorer uses a Gaussian
    # kernel on (stud_count, value_count, max_team_stack) which are nearly
    # identical across every slate in the learning DB, so the kernel
    # over-penalises any lineup whose construction differs slightly from the
    # mean. Set this > 0 only after running a backtest that shows it helps.
    construction_weight: float = 0.0

    ownership_error: float = 0.25
    use_proj_correction: bool = False
    proj_corr_min_slates: int = 5
    proj_corr_full_trust: int = 30
    proj_corr_max_pct: float = 0.15
    proj_source2_path: str = ""
    proj_source2_weight: float = 0.0
    use_gb_ownership: bool = False
    gb_override_file: bool = False
    gb_min_slates: int = 3
    gb_full_trust_slates: int = 25
    ownership_groups: int = 6
    optimal_sims: int = 300

    slate_path: str = ""
    template_path: str = ""
    output_path: str = "generated/fanduel_upload.csv"

    player_limits: dict = field(default_factory=dict)

    def slots(self):
        return ROSTER_FORMATS[self.roster_format]


# ==================================================
# PLAYER MODEL
# ==================================================

class Player:

    def __init__(
        self, name, salary, projection, team, opponent, positions,
        minutes=0.0, spread=None, total=None, injury="", player_id=None,
        ownership=None,
    ):
        self.name = name
        self.salary = float(salary)
        self.projection = float(projection)
        self.team = team
        self.opponent = opponent
        if isinstance(positions, (list, tuple, set)):
            self.positions = parse_positions(*positions)
        else:
            self.positions = parse_positions(positions)
        self.position_set = set(self.positions)
        self.minutes = float(minutes or 0)
        self.spread = spread
        self.total = total
        self.injury = injury
        self.player_id = player_id
        self.ownership_given = ownership is not None
        self.ownership = float(ownership) if ownership is not None else 0
        self.ceiling = 0
        self.floor = 0
        self.gpp_probability = 0
        self.win_probability = 0
        self.leverage = 0
        self.simulations = None
        self.min_exposure = 0.0
        self.max_exposure = None
        self.idx = -1
        self.sigma_m = None
        self.sigma_e = None
        self.boom_rel = 1.0
        self.bust_rel = 1.0
        self.hurt_rel = 1.0
        self.nba_team = None
        self.nba_games = 99
        self.recent_fp = None
        self.optimal_pct = 0.0

    def value_score(self):
        return self.projection / max(1, self.salary / 1000)

    def __repr__(self):
        return f"{self.name} | {self.positions} | ${self.salary}"


# ==================================================
# PLAYER POOL
# ==================================================

class PlayerPool:

    def __init__(self):
        self.players = {}
        self._finalized = False

    def add_player(self, player):
        key = player.name
        if key in self.players:
            key = f"{player.name} ({player.team})"
            player.name = key
        self.players[key] = player
        self._finalized = False

    def get_player(self, name):
        return self.players.get(name)

    def all_players(self):
        return list(self.players.values())

    def names(self):
        return list(self.players.keys())

    def __len__(self):
        return len(self.players)

    def __iter__(self):
        return iter(self.players.values())

    def array(self, attr, dtype=float):
        return np.array([getattr(p, attr) for p in self.players.values()], dtype=dtype)

    def finalize(self):
        players = self.all_players()
        for i, p in enumerate(players):
            p.idx = i

        teams = sorted({p.team for p in players})
        self.team_names = teams
        team_lookup = {t: i for i, t in enumerate(teams)}
        self.team_idx = np.array([team_lookup[p.team] for p in players], dtype=int)

        game_lookup = {}
        game_teams = []
        game_idx = []
        for p in players:
            key = frozenset((p.team, p.opponent))
            if key not in game_lookup:
                game_lookup[key] = len(game_teams)
                members = sorted(key - {""}) or [p.team]
                game_teams.append(members)
            game_idx.append(game_lookup[key])

        self.game_idx = np.array(game_idx, dtype=int)
        self.game_teams = game_teams
        G = len(game_teams)

        self.team_sign = np.ones(len(teams))
        for g, members in enumerate(game_teams):
            if len(members) > 1:
                self.team_sign[team_lookup[members[1]]] = -1

        margin = np.zeros(G)
        total = np.full(G, 225.0)
        for g, members in enumerate(game_teams):
            a = members[0]
            b = members[1] if len(members) > 1 else None
            spread_a = [p.spread for p in players if p.team == a and p.spread is not None]
            spread_b = [p.spread for p in players if b and p.team == b and p.spread is not None]
            if spread_a:
                margin[g] = -float(np.mean(spread_a))
            elif spread_b:
                margin[g] = float(np.mean(spread_b))
            totals = [p.total for p in players
                      if p.total is not None and self.game_idx[p.idx] == g]
            if totals:
                total[g] = float(np.mean(totals))

        self.game_margin = margin
        self.game_total = total
        self._finalized = True
        return self

    def to_dataframe(self):
        rows = []
        for p in self.players.values():
            rows.append({
                "name": p.name,
                "pos": "/".join(p.positions),
                "team": p.team,
                "opp": p.opponent,
                "salary": p.salary,
                "projection": p.projection,
                "ownership": p.ownership,
                "ceiling": p.ceiling,
                "floor": p.floor,
                "gpp_prob": p.gpp_probability,
                "leverage": p.leverage,
            })
        return pd.DataFrame(rows)


# ==================================================
# DATA LOADER
# ==================================================

class DataLoader:

    ALIASES = {
        "name": ["name", "nickname", "player", "player name", "full name"],
        "salary": ["salary", "sal", "fd salary", "fd_salary"],
        "projection": [
            "projection", "proj", "fpts", "fp", "projected points",
            "median", "fantasy points", "projection ppg", "points"
        ],
        "team": ["team", "tm"],
        "opp": ["opp", "opponent", "vs"],
        "position": ["position", "pos", "positions"],
        "position_alt": ["position_alt", "alt position", "pos2"],
        "ownership": [
            "ownership", "own", "own%", "ownership%", "proj own",
            "projected ownership", "ownership proj"
        ],
        "injury": ["injury", "injury indicator", "status"],
        "minutes": ["minutes", "min", "mins", "proj min", "projected minutes"],
        "spread": ["spread"],
        "total": ["total", "game total", "o/u", "ou"],
        "id": ["id", "player id", "playerid"],
    }

    FALLBACK = {
        "projection": "proj",
        "ownership": "own",
        "injury": "injur",
    }

    def _find(self, columns, key):
        for alias in self.ALIASES[key]:
            if alias in columns:
                return alias
        frag = self.FALLBACK.get(key)
        if frag:
            for c in columns:
                if frag in c:
                    return c
        return None

    def load_slate(self, path, settings, log=print):
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.strip()
        cols = list(df.columns)
        m = {k: self._find(cols, k) for k in self.ALIASES}

        for required in ("name", "salary", "projection", "team"):
            if m[required] is None:
                raise ValueError(
                    f"Slate file is missing a '{required}' column "
                    f"(columns found: {cols})"
                )

        n0 = len(df)
        outs = []
        if m["injury"]:
            bad = (
                df[m["injury"]].fillna("").astype(str).str.upper()
                .isin(["O", "OUT", "D", "DOUBTFUL", "SUSPENDED", "INJ"])
            )
            outs = [normalize_name(x) for x in df.loc[bad, m["name"]]]
            df = df[~bad]

        def num(key, default=np.nan):
            if m[key] is None:
                return pd.Series(default, index=df.index, dtype=float)
            return pd.to_numeric(df[m[key]], errors="coerce")

        salary = num("salary")
        proj = num("projection").fillna(0)
        own = num("ownership")
        if m["ownership"] is not None and own.max() > 1.5:
            own = own / 100.0
        minutes = num("minutes")
        spread = num("spread")
        total = num("total")

        pool = PlayerPool()
        pool.slate_date = None
        if "game_date" in df.columns:
            _d = pd.to_datetime(df["game_date"], errors="coerce").dropna()
            if len(_d):
                pool.slate_date = _d.iloc[0].normalize()

        dropped = 0
        for i, idx in enumerate(df.index):
            r = df.loc[idx]
            if np.isnan(salary[idx]) or salary[idx] <= 0:
                dropped += 1
                continue
            if proj[idx] < settings.min_projection:
                dropped += 1
                continue
            pos = parse_positions(
                r.get(m["position"]) if m["position"] else None,
                r.get(m["position_alt"]) if m["position_alt"] else None,
            )
            if not pos:
                dropped += 1
                continue
            o = own[idx]
            pl = Player(
                name=normalize_name(r[m["name"]]),
                salary=salary[idx],
                projection=proj[idx],
                team=str(r[m["team"]]).strip().upper(),
                opponent=(
                    str(r[m["opp"]]).strip().upper()
                    if m["opp"] and not pd.isna(r[m["opp"]]) else ""
                ),
                positions=pos,
                minutes=0 if np.isnan(minutes[idx]) else minutes[idx],
                spread=None if np.isnan(spread[idx]) else float(spread[idx]),
                total=None if np.isnan(total[idx]) else float(total[idx]),
                injury=str(r[m["injury"]]) if m["injury"] else "",
                player_id=str(r[m["id"]]) if m["id"] else None,
                ownership=None if (np.isnan(o) or o <= 0) else float(o),
            )
            pl.extra = {
                k: pd.to_numeric(r.get(k), errors="coerce")
                for k in ("l5_dvp_rank", "l5_fppg_avg", "l10_fppg_avg",
                          "szn_fppg_avg", "implied_team_score")
                if k in r.index
            }
            pool.add_player(pl)

        pool.out_names = outs
        pool.finalize()
        log(
            f"Loaded {len(pool)} players "
            f"({n0 - len(df)} injured removed, {dropped} dropped by filters)"
        )
        return pool

    def load_ids(self, path, pool=None):
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.strip()
        cols = list(df.columns)
        name_col = self._find(cols, "name")
        id_col = self._find(cols, "id")
        if name_col is None or id_col is None:
            raise ValueError("Template needs an id column and a nickname/name column")

        pos_col = None
        roster_col = None
        for c in cols:
            lc = c.lower()
            if lc == "position":
                pos_col = c
            if "roster position" in lc:
                roster_col = c

        data = {}
        for _, row in df.iterrows():
            name = normalize_name(row[name_col])
            data[name] = {
                "id": str(row[id_col]),
                "position": str(row[pos_col]).strip().upper()
                if pos_col and not pd.isna(row[pos_col]) else "",
                "roster_position": str(row[roster_col]).strip().upper()
                if roster_col and not pd.isna(row[roster_col]) else "",
            }
        return data


# ==================================================
# OWNERSHIP ENGINE
# ==================================================

class OwnershipEngine:
    """
    Uses ownership from the slate file when present. Otherwise estimates it
    from value, projection, minutes and game environment, then normalizes so
    the slate sums to (number of roster slots) x 100%.
    """

    CAP = 0.65

    def apply(self, pool, n_slots=9, temperature=0.8, log=print,
              gb=None, override=False):

        players = pool.all_players()
        proj = np.array([p.projection for p in players])
        sal = np.array([p.salary for p in players])
        mins = np.array([
            p.minutes if p.minutes > 0 else estimate_minutes(p.projection)
            for p in players
        ])
        tot = pool.game_total[pool.game_idx]

        provided = np.array([p.ownership_given for p in players])
        own = np.array([p.ownership for p in players], dtype=float)

        gb_w = None
        if gb is not None:
            gb_w, gb_total = gb
            n_slots = gb_total
            if override:
                provided = np.zeros(len(players), dtype=bool)
                own = np.zeros(len(players), dtype=float)

        if provided.all():
            log("Ownership: using values from slate file")
            return

        def z(a):
            return (a - a.mean()) / (a.std() + 1e-9)

        value = proj / (sal / 1000)
        raw = (
            1.0 * z(value)
            + 1.0 * z(proj)
            + 0.3 * z(mins)
            + 0.2 * z(tot)
        )
        w = np.exp(raw / temperature)
        if gb_w is not None:
            w = np.asarray(gb_w, dtype=float)

        remaining = max(n_slots - own[provided].sum(), 0.02 * (~provided).sum())
        est = w[~provided]
        own[~provided] = est / est.sum() * remaining

        for _ in range(40):
            over = own > self.CAP
            if not over.any():
                break
            excess = (own[over] - self.CAP).sum()
            own[over] = self.CAP
            free = (~over) & (~provided)
            own[free] += excess * own[free] / own[free].sum()

        own = np.maximum(own, 0.002)

        # FIX: the cap + floor loop can push the total slightly above n_slots
        # (because the 0.002 floor is applied after redistribution). Renormalise
        # so downstream code can safely assume sum(own) == n_slots.
        total = float(own.sum())
        if total > n_slots * 1.01:
            own *= n_slots / total

        for p, o in zip(players, own):
            p.ownership = float(o)

        src = "estimated" if not provided.any() else "partly estimated"
        if gb_w is not None:
            src += " by gradient-boosted model"
        log(f"Ownership: {src} (slate total {own.sum() * 100:.0f}%)")


# ==================================================
# GRADIENT-BOOSTED OWNERSHIP MODEL (optional)
# ==================================================

OWN_DIR = "ownership_data"
OWN_TRAIN_FILE = os.path.join(OWN_DIR, "ownership_training.csv")
OWN_EPS = 0.005

OWN_FEATURES = [
    "salary", "projection", "value", "minutes",
    "game_total", "spread", "implied",
    "sal_pct", "proj_pct", "value_pct", "sal_z", "proj_z", "value_z",
    "proj_over_max", "value_over_max",
    "n_games", "n_players",
    "is_pg", "is_sg", "is_sf", "is_pf", "is_c", "n_pos",
    "pos_proj_pct", "pos_value_pct", "pos_size",
    "dominated_by", "dominates",
    "team_rank", "team_proj_share", "team_pool_size", "team_proj_sum",
    "cheap_flag", "heur_log",
]


def _raw_proj(p):
    r = getattr(p, "raw_projection", None)
    return float(r) if r is not None else float(p.projection)


def heuristic_ownership(pool, n_slots=9, temperature=0.8):
    players = pool.all_players()
    proj = np.array([_raw_proj(p) for p in players], dtype=float)
    sal = np.array([p.salary for p in players], dtype=float)
    mins = np.array([
        p.minutes if p.minutes > 0 else estimate_minutes(_raw_proj(p))
        for p in players
    ])
    tot = pool.game_total[pool.game_idx]

    def z(a):
        return (a - a.mean()) / (a.std() + 1e-9)

    raw = (
        1.0 * z(proj / (sal / 1000)) + 1.0 * z(proj)
        + 0.3 * z(mins) + 0.2 * z(tot)
    )
    w = np.exp(raw / temperature)
    return np.minimum(w / w.sum() * n_slots, 0.65)


def ownership_features(pool, n_slots=9):
    players = pool.all_players()
    n = len(players)
    sal = np.array([p.salary for p in players], dtype=float)
    proj = np.array([_raw_proj(p) for p in players], dtype=float)
    value = proj / np.maximum(sal / 1000, 1.0)
    mins = np.array([
        p.minutes if p.minutes > 0 else estimate_minutes(_raw_proj(p))
        for p in players
    ])
    tot = pool.game_total[pool.game_idx]
    spread = np.array([p.spread if p.spread is not None else 0.0 for p in players])
    implied = tot / 2 - spread / 2

    def pct(a):
        return (pd.Series(a).rank(method="average").values - 1) / max(n - 1, 1)

    def z(a):
        return (a - a.mean()) / (a.std() + 1e-9)

    heur = heuristic_ownership(pool, n_slots)

    A = (sal[None, :] <= sal[:, None]) & (proj[None, :] >= proj[:, None])
    np.fill_diagonal(A, False)
    B = (sal[None, :] >= sal[:, None]) & (proj[None, :] <= proj[:, None])
    np.fill_diagonal(B, False)

    df = pd.DataFrame({
        "team": [p.team for p in players],
        "prim": [p.positions[0] if p.positions else "" for p in players],
        "proj": proj,
        "value": value,
    })

    f = pd.DataFrame(index=range(n))
    f["salary"] = sal
    f["projection"] = proj
    f["value"] = value
    f["minutes"] = mins
    f["game_total"] = tot
    f["spread"] = spread
    f["implied"] = implied
    f["sal_pct"] = pct(sal)
    f["proj_pct"] = pct(proj)
    f["value_pct"] = pct(value)
    f["sal_z"] = z(sal)
    f["proj_z"] = z(proj)
    f["value_z"] = z(value)
    f["proj_over_max"] = proj / max(proj.max(), 1e-9)
    f["value_over_max"] = value / max(value.max(), 1e-9)
    f["n_games"] = len(pool.game_teams)
    f["n_players"] = n

    for pos in ("PG", "SG", "SF", "PF", "C"):
        f[f"is_{pos.lower()}"] = [1 if pos in p.positions else 0 for p in players]

    f["n_pos"] = [len(p.positions) for p in players]

    g = df.groupby("prim")
    f["pos_proj_pct"] = g["proj"].rank(pct=True).values
    f["pos_value_pct"] = g["value"].rank(pct=True).values
    f["pos_size"] = g["proj"].transform("size").values

    f["dominated_by"] = A.sum(axis=1)
    f["dominates"] = B.sum(axis=1)

    gt = df.groupby("team")
    f["team_rank"] = gt["proj"].rank(ascending=False, method="min").values
    f["team_proj_sum"] = gt["proj"].transform("sum").values
    f["team_proj_share"] = proj / np.maximum(f["team_proj_sum"].values, 1e-9)
    f["team_pool_size"] = gt["proj"].transform("size").values
    f["cheap_flag"] = (sal <= 4500).astype(int)
    f["heur_log"] = np.log(heur + OWN_EPS)

    return f[OWN_FEATURES]


def parse_drafted(results_df):
    cols = {str(c).strip().lower(): c for c in results_df.columns}
    if "player" not in cols or "%drafted" not in cols:
        return {}
    out = {}
    for name, val in zip(results_df[cols["player"]], results_df[cols["%drafted"]]):
        if pd.isna(name) or pd.isna(val):
            continue
        v = pd.to_numeric(str(val).replace("%", "").strip(), errors="coerce")
        if pd.notna(v):
            out[name_key(name)] = float(v) / 100.0
    return out


class GBOwnershipModel:
    """Gradient-boosted (sklearn HistGradientBoosting) ownership predictor."""

    _cache = {}

    def __init__(self, path=OWN_TRAIN_FILE, min_slates=3, full_trust_slates=25):
        self.path = path
        self.min_slates = int(min_slates)
        self.full_trust = max(1, int(full_trust_slates))

    def load(self):
        cols = ["slate_id", "name"] + OWN_FEATURES + ["own"]
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=cols)
        return pd.read_csv(self.path)

    def n_slates(self):
        d = self.load()
        return int(d["slate_id"].nunique()) if len(d) else 0

    def save_slate(self, pool, drafted, slate_id, n_slots=9):
        if not drafted:
            return {"saved": 0, "reason": "no %Drafted data in results file"}

        players = pool.all_players()
        keys = [name_key(p.name) for p in players]
        matched = sum(1 for k in keys if k in drafted)
        if matched < max(10, 0.4 * len(players)):
            return {
                "saved": 0,
                "reason": f"only {matched}/{len(players)} pool players matched the results"
            }

        f = ownership_features(pool, n_slots)
        f.insert(0, "name", [p.name for p in players])
        f.insert(0, "slate_id", slate_id)
        f["own"] = [drafted.get(k, 0.0) for k in keys]

        old = self.load()
        if len(old):
            old = old[old["slate_id"] != slate_id]

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        pd.concat([old, f], ignore_index=True).to_csv(self.path, index=False)
        return {"saved": len(f), "matched": matched, "unmatched": len(f) - matched}

    @staticmethod
    def _regressors():
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
        except ImportError:
            raise RuntimeError(
                "scikit-learn is required for the gradient-boosted ownership "
                "model:  pip install scikit-learn"
            )
        return [
            HistGradientBoostingRegressor(
                max_depth=d, learning_rate=0.05, max_iter=250,
                min_samples_leaf=leaf, l2_regularization=1.0,
                early_stopping=False, random_state=seed
            )
            for d, leaf, seed in ((3, 20, 1), (4, 25, 2), (5, 30, 3))
        ]

    def _fit(self, df):
        X = df[OWN_FEATURES].astype(float)
        y = np.log(np.clip(df["own"].astype(float).values, 0, None) + OWN_EPS)
        models = self._regressors()
        for m in models:
            m.fit(X, y)
        return models

    @staticmethod
    def _predict_log(models, X):
        return np.mean([m.predict(X.astype(float)) for m in models], axis=0)

    def _get_models(self, df):
        st = os.stat(self.path)
        key = (os.path.abspath(self.path), st.st_mtime, st.st_size)
        if key not in GBOwnershipModel._cache:
            GBOwnershipModel._cache.clear()
            GBOwnershipModel._cache[key] = self._fit(df)
        return GBOwnershipModel._cache[key]

    def predict(self, pool, n_slots=9, log=print):
        df = self.load()
        n_sl = int(df["slate_id"].nunique()) if len(df) else 0
        if n_sl < self.min_slates:
            log(
                f"GB ownership: only {n_sl} training slate(s) "
                f"(need {self.min_slates}); using default ownership. "
                "Run backtests to build data."
            )
            return None

        models = self._get_models(df)
        X = ownership_features(pool, n_slots)
        gb_log = self._predict_log(models, X)

        w = min(1.0, n_sl / self.full_trust)
        pred_log = w * gb_log + (1 - w) * X["heur_log"].values
        own = np.clip(np.exp(pred_log) - OWN_EPS, 0.002, 0.65)

        totals = df.groupby("slate_id")["own"].sum()
        total = float(np.clip(totals.mean(), 0.5 * n_slots, n_slots))

        log(
            f"GB ownership: model trained on {n_sl} slate(s), "
            f"{len(df):,} player rows (model weight {w:.0%}, "
            f"expected pool total {total * 100:.0f}%)"
        )
        return own, total

    def cross_validate(self, max_slates=40):
        df = self.load()
        slates = list(df["slate_id"].unique()) if len(df) else []
        if len(slates) < 2:
            return (
                f"Ownership model: {len(slates)} slate(s) saved. Need at least 2 "
                "to cross-validate (aim for 25+). Run backtests or use "
                "'Add slate + results' to build data."
            )

        rows = []
        for sid in slates[-max_slates:]:
            tr = df[df["slate_id"] != sid]
            te = df[df["slate_id"] == sid]

            models = self._fit(tr)
            pred = np.clip(np.exp(self._predict_log(models, te[OWN_FEATURES])) - OWN_EPS, 0.002, None)
            heur = np.exp(te["heur_log"].values) - OWN_EPS

            true = te["own"].values.astype(float)
            tot = true.sum()
            pred = pred / pred.sum() * tot
            heur = heur / heur.sum() * tot

            hi = true >= 0.15
            rows.append({
                "slate": sid, "n": len(te),
                "gb_mae": np.abs(pred - true).mean() * 100,
                "def_mae": np.abs(heur - true).mean() * 100,
                "gb_hi": np.abs(pred - true)[hi].mean() * 100 if hi.any() else np.nan,
                "def_hi": np.abs(heur - true)[hi].mean() * 100 if hi.any() else np.nan,
                "gb_corr": np.corrcoef(pred, true)[0, 1],
                "def_corr": np.corrcoef(heur, true)[0, 1],
            })

        r = pd.DataFrame(rows)
        lines = [
            "=== OWNERSHIP MODEL: LEAVE-ONE-SLATE-OUT ===",
            f"Slates: {len(slates)}   Player rows: {len(df):,}",
            "",
            "                          GB model   Default",
            f"MAE, all players (pts)    {r.gb_mae.mean():8.2f}   {r.def_mae.mean():7.2f}",
            f"MAE, owned >=15% (pts)    {r.gb_hi.mean():8.2f}   {r.def_hi.mean():7.2f}",
            f"Correlation               {r.gb_corr.mean():8.3f}   {r.def_corr.mean():7.3f}",
            "",
        ]
        better = (r.gb_mae < r.def_mae).sum()
        lines.append(f"GB beats default on {better}/{len(r)} slates.")
        if len(slates) < 15:
            lines.append(
                "Note: under ~15 slates, results are noisy; the optimizer "
                "blends in the default estimate until the model has more data."
            )
        return "\n".join(lines)


# ==================================================
# SIMULATION ENGINE
# ==================================================

def estimate_minutes(projection):
    return float(38 * (1 - np.exp(-np.asarray(projection, dtype=float) / 18)))


@dataclass
class SimResult:
    sims: np.ndarray
    game_factor: np.ndarray
    starters: np.ndarray
    team_scores: np.ndarray
    game_totals: np.ndarray
    game_margins: np.ndarray
    overtime_flags: np.ndarray


@dataclass
class SimParams:
    pace_sd: float = 0.05
    shoot_sd: float = 0.035
    team_sd: float = 0.04
    margin_sd: float = 12.0
    blow_start: float = 9.0
    blow_span: float = 13.0
    starter_loss_win: float = 0.34
    starter_loss_lose: float = 0.28
    bench_gain_rot: float = 0.45
    bench_gain_deep: float = 0.15
    usage_sd: float = 0.06
    hot_rate: float = 0.08
    ot_rate: float = 0.06
    starter_sigma_m: float = 0.07
    bench_sigma_m: float = 0.20
    bust_rate: float = 0.04
    total_sd: float = 18.0


class SimulationEngine:

    def __init__(self, pool, rng, params=None):
        self.pool = pool
        self.rng = rng
        self.params = params or SimParams()

    def _pair_blocks(self):
        pool = self.pool
        excess = getattr(pool, "pair_excess", None) or {}
        if not excess:
            return []
        by_team = {}
        for (a, b), e in excess.items():
            by_team.setdefault(pool.team_idx[a], []).append((a, b, e))
        blocks = []
        for t, pairs in by_team.items():
            members = sorted({i for a, b, _ in pairs for i in (a, b)})
            pos = {m: k for k, m in enumerate(members)}
            R = np.eye(len(members))
            for a, b, e in pairs:
                R[pos[a], pos[b]] = R[pos[b], pos[a]] = e
            w, v = np.linalg.eigh(R)
            R = (v * np.maximum(w, 0.05)) @ v.T
            d = np.sqrt(np.diag(R))
            R = R / np.outer(d, d)
            blocks.append((np.array(members), np.linalg.cholesky(R)))
        return blocks

    def run(self, S, log=print):
        pool = self.pool
        rng = self.rng
        P = self.params
        f32 = np.float32

        players = pool.all_players()
        N = len(players)
        G = len(pool.game_teams)
        T = len(pool.team_names)

        proj = pool.array("projection")
        sal = pool.array("salary")
        g_idx = pool.game_idx
        t_idx = pool.team_idx

        base_min = np.array([
            p.minutes if p.minutes > 0 else estimate_minutes(p.projection)
            for p in players
        ])
        fpm = proj / np.maximum(base_min, 1)

        starters = np.zeros(N, dtype=bool)
        for t in range(T):
            members = np.where(t_idx == t)[0]
            top = members[np.argsort(-base_min[members])[:5]]
            starters[top] = True

        total_scale = (pool.game_total / 225.0)[None, :]
        game_mult = getattr(pool, "game_sd_mult", np.ones(G))
        team_blow = getattr(pool, "team_blow", np.ones(T))[t_idx][None, :]
        opp_var = getattr(pool, "opp_var", np.ones(N))

        game_sd = float(np.hypot(P.pace_sd, P.shoot_sd))
        z_game = rng.standard_normal((S, G), dtype=f32)
        game_factor = z_game * (game_sd * game_mult[None, :] * total_scale)
        total_dev = z_game * (P.total_sd * game_mult[None, :])
        team_eff = rng.standard_normal((S, T), dtype=f32) * P.team_sd

        margin_a = (
            pool.game_margin[None, :]
            + P.margin_sd * rng.standard_normal((S, G), dtype=f32)
        )
        blow = np.clip(
            (np.abs(margin_a) - P.blow_start) / P.blow_span, 0, 1
        ).astype(f32)

        ot = (rng.random((S, G)) < P.ot_rate).astype(f32)
        team_margin_by_game = margin_a[:, g_idx] * pool.team_sign[t_idx][None, :]
        winning = team_margin_by_game > 0

        b = blow[:, g_idx]
        starter_loss = np.where(winning, P.starter_loss_win, P.starter_loss_lose) * b * team_blow
        bench_gain = np.where(
            base_min >= 12, P.bench_gain_rot, P.bench_gain_deep
        )[None, :] * b * team_blow

        mult = np.where(starters[None, :], 1 - starter_loss, 1 + bench_gain)
        mult = mult * (1 + 0.10 * ot[:, g_idx])

        sig_m = np.array([
            p.sigma_m if p.sigma_m
            else (P.starter_sigma_m if starters[i] else P.bench_sigma_m)
            for i, p in enumerate(players)
        ], dtype=f32)
        mult = mult * np.exp(sig_m * rng.standard_normal((S, N), dtype=f32) - sig_m ** 2 / 2)

        hurt_p = np.clip(0.012 * np.array([p.hurt_rel for p in players]), 0.002, 0.08)
        hurt = rng.random((S, N)) < hurt_p[None, :]
        mult = mult * np.where(hurt, rng.uniform(0.15, 0.55, (S, N)), 1.0)
        minutes = np.clip(base_min[None, :] * mult, 0, 46)

        tier_e = np.select(
            [sal >= 10000, sal >= 8000, sal >= 6500, sal >= 5000],
            [0.20, 0.23, 0.26, 0.30],
            default=0.34,
        )
        sig_e = np.array([
            p.sigma_e if p.sigma_e else tier_e[i]
            for i, p in enumerate(players)
        ], dtype=f32) * np.sqrt(opp_var).astype(f32)

        z = rng.standard_normal((S, N), dtype=f32)
        for members, L in self._pair_blocks():
            z[:, members] = z[:, members] @ L.T.astype(f32)

        usage = rng.standard_normal((S, N), dtype=f32) * P.usage_sd
        w = base_min / base_min.sum()
        onehot = np.zeros((N, T), dtype=f32)
        onehot[np.arange(N), t_idx] = 1
        team_w = (w[:, None] * onehot).sum(0)
        team_mean = (usage * w[None, :]) @ onehot / np.maximum(team_w, 1e-9)
        usage = usage - team_mean[:, t_idx]

        hot_p = np.clip(P.hot_rate * np.array([p.boom_rel for p in players]), 0.005, 0.40)
        hot = (rng.random((S, N)) < hot_p[None, :]) * rng.uniform(0.10, 0.35, (S, N))

        bust_p = np.clip(P.bust_rate * np.array([p.bust_rel for p in players]), 0.0, 0.30)
        bust = (rng.random((S, N)) < bust_p[None, :]) * np.log(rng.uniform(0.25, 0.60, (S, N)))

        log_eff = (
            sig_e * z
            + game_factor[:, g_idx]
            + team_eff[:, t_idx]
            + usage
            + hot
            + bust
        )
        sims = minutes * fpm[None, :] * np.exp(log_eff)
        sims = np.maximum(sims, 0)
        sims *= (proj / np.maximum(sims.mean(0), 1e-6))[None, :]
        sims = sims.astype(f32)

        log(f"Simulated {S:,} slates x {N} players")

        team_scores = np.zeros((S, G, 2), dtype=f32)
        for g in range(G):
            vegas_total = pool.game_total[g] + total_dev[:, g]
            team_a = (vegas_total + margin_a[:, g]) / 2.0
            team_b = (vegas_total - margin_a[:, g]) / 2.0
            team_a += ot[:, g] * rng.uniform(6, 14, S)
            team_b += ot[:, g] * rng.uniform(6, 14, S)
            team_scores[:, g, 0] = team_a
            team_scores[:, g, 1] = team_b

        return SimResult(
            sims=sims.astype(f32),
            game_factor=game_factor.astype(f32),
            starters=starters,
            team_scores=team_scores.astype(f32),
            game_totals=(team_scores[:, :, 0] + team_scores[:, :, 1]).astype(f32),
            game_margins=(team_scores[:, :, 0] - team_scores[:, :, 1]).astype(f32),
            overtime_flags=ot.astype(bool),
        )

    def shape_freq(self, result):
        proj = self.pool.array("projection")[None, :]
        return (
            float((result.sims >= 1.5 * proj).mean()),
            float((result.sims <= 0.55 * proj).mean()),
        )

    def diagnostics(self, result, empirical=None):
        pool = self.pool
        sims = result.sims
        t = pool.team_idx
        g = pool.game_idx
        proj = pool.array("projection")

        top_by_team = {}
        for ti in range(len(pool.team_names)):
            members = np.where(t == ti)[0]
            top_by_team[ti] = members[np.argsort(-proj[members])[:2]]

        team_c = []
        for ti, idx in top_by_team.items():
            if len(idx) == 2:
                team_c.append(np.corrcoef(sims[:, idx[0]], sims[:, idx[1]])[0, 1])

        opp_c = []
        for gi in range(len(pool.game_teams)):
            members = [ti for ti in range(len(pool.team_names))
                       if top_by_team[ti].size and g[top_by_team[ti][0]] == gi]
            if len(members) == 2:
                a = top_by_team[members[0]][0]
                b = top_by_team[members[1]][0]
                opp_c.append(np.corrcoef(sims[:, a], sims[:, b])[0, 1])

        cv = sims.std(0) / np.maximum(sims.mean(0), 1e-6)
        sal = pool.array("salary")
        lines = ["SIMULATION DIAGNOSTICS"]

        def row(label, sim_v, emp_v):
            tail = f"   (real NBA: {emp_v:+.3f})" if emp_v is not None and emp_v == emp_v else ""
            return f"  {label:42s} {sim_v:+.3f}{tail}"

        emp = empirical or {}
        lines.append(row("Top-2 teammate correlation", np.mean(team_c), emp.get("corr_team")))
        lines.append(row("Top-1 opponent correlation", np.mean(opp_c), emp.get("corr_opp")))

        boom, bust = self.shape_freq(result)
        lines.append(row("Big-game frequency (>=1.5x proj)", boom, emp.get("boom_league")).replace("+", " "))
        lines.append(row("Dud frequency (<=0.55x proj)", bust, emp.get("bust_league")).replace("+", " "))

        for lo, hi, label in [(9000, 99999, "$9k+"), (6500, 9000, "$6.5-9k"),
                              (5000, 6500, "$5-6.5k"), (0, 5000, "<$5k")]:
            m = (sal >= lo) & (sal < hi)
            if m.any():
                lines.append(f"  Coefficient of variation {label:8s}: {cv[m].mean():.2f}")

        return "\n".join(lines)

    def summarize_games(self, result):
        summaries = []
        for g, teams in enumerate(self.pool.game_teams):
            team_a = teams[0]
            team_b = teams[1] if len(teams) > 1 else "UNKNOWN"

            a_scores = result.team_scores[:, g, 0]
            b_scores = result.team_scores[:, g, 1]

            summaries.append({
                "game": f"{team_a} vs {team_b}",
                "team_a": team_a, "team_b": team_b,
                "avg_a_score": float(a_scores.mean()),
                "avg_b_score": float(b_scores.mean()),
                "avg_total": float((a_scores + b_scores).mean()),
                "avg_margin": float((a_scores - b_scores).mean()),
                "a_win_pct": float((a_scores > b_scores).mean()),
                "b_win_pct": float((b_scores > a_scores).mean()),
                "ot_pct": float(result.overtime_flags[:, g].mean()),
                "simulations": int(len(a_scores)),
            })
        return summaries


def lineup_scores(sims, idx, chunk=200):
    C = idx.shape[0]
    out = np.empty((C, sims.shape[0]), dtype=np.float32)
    for a in range(0, C, chunk):
        out[a:a + chunk] = sims[:, idx[a:a + chunk]].sum(axis=2).T
    return out


# ==================================================
# TOURNAMENT METRICS
# ==================================================

class OptimalLineups:
    """
    Optimal Lineup %: in each simulated slate, solve for the best possible
    lineup and count how often each player is in it.
    """

    def compute(self, pool, sims, settings, rng, n_sims, ui):
        from dataclasses import replace

        # FIX: relax max_per_team so the "optimal" lineup can stack the best
        # game if the math says so. The actual portfolio is still capped by
        # settings.max_per_team; this is a diagnostic, not a constraint that
        # applies to output.
        model = LineupModel(pool, replace(
            settings,
            salary_floor=0,
            min_stack=0,
            max_per_team=len(settings.slots()),
        ))

        S, N = sims.shape
        sal = pool.array("salary")
        players = pool.all_players()

        by_pos = {
            pos: np.array([i for i, p in enumerate(players) if pos in p.position_set])
            for pos in POS_ORDER
        }

        pick = rng.choice(S, min(n_sims, S), replace=False)
        counts = np.zeros(N)
        scores = []

        for k, s in enumerate(pick):
            pts = sims[s].astype(float)
            keep = set(np.argsort(-pts)[:30]) | set(np.argsort(-(pts / sal))[:30])
            for e in by_pos.values():
                if len(e):
                    keep |= set(e[np.argsort(-pts[e])[:8]]) | set(e[np.argsort(-(pts[e] / sal[e]))[:8]])
            banned = [i for i in range(N) if i not in keep]

            # FIX: time limit + gap tuned for a diagnostic solve.
            lu = model.solve(pts, banned=banned, time_limit=2.0, mip_rel_gap=0.02)
            if lu is None:
                continue
            counts[lu] += 1
            scores.append(pts[lu].sum())
            if k % 25 == 0:
                ui.update_progress(k, len(pick))

        n = max(len(scores), 1)
        for i, p in enumerate(players):
            p.optimal_pct = float(counts[i] / n)

        return np.array(scores)


class TournamentMetrics:

    @staticmethod
    def apply_player_metrics(pool, result, settings):
        sims = result.sims
        S = sims.shape[0]
        sal = pool.array("salary")

        ceil = np.percentile(sims, 95, axis=0)
        flr = np.percentile(sims, 10, axis=0)
        thresh = settings.gpp_value_multiple * sal / 1000
        gpp = (sims >= thresh[None, :]).mean(0)
        win = np.bincount(sims.argmax(1), minlength=sims.shape[1]) / S
        own = pool.array("ownership")

        for i, p in enumerate(pool.all_players()):
            p.ceiling = float(ceil[i])
            p.floor = float(flr[i])
            p.gpp_probability = float(gpp[i])
            p.win_probability = float(win[i])
            p.leverage = float(gpp[i] * 100 / max(own[i] * 100, 0.5))
            p.simulations = sims[:, i]


class PayoutCurve:

    def __init__(self, cash_pct, min_cash, alpha, rake, cap=5000.0):
        self.cash_pct = cash_pct
        self.min_cash = min_cash
        self.alpha = alpha
        self.cap = cap

        grid = (np.arange(200000) + 0.5) / 200000
        self.scale = (1 - rake) / self._raw(grid).mean()

    def _raw(self, q):
        q = np.asarray(q, dtype=np.float64)
        pay = self.min_cash * (self.cash_pct / np.maximum(q, 1e-6)) ** self.alpha
        return np.where(q <= self.cash_pct, np.minimum(pay, self.cap), 0.0)

    def __call__(self, q):
        return (self._raw(q) * self.scale).astype(np.float32)


# ==================================================
# FIELD GENERATOR + PORTFOLIO SIMULATOR
# ==================================================

def perturb_ownership(own, team_idx, sd, rng, cap=0.65):
    if sd <= 0:
        return own
    z = rng.standard_normal(len(own)) * sd
    z += rng.standard_normal(int(team_idx.max()) + 1)[team_idx] * sd * 0.5
    o = np.maximum(own * np.exp(z), 1e-4)
    o *= own.sum() / o.sum()
    return np.minimum(o, cap)


class FieldGenerator:
    """Ownership-weighted synthetic contest field (salary / team legal)."""

    SPEND_TOL = 1500

    # FIX: added log param and under-fill warning. If the generator builds
    # noticeably fewer lineups than requested (tight slate, low max_per_team,
    # etc.) the downstream field-size normalisation inflates every finish
    # percentile and corrupts ROI. Warn loudly.
    def generate(self, pool, settings, rng, F, own=None, log=None):
        players = pool.all_players()
        N = len(players)
        own = np.maximum(pool.array("ownership") if own is None else own, 1e-4)
        sal = pool.array("salary")
        team = pool.team_idx
        T = len(pool.team_names)
        slots = settings.slots()
        ns = len(slots)

        order = sorted(range(ns), key=lambda i: len(slots[i][1]))
        elig = [
            np.array([bool(p.position_set & slots[si][1]) for p in players])
            for si in range(ns)
        ]
        min_sal = sal.min()

        out = np.empty((F, ns), dtype=np.int32)
        f = 0
        tries = 0
        while f < F and tries < F * 30:
            tries += 1
            used = np.zeros(N, dtype=bool)
            tcount = np.zeros(T, dtype=int)
            remaining = settings.salary_cap
            row = [-1] * ns
            ok = True

            for k, si in enumerate(order):
                left = ns - k - 1
                mask = (
                    elig[si] & ~used
                    & (tcount[team] < settings.max_per_team)
                    & (sal <= remaining - left * min_sal)
                )
                if left == 0:
                    mask &= sal >= remaining - self.SPEND_TOL
                if not mask.any():
                    ok = False
                    break
                w = np.where(mask, own, 0.0)
                i = rng.choice(N, p=w / w.sum())
                row[si] = i
                used[i] = True
                tcount[team[i]] += 1
                remaining -= sal[i]

            if ok:
                out[f] = row
                f += 1

        if f < F:
            out = out[:f]
            if log and f < 0.9 * F:
                log(
                    f"WARNING: field generator built only {f}/{F} lineups "
                    "(tight slate, or max_per_team/min_stack is too strict). "
                    "Finish-percentile and ROI estimates will be inflated."
                )
        return out


class PortfolioSimulator:

    def __init__(self, sims, fields, curve):
        if isinstance(fields, np.ndarray):
            fields = [fields]

        K = len(fields)
        S = sims.shape[0]
        self.F = min(len(f) for f in fields)
        self.field_sorted = np.empty((S, self.F), dtype=np.float32)

        for k, f in enumerate(fields):
            sub = np.arange(k, S, K)
            scores = lineup_scores(sims[sub], f[:self.F])
            self.field_sorted[sub] = np.sort(scores.T, axis=1)

        self.curve = curve

    def finish_pct(self, cand_scores):
        C, S = cand_scores.shape
        ahead = np.empty((C, S), dtype=np.int32)
        for s in range(S):
            ahead[:, s] = self.F - np.searchsorted(
                self.field_sorted[s], cand_scores[:, s], side="right"
            )
        return ((ahead + 0.5) / (self.F + 1)).astype(np.float32)

    def payouts(self, q):
        return self.curve(q)


# ==================================================
# LINEUP OPTIMIZER (MILP)
# ==================================================

class LineupModel:
    """
    One lineup MILP, built once, re-solved with a new objective each time.
    """

    def __init__(self, pool, settings):
        self.pool = pool
        self.settings = settings
        self.slots = settings.slots()

        players = pool.all_players()
        N = len(players)
        ns = len(self.slots)

        pairs = [
            (i, si)
            for i, p in enumerate(players)
            for si, (_, elig) in enumerate(self.slots)
            if p.position_set & elig
        ]
        self.pairs = pairs
        n = len(pairs)
        T = len(pool.team_names)
        use_stack = settings.min_stack > 0
        nvars = n + (T if use_stack else 0)

        self.n = n
        self.nvars = nvars
        self.pair_player = np.array([p for p, _ in pairs])
        sal = pool.array("salary")

        rows, cols, vals, lo, hi = [], [], [], [], []
        r = 0

        by_slot = [[] for _ in range(ns)]
        by_player = [[] for _ in range(N)]
        for k, (i, si) in enumerate(pairs):
            by_slot[si].append(k)
            by_player[i].append(k)

        for si in range(ns):
            for k in by_slot[si]:
                rows.append(r); cols.append(k); vals.append(1)
            lo.append(1); hi.append(1); r += 1

        self.player_row0 = r
        for i in range(N):
            for k in by_player[i]:
                rows.append(r); cols.append(k); vals.append(1)
            lo.append(0); hi.append(1); r += 1

        for a in range(ns):
            for b in range(a + 1, ns):
                if self.slots[a][1] == self.slots[b][1]:
                    for k in by_slot[b]:
                        rows.append(r); cols.append(k); vals.append(pairs[k][0] + 1)
                    for k in by_slot[a]:
                        rows.append(r); cols.append(k); vals.append(-(pairs[k][0] + 1))
                    lo.append(1); hi.append(np.inf); r += 1

        for k, (i, _) in enumerate(pairs):
            rows.append(r); cols.append(k); vals.append(sal[i] / 100.0)
        lo.append(settings.salary_floor / 100.0); hi.append(settings.salary_cap / 100.0); r += 1

        team_pairs = [[] for _ in range(T)]
        for k, (i, _) in enumerate(pairs):
            team_pairs[pool.team_idx[i]].append(k)

        for t in range(T):
            for k in team_pairs[t]:
                rows.append(r); cols.append(k); vals.append(1)
            lo.append(0); hi.append(settings.max_per_team); r += 1

        if use_stack:
            for t in range(T):
                for k in team_pairs[t]:
                    rows.append(r); cols.append(k); vals.append(1)
                rows.append(r); cols.append(n + t); vals.append(-settings.min_stack)
                lo.append(0); hi.append(np.inf); r += 1
            for t in range(T):
                rows.append(r); cols.append(n + t); vals.append(1)
            lo.append(1); hi.append(np.inf); r += 1

        self.A = coo_matrix(
            (vals, (rows, cols)), shape=(r, nvars)
        ).tocsr() if HAVE_HIGHS else _csr_fallback(rows, cols, vals, r, nvars)

        self.lo = np.array(lo, dtype=float)
        self.hi = np.array(hi, dtype=float)

    # FIX: time_limit and mip_rel_gap are now per-call parameters, so
    # CandidateGenerator can use a loose solve (1s, 5% gap) and final portfolio
    # selection a tight one (5s, 1% gap). Without this, a single infeasible
    # candidate could burn the full 5s proving infeasibility.
    def solve(self, obj, banned=(), locked=(),
              time_limit=5.0, mip_rel_gap=0.01):
        lo = self.lo.copy()
        hi = self.hi.copy()

        for i in banned:
            hi[self.player_row0 + i] = 0
        for i in locked:
            lo[self.player_row0 + i] = 1

        c = np.zeros(self.nvars)
        c[:self.n] = -obj[self.pair_player]

        x = (
            self._solve_highs(c, lo, hi, time_limit, mip_rel_gap)
            if HAVE_HIGHS else self._solve_pulp(c, lo, hi)
        )
        if x is None:
            return None

        lineup = [-1] * len(self.slots)
        for k in np.where(x[:self.n] > 0.5)[0]:
            i, si = self.pairs[k]
            lineup[si] = i
        return lineup if min(lineup) >= 0 else None

    def _solve_highs(self, c, lo, hi, time_limit, mip_rel_gap):
        res = milp(
            c=c,
            constraints=LinearConstraint(self.A, lo, hi),
            integrality=np.ones(self.nvars),
            bounds=Bounds(0, 1),
            options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap},
        )
        return res.x if res.x is not None else None

    def _solve_pulp(self, c, lo, hi):
        if not HAVE_PULP:
            raise RuntimeError("Install scipy (preferred) or pulp: pip install scipy")
        prob = pulp.LpProblem("lineup", pulp.LpMinimize)
        xs = [pulp.LpVariable(f"x{k}", cat="Binary") for k in range(self.nvars)]
        prob += pulp.lpSum(c[k] * xs[k] for k in range(self.nvars) if c[k] != 0)

        A = self.A
        for r in range(A.shape[0]):
            sl = slice(A.indptr[r], A.indptr[r + 1])
            expr = pulp.lpSum(
                float(v) * xs[j] for j, v in zip(A.indices[sl], A.data[sl])
            )
            if lo[r] == hi[r]:
                prob += expr == lo[r]
            else:
                if lo[r] > -np.inf:
                    prob += expr >= lo[r]
                if hi[r] < np.inf:
                    prob += expr <= hi[r]

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[prob.status] != "Optimal":
            return None
        return np.array([v.varValue or 0 for v in xs])


def _csr_fallback(rows, cols, vals, r, nvars):
    from collections import namedtuple
    M = namedtuple("M", "indptr indices data shape")
    order = sorted(range(len(rows)), key=lambda k: (rows[k], cols[k]))
    indptr = [0] * (r + 1)
    for k in order:
        indptr[rows[k] + 1] += 1
    for k in range(r):
        indptr[k + 1] += indptr[k]
    return M(
        np.asarray(indptr, dtype=np.int64),
        np.asarray([cols[k] for k in order], dtype=np.int64),
        np.asarray([vals[k] for k in order], dtype=np.float64),
        (r, nvars),
    )


class CandidateGenerator:

    def __init__(self, pool, settings, sim, rng, model):
        self.pool = pool
        self.s = settings
        self.sim = sim
        self.rng = rng
        self.model = model

    def _scenario_objective(self):
        rng = self.rng
        sims = self.sim.sims
        S, N = sims.shape
        G = self.sim.game_factor.shape[1]
        r = rng.random()

        if r < 0.55:
            g = rng.integers(G)
            gf = self.sim.game_factor[:, g]
            sub = np.where(gf >= np.percentile(gf, 70))[0]
        elif r < 0.85:
            gp = self.pool.array("gpp_probability") + self.pool.array("optimal_pct") + 1e-4
            i = rng.choice(N, p=gp / gp.sum())
            col = sims[:, i]
            sub = np.where(col >= np.percentile(col, 90))[0]
        else:
            sub = np.arange(S)

        if len(sub) > 400:
            sub = rng.choice(sub, 400, replace=False)
        return sims[sub].mean(0)

    def generate(self, n_target, existing_keys, ui, stop):
        s = self.s
        rng = self.rng
        players = self.pool.all_players()
        N = len(players)
        own = self.pool.array("ownership")
        chalk = own / own.max()

        gen_cap = np.array([
            min(
                0.95,
                2.0 * (p.max_exposure if p.max_exposure is not None
                       else s.max_exposure) + 0.10
            )
            for p in players
        ])
        mins = np.array([p.min_exposure for p in players])
        usage = np.zeros(N)
        keys = set(existing_keys)
        cands = []
        rejected = 0
        attempts = 0

        # FIX: track locked-infeasible fallbacks and report once at the end
        # rather than spamming the log per candidate.
        locked_infeasible = 0

        while len(cands) < n_target and attempts < n_target * 4:
            if stop.is_set():
                break
            attempts += 1

            obj = self._scenario_objective()
            obj = obj * (1 - s.leverage * chalk)
            obj = obj * (1 + s.randomness * rng.standard_normal(N))

            banned = set()
            if len(cands) >= 30:
                share = usage / len(cands)
                banned = set(np.where(share > gen_cap)[0].tolist())

            locked = set(
                np.where((mins > 0) & (rng.random(N) < mins))[0].tolist()
            ) - banned

            # FIX: loose solve for candidates - they only need to be "good",
            # not provably optimal. 1s / 5% gap is plenty and prevents
            # infeasible candidates from blocking for the full 5s each.
            lineup = self.model.solve(
                obj, banned, locked, time_limit=1.0, mip_rel_gap=0.05
            )
            if lineup is None and locked:
                locked_infeasible += 1
                lineup = self.model.solve(
                    obj, banned, (), time_limit=1.0, mip_rel_gap=0.05
                )
            if lineup is None:
                rejected += 1
                continue

            key = tuple(sorted(lineup))
            if key in keys:
                rejected += 1
                continue

            keys.add(key)
            cands.append({"slots": lineup, "key": key})
            usage[list(key)] += 1

            if len(cands) % 25 == 0:
                ui.update_progress(len(cands), n_target)

        if locked_infeasible:
            ui.log(
                f"Candidate generation: {locked_infeasible} candidates could not "
                "satisfy all min-exposure locks and were re-solved without locks "
                "(min-exposure will be softer than requested)."
            )

        return cands, rejected


# ==================================================
# PORTFOLIO SELECTION
# ==================================================

class PortfolioBuilder:

    def select(self, P, member, n, settings, pool, weights=None):
        C, S = P.shape
        N = member.shape[1]
        players = pool.all_players()

        max_cnt = np.array([
            int(math.floor(
                (p.max_exposure if p.max_exposure is not None
                 else settings.max_exposure) * n + 1e-9
            ))
            for p in players
        ])
        min_cnt = np.array([
            int(math.ceil(p.min_exposure * n - 1e-9)) for p in players
        ])

        counts = np.zeros(N, dtype=int)
        best = np.zeros(S, dtype=np.float32)
        mean_pay = P.mean(1)
        boost = mean_pay.max() * 0.3 if min_cnt.any() else 0.0

        # FIX: int8 -> int32. int8 dot products can silently overflow if
        # max_overlap or roster size ever exceeds 127, and small-int BLAS
        # kernels are slower than int32 on most backends.
        member_int = member.astype(np.int32)

        def gain(c):
            g = np.maximum(P[c] - best, 0).mean() + 0.15 * mean_pay[c]
            if boost:
                g += boost * np.count_nonzero(member[c] & (counts < min_cnt))
            return float(g * (1.0 if weights is None else weights[c]))

        heap = [(-gain(c), c) for c in range(C)]
        heapq.heapify(heap)

        sel = []
        sel_member = np.zeros((n, N), dtype=np.int32)

        while heap and len(sel) < n:
            neg, c = heapq.heappop(heap)
            m = member[c]

            if np.any(counts[m] + 1 > max_cnt[m]):
                continue
            if sel and (sel_member[:len(sel)] @ member_int[c]).max() > settings.max_overlap:
                continue

            g = gain(c)
            if heap and g < -heap[0][0] - 1e-12:
                heapq.heappush(heap, (-g, c))
                continue

            sel_member[len(sel)] = member_int[c]
            sel.append(c)
            counts[m] += 1
            best = np.maximum(best, P[c])

        return sel


# ==================================================
# VALIDATOR
# ==================================================

class Validator:

    def validate_lineup(self, pool, slot_idx, settings):
        errors = []
        players = pool.all_players()
        slots = settings.slots()

        if len(slot_idx) != len(slots) or len(set(slot_idx)) != len(slots):
            return ["wrong number of unique players"]

        for (slot, elig), i in zip(slots, slot_idx):
            if not (players[i].position_set & elig):
                errors.append(f"{players[i].name} not eligible for {slot}")

        sal = sum(players[i].salary for i in slot_idx)
        if sal > settings.salary_cap:
            errors.append(f"salary {sal:.0f} over cap")
        if sal < settings.salary_floor:
            errors.append(f"salary {sal:.0f} under floor")

        tc = Counter(players[i].team for i in slot_idx)
        if max(tc.values()) > settings.max_per_team:
            errors.append("too many players from one team")
        if settings.min_stack and max(tc.values()) < settings.min_stack:
            errors.append("minimum stack not met")

        return errors

    def validate_portfolio(self, pool, lineups, settings, id_map=None):
        report = {"errors": [], "warnings": []}
        players = pool.all_players()
        n = len(lineups)
        seen = set()

        for k, lu in enumerate(lineups):
            for e in self.validate_lineup(pool, lu, settings):
                report["errors"].append(f"Lineup {k + 1}: {e}")
            key = tuple(sorted(lu))
            if key in seen:
                report["errors"].append(f"Lineup {k + 1}: duplicate")
            seen.add(key)

            if id_map is not None:
                for i in lu:
                    if normalize_name(players[i].name) not in id_map \
                            and not players[i].player_id:
                        report["errors"].append(
                            f"Lineup {k + 1}: no FanDuel id for {players[i].name}"
                        )

        sets = [set(lu) for lu in lineups]
        worst = 0
        for a in range(n):
            for b in range(a + 1, n):
                worst = max(worst, len(sets[a] & sets[b]))

        if worst > settings.max_overlap:
            report["errors"].append(
                f"max overlap {worst} exceeds limit {settings.max_overlap}"
            )

        counts = Counter(i for lu in lineups for i in lu)
        for i, c in counts.items():
            p = players[i]
            cap = p.max_exposure if p.max_exposure is not None else settings.max_exposure
            if n and c / n > cap + 1e-9:
                report["errors"].append(
                    f"{p.name} exposure {c / n:.0%} over cap {cap:.0%}"
                )

        for p in players:
            if p.min_exposure > 0 and n and counts.get(p.idx, 0) / n + 1e-9 < p.min_exposure:
                report["warnings"].append(
                    f"{p.name} below min exposure "
                    f"({counts.get(p.idx, 0) / n:.0%} < {p.min_exposure:.0%})"
                )

        report["max_overlap"] = worst
        return report


# ==================================================
# CONTEST LEARNING
# ==================================================

class ContestLearningEngine:

    LEGACY_FILE = "contest_learning.csv"

    def __init__(self, contest_type="150-Max"):
        slug = re.sub(r"[^a-z0-9]+", "_", str(contest_type).lower()).strip("_")
        self.DB_FILE = f"contest_learning_{slug}.csv"
        if (
            contest_type == "150-Max"
            and not os.path.exists(self.DB_FILE)
            and os.path.exists(self.LEGACY_FILE)
        ):
            self.DB_FILE = self.LEGACY_FILE

    def save_profile(self, profile):
        exists = os.path.exists(self.DB_FILE)
        with open(self.DB_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow([
                    "slate_id", "salary_used", "ownership_sum",
                    "stud_count", "value_count", "max_team_stack",
                    "avg_game_total", "lineup_score",
                ])
            w.writerow([
                profile.slate_id, profile.salary_used, profile.ownership_sum,
                profile.stud_count, profile.value_count, profile.max_team_stack,
                profile.avg_game_total, profile.lineup_score,
            ])

    def load_profiles(self):
        if not os.path.exists(self.DB_FILE):
            return pd.DataFrame()
        return pd.read_csv(self.DB_FILE)

    def summarize(self):
        df = self.load_profiles()
        if len(df) == 0:
            return None
        return {
            "contests": df["slate_id"].nunique(),
            "avg_salary": df["salary_used"].mean(),
            "salary_std": max(df["salary_used"].std(), 1),
            "avg_ownership": df["ownership_sum"].mean(),
            "ownership_std": max(df["ownership_sum"].std(), 1),
            "avg_studs": df["stud_count"].mean(),
            "avg_values": df["value_count"].mean(),
            "studs_std": max(df["stud_count"].std(), 0.5),
            "avg_stack": df["max_team_stack"].mean(),
            "stack_std": max(df["max_team_stack"].std(), 0.5),
            "avg_game_total": df["avg_game_total"].mean(),
            "game_total_std": max(df["avg_game_total"].std(), 1),
        }

    def remove_slate(self, slate_id):
        if not os.path.exists(self.DB_FILE):
            return
        df = pd.read_csv(self.DB_FILE)
        if "slate_id" not in df.columns:
            return
        df = df[df["slate_id"] != slate_id]
        df.to_csv(self.DB_FILE, index=False)


class ConstructionScorer:

    def __init__(self, contest_type="150-Max"):
        self.hist = ContestLearningEngine(contest_type).summarize()

    def score(self, salary_used, ownership_sum, stud_count,
              max_team_stack, avg_game_total):
        if not self.hist:
            return 50.0

        def z(value, avg, sd):
            return abs(value - avg) / max(sd, 1e-9)

        salary_score = np.exp(-0.5 * z(salary_used, self.hist["avg_salary"], self.hist["salary_std"]))
        ownership_score = np.exp(-0.5 * z(ownership_sum, self.hist["avg_ownership"], self.hist["ownership_std"]))
        stud_score = np.exp(-0.5 * z(stud_count, self.hist["avg_studs"], self.hist["studs_std"]))
        stack_score = np.exp(-0.5 * z(max_team_stack, self.hist["avg_stack"], self.hist["stack_std"]))
        game_score = np.exp(-0.5 * z(avg_game_total, self.hist["avg_game_total"], self.hist["game_total_std"]))

        total = (salary_score + ownership_score + stud_score
                 + stack_score + game_score) / 5
        return round(total * 100, 2)


def construction_features(pool, lu):
    players = pool.all_players()
    teams = Counter(players[i].team for i in lu)
    totals = [players[i].total for i in lu if players[i].total]
    return dict(
        salary_used=float(sum(players[i].salary for i in lu)),
        ownership_sum=float(sum(players[i].ownership for i in lu) * 100),
        stud_count=sum(1 for i in lu if players[i].salary >= 9000),
        max_team_stack=max(teams.values()),
        avg_game_total=float(np.mean(totals)) if totals else 0.0,
    )


# ==================================================
# CALIBRATION (backtest)
# ==================================================

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def name_key(name):
    s = re.sub(r"\s*\([^)]*\)\s*$", "", str(name))
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    ).lower()
    s = re.sub(r"[.'’`]", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(t for t in s.split() if t not in _NAME_SUFFIXES)


# FIX: added 'dnp' column so DNPs are not silently dropped. Old CSVs
# without this column are upgraded on load.
CALIBRATION_COLS = [
    "slate_id", "name", "salary", "minutes", "projection", "actual", "pit", "dnp",
]


def calibration_rows(pool, actual_by_name, slate_id):
    """
    For every pool player with a simulated distribution and a real score,
    return the PIT (probability integral transform). DNPs are included with
    PIT=0 and dnp=1 so the report can show the full picture.

    Returns (DataFrame, info dict).
    """
    actual = {}
    for k, v in actual_by_name.items():
        v = pd.to_numeric(v, errors="coerce")
        if pd.notna(k) and pd.notna(v):
            actual[name_key(k)] = float(v)

    rows = []
    no_sim = unmatched = dnp = 0

    for p in pool.all_players():
        if p.simulations is None:
            no_sim += 1
            continue
        a = actual.get(name_key(p.name))
        if a is None:
            unmatched += 1
            continue

        if a <= 0:
            # FIX: DNP / 0-minute game. The sim conditions on the player being
            # active, so a DNP lands below every sim outcome -> PIT = 0.
            # Include it (flagged) so the report can split DNP rate from
            # active-player calibration instead of hiding DNPs entirely.
            dnp += 1
            rows.append({
                "slate_id": slate_id,
                "name": p.name,
                "salary": p.salary,
                "minutes": p.minutes,
                "projection": p.projection,
                "actual": 0.0,
                "pit": 0.0,
                "dnp": 1,
            })
            continue

        s = np.asarray(p.simulations, dtype=np.float64)
        pit = float((s < a).mean() + 0.5 * (s == a).mean())
        rows.append({
            "slate_id": slate_id,
            "name": p.name,
            "salary": p.salary,
            "minutes": p.minutes,
            "projection": p.projection,
            "actual": a,
            "pit": pit,
            "dnp": 0,
        })

    df = pd.DataFrame(rows, columns=CALIBRATION_COLS)
    return df, {
        "matched": len(df),           # total (includes DNP)
        "active": len(df) - dnp,
        "unmatched": unmatched,
        "dnp": dnp,
        "no_sim": no_sim,
    }


class CalibrationLog:

    DB_FILE = "calibration_history.csv"

    def save(self, df, slate_id):
        old = self.load()
        if len(old):
            old = old[old["slate_id"] != slate_id]
        pd.concat([old, df], ignore_index=True).to_csv(self.DB_FILE, index=False)

    def load(self):
        if not os.path.exists(self.DB_FILE):
            return pd.DataFrame(columns=CALIBRATION_COLS)
        df = pd.read_csv(self.DB_FILE)
        # FIX: back-compat with CSVs written before the 'dnp' column existed.
        if "dnp" not in df.columns:
            df["dnp"] = 0
        return df


def calibration_report(df, title="SIM CALIBRATION"):
    """Text report: is the realized score landing where the sim said it would?"""

    # FIX: split active players from DNPs. DNPs count as PIT=0 and are
    # reported separately; otherwise they'd contaminate the tail statistics.
    has_dnp = "dnp" in df.columns
    if has_dnp:
        dnp_mask = df["dnp"] == 1
        dnp_df = df[dnp_mask].copy()
        active_df = df[~dnp_mask].copy()
    else:
        dnp_df = pd.DataFrame()
        active_df = df

    n = len(active_df)
    n_dnp = len(dnp_df)

    if n < 10:
        if n_dnp < 5:
            return (
                f"{title}\n"
                f"  Not enough matched players ({n} active + {n_dnp} DNP) "
                "to judge calibration."
            )
        return (
            f"{title}\n"
            f"  Only {n} active-player rows (need 10+ for a PIT distribution).\n"
            f"  DNP / 0-minute games: {n_dnp} of {n + n_dnp} matched "
            f"({n_dnp / (n + n_dnp):.1%})"
        )

    pit = active_df["pit"].values
    n_slates = active_df["slate_id"].nunique() if "slate_id" in active_df else 1

    def share(mask):
        return float(np.mean(mask))

    def flag(obs, exp, count):
        se = math.sqrt(exp * (1 - exp) / max(count, 1))
        return "  <-- off" if abs(obs - exp) > 2.5 * se else ""

    lines = [
        title,
        "-" * 60,
        f"  {n} active player-games from {n_slates} slate(s)",
        f"  Mean percentile of actual score: {pit.mean():.3f}   (0.500 = unbiased)",
        "",
        "  Share of actual scores in the sim's tails (ideal in brackets):",
    ]

    for label, mask, exp in [
        ("above 95th percentile", pit > 0.95, 0.05),
        ("above 90th percentile", pit > 0.90, 0.10),
        ("below 10th percentile", pit < 0.10, 0.10),
        ("below  5th percentile", pit < 0.05, 0.05),
    ]:
        obs = share(mask)
        lines.append(f"    {label:24s} {obs:6.1%}  [{exp:.0%}]{flag(obs, exp, n)}")

    lines += ["", "  Where actual scores landed, by sim decile (ideal 10% each):"]
    hist, _ = np.histogram(pit, bins=10, range=(0, 1))
    for k, c in enumerate(hist):
        frac = c / n
        lines.append(
            f"    {k * 10:3d}-{(k + 1) * 10:3d}%  {frac:6.1%}  "
            + "#" * int(round(frac * 100 / 2))
        )

    def split(label, groups):
        out = ["", f"  By {label}:   (n, mean pct, >p90, <p10, avg actual - proj)"]
        for name, mask in groups:
            m = np.asarray(mask)
            if m.sum() < 8:
                continue
            sub = active_df[m]
            out.append(
                f"    {name:14s} n={int(m.sum()):4d}  {sub['pit'].mean():.3f}  "
                f"{(sub['pit'] > 0.9).mean():5.1%}  {(sub['pit'] < 0.1).mean():5.1%}  "
                f"{(sub['actual'] - sub['projection']).mean():+6.2f}"
            )
        return out if len(out) > 2 else []

    sal = active_df["salary"].values
    lines += split("salary", [
        ("$9k+", sal >= 9000),
        ("$6.5k-9k", (sal >= 6500) & (sal < 9000)),
        ("$5k-6.5k", (sal >= 5000) & (sal < 6500)),
        ("under $5k", sal < 5000),
    ])

    mins = active_df["minutes"].values
    lines += split("role (projected minutes)", [
        ("26+ min", mins >= 26),
        ("under 26 min", (mins > 0) & (mins < 26)),
    ])

    hi, lo = share(pit > 0.90), share(pit < 0.10)
    lines += ["", "  Read:"]

    if hi > 0.10 + 2.5 * math.sqrt(0.09 / n) and lo <= 0.10:
        lines.append("    Ceilings look too low: big games happen more often than the sim says.")
    elif lo > 0.10 + 2.5 * math.sqrt(0.09 / n) and hi <= 0.10:
        lines.append("    Floors look too high or projections inflated: duds happen more than the sim says.")
    elif hi > 0.10 + 2.5 * math.sqrt(0.09 / n) and lo > 0.10 + 2.5 * math.sqrt(0.09 / n):
        lines.append("    Sim is too narrow: outcomes land in both tails more than expected.")
    elif hi < 0.10 - 2.5 * math.sqrt(0.09 / n) and lo < 0.10 - 2.5 * math.sqrt(0.09 / n):
        lines.append("    Sim is too wide: outcomes cluster in the middle more than expected.")
    else:
        lines.append("    No tail miscalibration beyond noise.")

    z_mean = (pit.mean() - 0.5) / math.sqrt(1.0 / 12.0 / n)
    if z_mean < -2.0:
        lines.append(
            f"    Actual scores run below the sim's middle (mean pct {pit.mean():.2f}): "
            "projections look inflated."
        )
    elif z_mean > 2.0:
        lines.append(
            f"    Actual scores run above the sim's middle (mean pct {pit.mean():.2f}): "
            "projections look low."
        )

    # FIX: DNP section
    if n_dnp >= 5:
        n_all = n + n_dnp
        lines += [
            "",
            f"  DNP / 0-minute games: {n_dnp} of {n_all} matched ({n_dnp / n_all:.1%})",
            "  These count as PIT = 0 and are excluded from the active-player distribution above.",
            "  If this rate looks high, your projections are not accounting for injuries / rotations.",
        ]

    if n_slates < 5:
        lines.append(
            f"    Only {n_slates} slate(s) so far; a single night is dominated by "
            "game-environment luck."
        )

    return "\n".join(lines)


# ==================================================
# EXPORTER
# ==================================================

class Exporter:

    def export(self, pool, lineups, settings, id_map, path):
        players = pool.all_players()
        slots = settings.slots()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        written = 0

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([name for name, _ in slots])
            for lu in lineups:
                row = []
                for i in lu:
                    p = players[i]
                    fd = id_map.get(normalize_name(p.name))
                    pid = p.player_id or (fd["id"] if isinstance(fd, dict) else fd)
                    if not pid:
                        raise KeyError(f"No FanDuel id for {p.name}")
                    row.append(pid)
                w.writerow(row)
                written += 1
        return written


# ==================================================
# RESULTS + PIPELINE
# ==================================================

@dataclass
class Results:
    pool: object = None
    lineups: list = field(default_factory=list)
    lineup_stats: list = field(default_factory=list)
    report: str = ""
    validation: dict = field(default_factory=dict)
    id_map: dict = field(default_factory=dict)


@dataclass
class HistoricalContestProfile:
    slate_id: str
    salary_used: float
    ownership_sum: float
    stud_count: int
    value_count: int
    max_team_stack: int
    avg_game_total: float
    lineup_score: float


class ConsoleUI:

    def log(self, msg): print(msg)
    def update_progress(self, cur, total): pass
    def set_status(self, s): print(f"[{s}]")
    def post(self, kind, payload=None): pass
    def stats(self, **kw): pass


class Pipeline:

    def __init__(self, settings, ui, stop_event=None):
        self.s = settings
        self.ui = ui
        self.stop = stop_event or threading.Event()

    def _apply_limits(self, pool):
        for p in pool:
            lim = self.s.player_limits.get(p.name)
            p.min_exposure = 0.0
            p.max_exposure = None
            if lim:
                p.min_exposure = float(lim[0] or 0)
                p.max_exposure = None if lim[1] is None else float(lim[1])
                if p.min_exposure > (
                    p.max_exposure if p.max_exposure is not None
                    else self.s.max_exposure
                ):
                    p.max_exposure = p.min_exposure

    def _try(self, ui, label, fn):
        try:
            return fn()
        except Exception as e:
            ui.log(f"{label} skipped ({e})")

    def _simulate(self, engine, pool, rng, logs, params, ui):
        s = self.s
        mode = s.sim_engine if s.sim_engine in ("game", "pbp", "blend") else "game"
        if mode == "game":
            return engine.run(s.sims, ui.log)

        try:
            from pbp_engine import PBPEngine, compute_pbp_profiles
            prof = {}
            if logs is not None:
                prof = compute_pbp_profiles(logs, halflife=s.nba_halflife)
                ui.log(f"PBP: measured shot/stat profiles for {len(prof)} players")

            n_pbp = s.sims if mode == "pbp" else int(round(s.sims * min(max(s.pbp_blend, 0.0), 1.0)))
            n_game = s.sims - n_pbp

            ui.set_status("Play-by-play simulation")
            t0 = time.time()
            raw = PBPEngine(
                pool, rng, prof, calib_sims=s.pbp_calib_sims
            ).run(max(n_pbp, 1), ui.log)
            ts = raw["team_scores"]
            pbp = SimResult(
                sims=raw["sims"], game_factor=raw["game_factor"], starters=raw["starters"],
                team_scores=ts, game_totals=ts.sum(axis=2),
                game_margins=ts[:, :, 0] - ts[:, :, 1],
                overtime_flags=raw["overtime_flags"],
            )
            ui.log(f"PBP: {n_pbp:,} possession-level sims in {time.time() - t0:.0f}s")

            if mode == "pbp" or n_game <= 0:
                return pbp

            g = engine.run(n_game, ui.log)
            return SimResult(
                sims=np.concatenate([g.sims, pbp.sims]),
                game_factor=np.concatenate([g.game_factor, pbp.game_factor]),
                starters=g.starters,
                team_scores=np.concatenate([g.team_scores, pbp.team_scores]),
                game_totals=np.concatenate([g.game_totals, pbp.game_totals]),
                game_margins=np.concatenate([g.game_margins, pbp.game_margins]),
                overtime_flags=np.concatenate([g.overtime_flags, pbp.overtime_flags]),
            )
        except Exception as e:
            ui.log(f"Play-by-play engine failed ({e}); using the original simulation")
            traceback.print_exc()
            return engine.run(s.sims, ui.log)

    def _calibrate(self, pool, params, emp, ui):
        """Nudge big-game / dud-game rates until the sim's boom and bust
        frequencies match what real NBA player-games show."""
        boom = bust = 0.0
        for it in range(4):
            # FIX: same seed every iteration. Previously each pass used a
            # different rng, so the multiplicative hot/bust update was
            # chasing sampling noise for the first ~2 passes.
            eng = SimulationEngine(pool, np.random.default_rng(42), params)
            r = eng.run(1500, log=lambda *a: None)
            boom, bust = eng.shape_freq(r)
            if boom <= 0 or bust <= 0:
                break
            params.hot_rate = float(np.clip(
                params.hot_rate * (emp["boom_league"] / boom) ** 1.0, 0.005, 0.30))
            params.bust_rate = float(np.clip(
                params.bust_rate * (emp["bust_league"] / bust) ** 1.0, 0.002, 0.25))

        ui.log(
            f"Shape calibration: hot_rate {params.hot_rate:.3f}, bust_rate "
            f"{params.bust_rate:.3f} (before last step sim boom {boom:.3f} vs real "
            f"{emp['boom_league']:.3f}, bust {bust:.3f} vs real {emp['bust_league']:.3f})"
        )

    def _construction_weights(self, pool, cands, s):
        if s.construction_weight <= 0:
            return None
        scorer = ConstructionScorer(s.contest_type)
        if not scorer.hist:
            return None
        sc = np.array([
            scorer.score(**construction_features(pool, c["slots"])) for c in cands
        ])
        f = np.clip(sc / max(np.median(sc), 1e-6), 0.5, 1.5)
        return 1 + s.construction_weight * (f - 1)

    def run(self, pool=None):
        s = self.s
        ui = self.ui
        t0 = time.time()
        rng = np.random.default_rng(s.seed)
        loader = DataLoader()

        # ---- 1. data
        ui.set_status("Loading data")
        if pool is None:
            pool = loader.load_slate(s.slate_path, s, ui.log)

        id_map = {}
        if s.template_path and os.path.exists(s.template_path):
            id_map = loader.load_ids(s.template_path)
            ui.log(f"Loaded {len(id_map)} FanDuel ids from template")

        for p in pool:
            fd = id_map.get(normalize_name(p.name))
            if not fd:
                continue
            p.player_id = fd["id"]
            if fd["position"]:
                p.positions = parse_positions(fd["position"])
                p.position_set = set(p.positions)

        self._apply_limits(pool)
        pool.finalize()
        n_slots = len(s.slots())
        if len(pool) < n_slots + 5:
            raise ValueError("Not enough players in the pool")

        # ---- 1b. live news
        nn = None
        news_rep = None
        if s.use_news:
            ui.set_status("News")
            try:
                import nba_news as nn
                snap = nn.NewsFeed(log=ui.log).get(force=True)
                news_rep = nn.apply_news(
                    pool, snap, s.news_overrides,
                    update_lines=s.news_refresh_lines, log=ui.log,
                )
                ui.post("news", (snap, news_rep, pool))
            except Exception as e:
                ui.log(f"News skipped ({e})")
                nn = None

            if len(pool) < n_slots + 5:
                raise ValueError("Not enough players in the pool after news removals")

        # ---- 2. ownership
        ui.set_status("Ownership")
        gb_pred = None
        if s.use_gb_ownership:
            try:
                gb_pred = GBOwnershipModel(
                    min_slates=s.gb_min_slates,
                    full_trust_slates=s.gb_full_trust_slates,
                ).predict(pool, n_slots, ui.log)
            except Exception as e:
                ui.log(f"GB ownership unavailable ({e}); using default ownership")

        OwnershipEngine().apply(
            pool, n_slots, log=ui.log,
            gb=gb_pred, override=s.gb_override_file,
        )

        if nn is not None and news_rep is not None and s.news_own_boost > 0:
            self._try(ui, "News ownership", lambda: nn.boost_ownership(
                pool, news_rep, s.news_own_boost, ui.log))

        # ---- 3. simulation
        ui.set_status("Simulating")
        params = SimParams()
        empirical = None
        logs = None

        if s.use_nba_feed:
            try:
                from dataclasses import asdict
                import nba_feed as nf

                logs = nf.NBAFeed(s.nba_cache_dir, log=ui.log).get_logs()
                league = nf.compute_league_params(logs, asdict(params))
                params = SimParams(**league.sim_overrides)
                ui.log(league.summary())

                prof = nf.compute_profiles(logs, league, halflife=s.nba_halflife)
                nf.apply_profiles(pool, prof, ui.log)
                empirical = league.extra
                team_map = nf.build_team_map(pool)
            except Exception as e:
                ui.log(f"NBA feed unavailable ({e}) - using default sim parameters")
                logs = None

        for p in pool:
            if getattr(p, "raw_projection", None) is None:
                p.raw_projection = p.projection
            else:
                p.projection = p.raw_projection

        if s.use_proj_correction:
            try:
                ProjectionModel(
                    min_slates=s.proj_corr_min_slates,
                    full_trust_slates=s.proj_corr_full_trust,
                    max_pct=s.proj_corr_max_pct,
                ).apply(
                    pool,
                    logs if logs is not None else load_cached_logs(s.nba_cache_dir),
                    ui.log, n_slots,
                )
            except Exception as e:
                ui.log(f"Projection correction unavailable ({e}); projections unchanged")

        if s.proj_source2_weight > 0 and s.proj_source2_path:
            try:
                blend_projection_source(
                    pool, s.proj_source2_path, s.proj_source2_weight, ui.log
                )
            except Exception as e:
                ui.log(f"Projection source 2 failed ({e}); using main projections only")

        if logs is not None:
            if s.use_team_profiles:
                def _team():
                    tp = nf.compute_team_profiles(logs, league, prof)
                    nf.apply_team_profiles(pool, tp, team_map, ui.log)
                self._try(ui, "Team profiles", _team)

            if s.use_pair_corr:
                def _pairs():
                    excess, base = nf.compute_pair_excess(logs, pool, team_map)
                    pool.pair_excess = excess
                    ui.log(
                        f"Pair correlations: {len(excess)} teammate pairs learned "
                        f"(league baseline {base:+.3f})"
                    )
                self._try(ui, "Pair correlations", _pairs)

            def _lines():
                hist = nf.LineHistory()
                saved = hist.record(pool, team_map, s.slate_date or None)
                st = hist.miss_stats(logs)
                if st and "total_sd" in st:
                    params.total_sd = st["total_sd"]
                    params.margin_sd = st["margin_sd"]
                    ui.log(
                        f"Vegas miss (from {st['n']} saved games): total sd "
                        f"{st['total_sd']:.1f} (bias {st['total_bias']:+.1f}), "
                        f"margin sd {st['margin_sd']:.1f}"
                    )
                else:
                    n_have = st["n"] if st else 0
                    ui.log(
                        f"Vegas miss: {saved} games saved to line_history.csv "
                        f"({n_have} matched so far; need 30) - using default total sd "
                        f"{params.total_sd:.0f}"
                    )
            self._try(ui, "Line history", _lines)

            if getattr(pool, "out_names", None):
                self._try(ui, "Usage shift", lambda: nf.usage_shift_report(
                    pool, logs, pool.out_names, team_map, s.news_weight, ui.log))

            if s.calibrate_shape and empirical:
                self._try(ui, "Shape calibration", lambda: self._calibrate(pool, params, empirical, ui))

        engine = SimulationEngine(pool, rng, params)
        sim = self._simulate(engine, pool, rng, logs, params, ui)

        if nn is not None:
            self._try(ui, "Availability", lambda: nn.apply_availability(
                pool, sim.sims, rng, log=ui.log))

        game_summaries = engine.summarize_games(sim)
        ui.post("game_summary", game_summaries)
        for g in game_summaries:
            ui.log(
                f"{g['game']} | {g['avg_a_score']:.1f}-{g['avg_b_score']:.1f} | "
                f"Win% {g['a_win_pct']:.1%}/{g['b_win_pct']:.1%} | "
                f"OT {g['ot_pct']:.1%} | {g['simulations']} sims"
            )

        TournamentMetrics.apply_player_metrics(pool, sim, s)
        diag = engine.diagnostics(sim, empirical)
        ui.log(diag)

        self.opt_scores = None
        if s.optimal_sims > 0:
            ui.set_status("Optimal lineup %")
            self.opt_scores = OptimalLineups().compute(
                pool, sim.sims, s, rng, s.optimal_sims, ui
            )
            for p in pool:
                p.leverage = p.optimal_pct * 100 / max(p.ownership * 100, 0.5)
            top = sorted(pool, key=lambda p: -p.optimal_pct)[:8]
            ui.log("Optimal lineup %: "
                   + ", ".join(f"{p.name} {p.optimal_pct:.0%}" for p in top))

        ui.post("pool", pool)
        if self.stop.is_set():
            return None

        # ---- 4. contest field
        ui.set_status("Building contest field")
        K = max(1, s.ownership_groups) if s.ownership_error > 0 else 1
        own_base = np.maximum(pool.array("ownership"), 1e-4)
        fields = []
        for _ in range(K):
            own_k = perturb_ownership(own_base, pool.team_idx, s.ownership_error, rng)
            fields.append(FieldGenerator().generate(
                pool, s, rng, s.field_size, own_k, log=ui.log,
            ))
        field_idx = fields[0]

        curve = PayoutCurve(s.cash_pct, s.min_cash, s.payout_alpha, s.rake)
        psim = PortfolioSimulator(sim.sims, fields, curve)
        ui.log(
            f"Field: {K} ownership scenario(s) x {len(field_idx):,} synthetic lineups "
            f"(ownership error {s.ownership_error:.2f})"
        )

        # ---- 5. candidates + selection
        ui.set_status("Optimizing")
        solver = "HiGHS (scipy)" if HAVE_HIGHS else "CBC (pulp)"
        ui.log(f"MILP solver: {solver}")

        model = LineupModel(pool, s)
        gen = CandidateGenerator(pool, s, sim, rng, model)
        n = s.lineups
        target = int(n * s.candidate_mult)

        cands = []
        total_rejected = 0
        q_all = np.empty((0, s.sims), dtype=np.float32)
        chosen = []

        for rnd in range(3):
            new, rej = gen.generate(
                target if rnd == 0 else n * 2,
                [c["key"] for c in cands], ui, self.stop,
            )
            total_rejected += rej
            if not new:
                break

            cands += new
            ui.log(f"Candidates: {len(cands)} (round {rnd + 1})")
            ui.set_status("Scoring candidates vs field")

            idx_new = np.array([c["slots"] for c in new], dtype=np.int32)
            q_new = psim.finish_pct(lineup_scores(sim.sims, idx_new))
            q_all = np.vstack([q_all, q_new])
            P = psim.payouts(q_all)

            member = np.zeros((len(cands), len(pool)), dtype=bool)
            for r, c in enumerate(cands):
                member[r, list(c["key"])] = True

            ui.set_status("Selecting portfolio")
            weights = self._construction_weights(pool, cands, s)
            chosen = PortfolioBuilder().select(P, member, n, s, pool, weights)

            ui.stats(lineups=len(cands), accepted=len(chosen), rejected=total_rejected)

            if len(chosen) >= n or self.stop.is_set():
                break
            ui.log(
                f"Only {len(chosen)}/{n} lineups fit the constraints - "
                "generating more candidates"
            )

        if 0 < len(chosen) < n:
            for n_try in range(len(chosen), 0, -1):
                c2 = PortfolioBuilder().select(P, member, n_try, s, pool, weights)
                if len(c2) >= n_try:
                    chosen = c2[:n_try]
                    break
            else:
                ui.log(
                    "Could not satisfy every exposure cap at the smaller size - "
                    "caps may be slightly exceeded (see validation)."
                )

        if len(chosen) < n:
            ui.log(
                f"Built {len(chosen)} of {n} lineups under your limits. "
                "Raise candidate multiplier, max exposure, or max overlap for more."
            )

        if not chosen:
            raise RuntimeError(
                "No lineups could be built - check salary floor, exposure "
                "caps and roster format"
            )

        lineups = [cands[c]["slots"] for c in chosen]

        # ---- 6. validate
        ui.set_status("Validating")
        validation = Validator().validate_portfolio(
            pool, lineups, s, id_map if id_map else None
        )
        for e in validation["errors"][:20]:
            ui.log(f"VALIDATION ERROR: {e}")
        for w in validation["warnings"][:20]:
            ui.log(f"warning: {w}")
        ui.log(
            "Validation: "
            + ("PASSED" if not validation["errors"]
               else f"{len(validation['errors'])} problem(s)")
        )

        # ---- 7. portfolio simulation
        ui.set_status("Simulating portfolio")
        q = q_all[chosen]
        pay = psim.payouts(q)
        stats = self._lineup_stats(pool, lineups, q, pay, sim)
        report = self._report(pool, lineups, q, pay, stats, diag, s, time.time() - t0)

        res = Results(pool, lineups, stats, report, validation, id_map)
        ui.post("report", report)
        ui.post("done", res)
        ui.set_status(f"Done - {len(lineups)} lineups in {time.time() - t0:.0f}s")
        return res

    def _lineup_stats(self, pool, lineups, q, pay, sim):
        players = pool.all_players()
        proj = pool.array("projection")
        sal = pool.array("salary")
        own = pool.array("ownership")

        scorer = ConstructionScorer(self.s.contest_type)
        out = []

        for k, lu in enumerate(lineups):
            salary_used = float(sal[lu].sum())
            ownership_sum = float(own[lu].sum() * 100)
            team_counts = Counter(players[i].team for i in lu)
            max_team_stack = max(team_counts.values())
            stud_count = sum(1 for i in lu if players[i].salary >= 9000)
            game_totals = [players[i].total for i in lu if players[i].total]
            avg_game_total = np.mean(game_totals) if game_totals else 0

            construction_score = scorer.score(
                salary_used=salary_used,
                ownership_sum=ownership_sum,
                stud_count=stud_count,
                max_team_stack=max_team_stack,
                avg_game_total=avg_game_total,
            )

            out.append({
                "salary": salary_used,
                "proj": float(proj[lu].sum()),
                "own": ownership_sum,
                "top1": float((q[k] <= 0.01).mean()),
                "roi": float(pay[k].mean() - 1),
                "teams": max_team_stack,
                "construction": construction_score,
            })

        # FIX: compute correlation per-lineup instead of the full (N, N)
        # matrix. The old approach was O(N^2 * S); this is O(lineups * k^2 * S)
        # which is ~10x faster on a normal-size slate and avoids the (N,N)
        # allocation entirely.
        iu = np.triu_indices(len(lineups[0]), 1)
        for k, lu in enumerate(lineups):
            sub_sims = sim.sims[:, lu]
            c = np.corrcoef(sub_sims.T)
            sub = c[iu]
            lineup_sd = float(sub_sims.sum(1).std())
            indep_sd = float(np.sqrt(sub_sims.var(0).sum()))
            out[k]["corr"] = float(sub.mean())
            out[k]["lift"] = lineup_sd / max(indep_sd, 1e-9)

        return out

    def _report(self, pool, lineups, q, pay, stats, diag, s, elapsed):
        n = len(lineups)
        players = pool.all_players()
        port = pay.sum(0) / n
        any1 = (q.min(0) <= 0.01).mean()
        any01 = (q.min(0) <= 0.001).mean()
        counts = Counter(i for lu in lineups for i in lu)
        top = counts.most_common(12)
        stack = Counter(st["teams"] for st in stats)

        lines = [
            "PORTFOLIO SIMULATION",
            "-" * 60,
            f"Lineups:                      {n}",
            f"Sims / field size:            {s.sims:,} / {s.field_size:,}",
            f"Expected ROI (model):         {port.mean() - 1:+.1%}",
            f"P(at least one top 1% lineup):   {any1:.1%}",
            f"P(at least one top 0.1% lineup): {any01:.1%}",
            f"Median best finish percentile:   {np.median(q.min(0)):.2%}",
            f"Avg lineup projection:        {np.mean([x['proj'] for x in stats]):.1f}",
            f"Avg lineup ownership sum:     {np.mean([x['own'] for x in stats]):.0f}%",
            f"Avg salary used:              ${np.mean([x['salary'] for x in stats]):,.0f}",
            f"Unique players used:          {len(counts)}",
            f"Avg construction score:       {np.mean([x['construction'] for x in stats]):.0f}"
            f"   (weight {s.construction_weight:.2f}, {s.contest_type})",
            f"Avg lineup correlation:       {np.mean([x['corr'] for x in stats]):+.3f}"
            f"   (variance lift x{np.mean([x['lift'] for x in stats]):.2f} vs independent players)",
            "",
            "Max players from one team: "
            + ", ".join(f"{k}: {v}" for k, v in sorted(stack.items())),
            "",
            "TOP EXPOSURES",
        ]

        for i, c in top:
            p = players[i]
            lines.append(
                f"  {p.name:24s} {c / n:5.0%}   own {p.ownership:5.1%}   "
                f"proj {p.projection:5.1f}   ${p.salary:,.0f}"
            )

        opt = getattr(self, "opt_scores", None)
        if opt is not None and len(opt):
            lines += ["", "OPTIMAL LINEUP (best possible each simulated slate)",
                      f"  median {np.median(opt):.0f}   90th pct {np.percentile(opt, 90):.0f}   "
                      f"99th pct {np.percentile(opt, 99):.0f}   ({len(opt)} slates)"]

        lines += ["", diag, "", f"Run time: {elapsed:.0f}s", "",
                  "NOTE: the field and payout curve are models. ROI numbers are for",
                  "ranking strategies against each other, not a promise of results."]
        return "\n".join(lines)


# ==================================================
# PROJECTION ACCURACY: learned correction + source blending (optional)
# ==================================================

PROJ_DIR = "projection_data"
PROJ_TRAIN_FILE = os.path.join(PROJ_DIR, "projection_training.csv")

TEAM_ALIAS = {
    "GS": "GSW", "NO": "NOP", "NY": "NYK", "PHO": "PHX", "SA": "SAS",
    "BRK": "BKN", "CHO": "CHA", "UTH": "UTA", "WSH": "WAS",
}

PROJ_FEATURES = [
    "salary", "projection", "value", "game_total", "spread", "abs_spread",
    "implied", "is_fav", "proj_pct", "n_games",
    "is_pg", "is_sg", "is_sf", "is_pf", "is_c", "n_pos",
    "dvp_rank", "l5", "l10", "szn",
    "proj_minus_l5", "proj_minus_l10", "proj_minus_szn", "proj_over_l10",
    "team_days_since", "team_b2b", "team_games7", "team_pace",
    "opp_days_since", "opp_b2b", "opp_pace", "opp_def",
    "pl_min_l5", "pl_min_l10", "pl_min_szn", "min_trend",
    "pl_fp_l5", "pl_fp_l10", "pl_fp_szn", "pl_fp_sd10", "pl_fppm_l10",
    "pl_games", "proj_minus_pl_l10",
]

_LOG_FEATURES = [
    "team_days_since", "team_b2b", "team_games7", "team_pace",
    "opp_days_since", "opp_b2b", "opp_pace", "opp_def",
    "pl_min_l5", "pl_min_l10", "pl_min_szn", "min_trend",
    "pl_fp_l5", "pl_fp_l10", "pl_fp_szn", "pl_fp_sd10", "pl_fppm_l10",
    "pl_games",
]


def load_cached_logs(cache_dir="nba_cache"):
    try:
        import glob
        import nba_feed as nf
        paths = sorted(glob.glob(os.path.join(cache_dir, "player_logs_*.csv")))
        if not paths:
            return None
        raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        return nf.prepare_logs(raw)
    except Exception:
        return None


def _nba_team(t):
    t = str(t).strip().upper()
    return TEAM_ALIAS.get(t, t)


def logs_context_features(pool, logs, date=None):
    players = pool.all_players()
    out = pd.DataFrame(np.nan, index=range(len(players)), columns=_LOG_FEATURES)
    if logs is None or len(logs) == 0:
        return out
    date = pd.Timestamp(date) if date is not None else pd.Timestamp(dt.date.today())
    L = logs[logs["game_date"] < date]
    if len(L) == 0:
        return out

    tg = (
        L.groupby(["team_abbreviation", "game_id"])
        .agg(gdate=("game_date", "first"), pts=("pts", "sum"), fp=("fp", "sum"))
        .reset_index()
    )
    tg["game_pts"] = tg.groupby("game_id")["pts"].transform("sum")
    tg["game_fp"] = tg.groupby("game_id")["fp"].transform("sum")
    tg["fp_allowed"] = tg["game_fp"] - tg["fp"]
    league_fp = tg["fp"].mean()

    team = {}
    for t, g in tg.sort_values("gdate").groupby("team_abbreviation"):
        last = g["gdate"].iloc[-1]
        since = (date - last).days
        team[t] = {
            "days_since": since,
            "b2b": 1.0 if since == 1 else 0.0,
            "games7": float((g["gdate"] >= date - pd.Timedelta(days=7)).sum()),
            "pace": float(g["game_pts"].tail(10).mean()),
            "def": float(g["fp_allowed"].tail(15).mean() - league_fp),
        }

    keys = {name_key(p.name) for p in players}
    groups = {k: g.sort_values("game_date")
              for k, g in L[L["key"].isin(keys)].groupby("key")}

    for i, p in enumerate(players):
        tm = team.get(_nba_team(p.team))
        op = team.get(_nba_team(p.opponent))
        if tm:
            out.at[i, "team_days_since"] = tm["days_since"]
            out.at[i, "team_b2b"] = tm["b2b"]
            out.at[i, "team_games7"] = tm["games7"]
            out.at[i, "team_pace"] = tm["pace"]
        if op:
            out.at[i, "opp_days_since"] = op["days_since"]
            out.at[i, "opp_b2b"] = op["b2b"]
            out.at[i, "opp_pace"] = op["pace"]
            out.at[i, "opp_def"] = op["def"]

        g = groups.get(name_key(p.name))
        if g is None or len(g) == 0:
            continue
        g = g.tail(40)
        m5 = g["min"].tail(5).mean()
        m10 = g["min"].tail(10).mean()
        msz = g["min"].mean()
        out.at[i, "pl_min_l5"] = m5
        out.at[i, "pl_min_l10"] = m10
        out.at[i, "pl_min_szn"] = msz
        out.at[i, "min_trend"] = m5 - msz
        out.at[i, "pl_fp_l5"] = g["fp"].tail(5).mean()
        out.at[i, "pl_fp_l10"] = g["fp"].tail(10).mean()
        out.at[i, "pl_fp_szn"] = g["fp"].mean()
        out.at[i, "pl_fp_sd10"] = g["fp"].tail(10).std()
        out.at[i, "pl_fppm_l10"] = g["fp"].tail(10).sum() / max(g["min"].tail(10).sum(), 1.0)
        out.at[i, "pl_games"] = len(g)

    return out


def projection_features(pool, logs=None, n_slots=9):
    players = pool.all_players()
    n = len(players)
    sal = np.array([p.salary for p in players], dtype=float)
    proj = np.array([_raw_proj(p) for p in players], dtype=float)
    tot = pool.game_total[pool.game_idx]
    spread = np.array([p.spread if p.spread is not None else 0.0 for p in players])

    def ex(key):
        return np.array([
            (getattr(p, "extra", None) or {}).get(key, np.nan) for p in players
        ], dtype=float)

    l5, l10, szn = ex("l5_fppg_avg"), ex("l10_fppg_avg"), ex("szn_fppg_avg")
    implied = ex("implied_team_score")
    implied = np.where(np.isnan(implied), tot / 2 - spread / 2, implied)

    f = pd.DataFrame(index=range(n))
    f["salary"] = sal
    f["projection"] = proj
    f["value"] = proj / np.maximum(sal / 1000, 1.0)
    f["game_total"] = tot
    f["spread"] = spread
    f["abs_spread"] = np.abs(spread)
    f["implied"] = implied
    f["is_fav"] = (spread < 0).astype(int)
    f["proj_pct"] = (pd.Series(proj).rank().values - 1) / max(n - 1, 1)
    f["n_games"] = len(pool.game_teams)

    for pos in ("PG", "SG", "SF", "PF", "C"):
        f[f"is_{pos.lower()}"] = [1 if pos in p.positions else 0 for p in players]

    f["n_pos"] = [len(p.positions) for p in players]
    f["dvp_rank"] = ex("l5_dvp_rank")
    f["l5"], f["l10"], f["szn"] = l5, l10, szn
    f["proj_minus_l5"] = proj - l5
    f["proj_minus_l10"] = proj - l10
    f["proj_minus_szn"] = proj - szn
    f["proj_over_l10"] = proj / np.where(l10 > 1, l10, np.nan)

    ctx = logs_context_features(pool, logs, getattr(pool, "slate_date", None))
    for c in _LOG_FEATURES:
        f[c] = ctx[c].values
    f["proj_minus_pl_l10"] = proj - f["pl_fp_l10"].values

    return f[PROJ_FEATURES]


def blend_projection_source(pool, path, weight, log=print):
    weight = float(np.clip(weight, 0.0, 1.0))
    if weight <= 0 or not path or not os.path.exists(path):
        return 0

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    cols = list(df.columns)
    loader = DataLoader()
    c_name = loader._find(cols, "name")
    c_proj = loader._find(cols, "projection")
    if c_name is None or c_proj is None:
        raise ValueError(f"Projection source 2 needs name + projection columns (found {cols})")

    vals = pd.to_numeric(df[c_proj], errors="coerce")
    other = {
        name_key(n): float(v)
        for n, v in zip(df[c_name], vals) if pd.notna(v) and v > 0
    }

    hit = 0
    for p in pool:
        v = other.get(name_key(p.name))
        if v is not None:
            p.projection = (1 - weight) * p.projection + weight * v
            hit += 1

    log(f"Projection source 2: blended {hit}/{len(pool)} players at weight {weight:.0%}")
    return hit


class ProjectionModel:

    _cache = {}

    def __init__(self, path=PROJ_TRAIN_FILE, min_slates=5, full_trust_slates=30,
                 max_pct=0.15):
        self.path = path
        self.min_slates = int(min_slates)
        self.full_trust = max(1, int(full_trust_slates))
        self.max_pct = float(max_pct)

    def load(self):
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=["slate_id", "name"] + PROJ_FEATURES + ["actual"])
        return pd.read_csv(self.path)

    def n_slates(self):
        d = self.load()
        return int(d["slate_id"].nunique()) if len(d) else 0

    def save_slate(self, pool, by_id, by_name, slate_id, logs=None, n_slots=9):
        players = pool.all_players()
        actual = []
        for p in players:
            v = by_id.get(str(p.player_id)) if p.player_id else None
            if v is None:
                v = by_name.get(name_key(p.name))
            actual.append(np.nan if v is None else v)
        actual = np.array(actual, dtype=float)
        keep = np.isfinite(actual) & (actual > 0)
        if keep.sum() < 20:
            return {"saved": 0, "reason": f"only {int(keep.sum())} players with known scores"}

        f = projection_features(pool, logs, n_slots)
        f.insert(0, "name", [p.name for p in players])
        f.insert(0, "slate_id", slate_id)
        f["actual"] = actual
        f = f[keep]

        old = self.load()
        if len(old):
            old = old[old["slate_id"] != slate_id]
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        pd.concat([old, f], ignore_index=True).to_csv(self.path, index=False)

        have_logs = int(f["pl_games"].notna().sum())
        return {"saved": len(f), "with_logs": have_logs}

    @staticmethod
    def _regressors():
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
        except ImportError:
            raise RuntimeError(
                "scikit-learn is required for the projection model: "
                "pip install scikit-learn"
            )
        return [
            HistGradientBoostingRegressor(
                max_depth=d, learning_rate=0.04, max_iter=150,
                min_samples_leaf=leaf, l2_regularization=5.0,
                early_stopping=False, random_state=seed,
            )
            for d, leaf, seed in ((2, 40, 1), (3, 40, 2), (3, 60, 3))
        ]

    def _fit(self, df):
        X = df[PROJ_FEATURES].astype(float)
        y = (df["actual"] - df["projection"]).astype(float).values
        models = self._regressors()
        for m in models:
            m.fit(X, y)
        return models

    @staticmethod
    def _predict(models, X):
        return np.mean([m.predict(X.astype(float)) for m in models], axis=0)

    def cross_validate(self, df):
        slates = list(df["slate_id"].unique())
        oof = np.zeros(len(df))
        for sid in slates:
            te = (df["slate_id"] == sid).values
            models = self._fit(df[~te])
            oof[te] = self._predict(models, df.loc[te, PROJ_FEATURES])

        proj = df["projection"].values.astype(float)
        act = df["actual"].values.astype(float)
        cap = self.max_pct * proj

        def mae(alpha):
            adj = np.clip(alpha * oof, -cap, cap)
            return float(np.abs(act - (proj + adj)).mean())

        grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        scores = {a: mae(a) for a in grid}
        best = min(scores, key=scores.get)
        base = scores[0.0]
        return {
            "alpha": best, "mae_raw": base, "mae_corr": scores[best],
            "improvement": (base - scores[best]) / max(base, 1e-9),
            "scores": scores, "oof": oof, "n_slates": len(slates),
        }

    def _get(self, df):
        st = os.stat(self.path)
        key = (os.path.abspath(self.path), st.st_mtime, st.st_size)
        if key not in ProjectionModel._cache:
            ProjectionModel._cache.clear()
            ProjectionModel._cache[key] = {
                "cv": self.cross_validate(df) if df["slate_id"].nunique() >= 2 else None,
                "models": self._fit(df),
            }
        return ProjectionModel._cache[key]

    def apply(self, pool, logs=None, log=print, n_slots=9):
        df = self.load()
        n_sl = int(df["slate_id"].nunique()) if len(df) else 0
        if n_sl < self.min_slates:
            log(f"Projection correction: only {n_sl} training slate(s) "
                f"(need {self.min_slates}); projections unchanged.")
            return 0

        got = self._get(df)
        cv = got["cv"]
        if cv is None or cv["alpha"] <= 0 or cv["improvement"] < 0.005:
            log("Projection correction: cross-validation shows no reliable gain "
                f"({n_sl} slates); projections unchanged.")
            return 0

        players = pool.all_players()
        X = projection_features(pool, logs, n_slots)
        pred = self._predict(got["models"], X)
        trust = min(1.0, n_sl / self.full_trust)
        raw = np.array([_raw_proj(p) for p in players], dtype=float)
        adj = np.clip(cv["alpha"] * trust * pred, -self.max_pct * raw, self.max_pct * raw)

        for p, r, a in zip(players, raw, adj):
            p.raw_projection = float(r)
            p.projection = float(max(0.5, r + a))

        order = np.argsort(adj)

        def fmt(i):
            return f"{players[i].name} {raw[i]:.1f}->{players[i].projection:.1f}"

        log(
            f"Projection correction applied ({n_sl} slates, shrink {cv['alpha']:.2f} x "
            f"trust {trust:.0%}; CV MAE {cv['mae_raw']:.2f} -> {cv['mae_corr']:.2f}). "
            f"Mean |adj| {np.abs(adj).mean():.2f} pts."
        )
        log("  Biggest cuts:   " + ", ".join(fmt(i) for i in order[:4]))
        log("  Biggest boosts: " + ", ".join(fmt(i) for i in order[::-1][:4]))
        return len(players)

    def report(self):
        df = self.load()
        n_sl = int(df["slate_id"].nunique()) if len(df) else 0
        if n_sl == 0:
            return ("Projection model: no data yet. Backtests, batch backtests and "
                    "'Add slate + results' save training rows automatically.")

        err = df["actual"] - df["projection"]
        lines = [
            "=== PROJECTION ACCURACY ===",
            f"Slates: {n_sl}   Player rows: {len(df):,}",
            f"Raw projections: mean error {err.mean():+.2f} pts, MAE {err.abs().mean():.2f}",
            "",
            "Bias by salary tier (actual - projection):",
        ]
        tiers = pd.cut(
            df["salary"], [0, 4500, 5500, 6500, 8000, 9500, 99999],
            labels=["<=4.5k", "4.5-5.5k", "5.5-6.5k", "6.5-8k", "8-9.5k", "9.5k+"],
        )
        for t, g in err.groupby(tiers, observed=True):
            lines.append(f"  {t:9s} n={len(g):4d}  mean {g.mean():+6.2f}  MAE {g.abs().mean():5.2f}")

        lines.append("")
        lines.append("Bias by position (players eligible at each):")
        for pos in ("PG", "SG", "SF", "PF", "C"):
            m = df[f"is_{pos.lower()}"] == 1
            if m.any():
                lines.append(f"  {pos:3s} n={int(m.sum()):4d}  mean {err[m].mean():+6.2f}")

        if "team_b2b" in df and df["team_b2b"].notna().any():
            b = df["team_b2b"] == 1
            if b.any():
                lines.append("")
                lines.append(
                    f"Back-to-back teams: n={int(b.sum())}, mean {err[b].mean():+.2f} "
                    f"(rested: {err[~b & df['team_b2b'].notna()].mean():+.2f})"
                )
        lines.append("")

        if n_sl < 2:
            lines.append("Need at least 2 slates to cross-validate the correction model.")
            return "\n".join(lines)

        cv = self.cross_validate(df)
        lines.append(f"Leave-one-slate-out correction model ({n_sl} slates):")
        for a, s in cv["scores"].items():
            lines.append(f"  shrink {a:4.2f}: MAE {s:.3f}"
                         + ("   <- best" if a == cv["alpha"] else ""))
        lines.append(
            f"Improvement vs raw: {cv['improvement']:+.1%}  "
            + ("(will be applied)" if cv["alpha"] > 0 and cv["improvement"] >= 0.005
               and n_sl >= self.min_slates
               else f"(NOT applied: needs >= {self.min_slates} slates and >= 0.5% gain)")
        )
        return "\n".join(lines)


# ==================================================
# SLATE ARCHIVE + BATCH BACKTEST
# ==================================================

ARCHIVE_ROOT = "slate_archive"
BACKTEST_HISTORY = os.path.join(ARCHIVE_ROOT, "backtest_history.csv")

_PID_RE = re.compile(r"\((\d+-\d+)\)\s*$")


def read_results_csv(path):
    try:
        return pd.read_csv(
            path, encoding="utf-8-sig", engine="python", on_bad_lines="skip"
        )
    except Exception:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def results_player_scores(results_df):
    cols = {str(c).strip(): c for c in results_df.columns}
    if "Player" not in cols or "FPTS" not in cols:
        raise ValueError(
            "Results file needs 'Player' and 'FPTS' columns "
            f"(found: {list(results_df.columns)})"
        )
    by_id, by_name = {}, {}
    vals = pd.to_numeric(results_df[cols["FPTS"]], errors="coerce")
    for name, v in zip(results_df[cols["Player"]], vals):
        if pd.isna(name) or pd.isna(v):
            continue
        m = _PID_RE.search(str(name))
        if m:
            by_id[m.group(1)] = float(v)
        by_name[name_key(name)] = float(v)
    return by_id, by_name


def contest_rank_stats(results_df, scores):
    if "Points" not in results_df.columns or not len(scores):
        return None
    pts = np.sort(pd.to_numeric(results_df["Points"], errors="coerce").dropna().values)
    n = len(pts)
    if n == 0:
        return None
    rank = 1 + n - np.searchsorted(pts, np.asarray(scores, dtype=float), side="right")
    pct = rank / n
    return {
        "field": n,
        "top_score": float(pts[-1]),
        "best_rank": int(rank.min()),
        "best_pct": float(pct.min()),
        "top1": int((pct <= 0.01).sum()),
        "top5": int((pct <= 0.05).sum()),
        "top20": int((pct <= 0.20).sum()),
    }


class SlateArchive:

    def __init__(self, root=ARCHIVE_ROOT):
        self.root = root

    @staticmethod
    def fd_slate_id(ids):
        c = Counter(str(i).split("-")[0] for i in ids if "-" in str(i))
        return c.most_common(1)[0][0] if c else None

    def archive_run(self, settings, results, log=print):
        import dataclasses
        import shutil

        players = results.pool.all_players()
        fdid = self.fd_slate_id([p.player_id for p in players if p.player_id])

        date = ""
        try:
            if settings.slate_path and os.path.exists(settings.slate_path):
                sd = pd.read_csv(settings.slate_path, nrows=5)
                col = [c for c in sd.columns if c.strip().lower() == "game_date"]
                if col:
                    date = str(sd[col[0]].dropna().iloc[0])
        except Exception:
            pass

        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(self.root, fdid or f"unknown_{stamp}", stamp)
        os.makedirs(folder, exist_ok=True)

        for src, dst in ((settings.slate_path, "slate.csv"),
                         (settings.template_path, "template.csv")):
            if src and os.path.exists(src):
                shutil.copyfile(src, os.path.join(folder, dst))

        with open(os.path.join(folder, "settings.json"), "w") as f:
            json.dump(dataclasses.asdict(settings), f, indent=2, default=str)
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump({
                "fd_slate_id": fdid, "game_date": date, "created": stamp,
                "n_lineups": len(results.lineups), "n_players": len(players),
            }, f, indent=2)

        rows = []
        for k, lu in enumerate(results.lineups):
            for i in lu:
                p = players[i]
                rows.append({
                    "lineup": k + 1, "player_id": p.player_id, "name": p.name,
                    "team": p.team, "salary": p.salary,
                    "projection": p.projection, "ownership": p.ownership,
                })
        pd.DataFrame(rows).to_csv(os.path.join(folder, "lineups.csv"), index=False)

        pdf = results.pool.to_dataframe()
        pdf["player_id"] = [p.player_id for p in players]
        pdf.to_csv(os.path.join(folder, "pool.csv"), index=False)
        log(f"Archived run to {folder}")
        return folder

    def latest_run(self, fdid):
        base = os.path.join(self.root, str(fdid))
        if not fdid or not os.path.isdir(base):
            return None
        runs = sorted(
            d for d in os.listdir(base)
            if os.path.exists(os.path.join(base, d, "lineups.csv"))
        )
        return os.path.join(base, runs[-1]) if runs else None

    def attach_results(self, fdid, results_path):
        import shutil
        base = os.path.join(self.root, str(fdid))
        if fdid and os.path.isdir(base):
            shutil.copyfile(results_path, os.path.join(base, "results.csv"))


def batch_backtest(results_paths, settings, log=print):
    import dataclasses

    arch = SlateArchive()
    lines, rows = [], []
    n_slots = len(settings.slots())

    for path in results_paths:
        name = os.path.basename(path)
        try:
            rdf = read_results_csv(path)
            by_id, by_name = results_player_scores(rdf)
            fdid = SlateArchive.fd_slate_id(by_id.keys())
            run = arch.latest_run(fdid)
            if run is None:
                lines.append(f"- {name}: no archived run for slate {fdid}; skipped")
                continue

            arch.attach_results(fdid, path)

            lu = pd.read_csv(os.path.join(run, "lineups.csv"))
            scores, unknown_lineups = [], 0
            for _, g in lu.groupby("lineup", sort=True):
                tot, miss = 0.0, 0
                for pid, nm in zip(g["player_id"].astype(str), g["name"]):
                    v = by_id.get(pid, by_name.get(name_key(nm)))
                    if v is None:
                        miss += 1
                        v = 0.0
                    tot += v
                scores.append(tot)
                unknown_lineups += miss > 0

            st = contest_rank_stats(rdf, scores) or {}
            own_note = "ownership: no slate copy"
            slate_copy = os.path.join(run, "slate.csv")
            if os.path.exists(slate_copy):
                saved = {}
                try:
                    with open(os.path.join(run, "settings.json")) as f:
                        saved = json.load(f)
                except Exception:
                    pass
                s2 = dataclasses.replace(
                    settings,
                    min_projection=saved.get("min_projection", settings.min_projection),
                )
                pool = DataLoader().load_slate(slate_copy, s2, lambda *a: None)
                info = GBOwnershipModel().save_slate(pool, parse_drafted(rdf), name, n_slots)
                own_note = (
                    f"ownership: saved {info['saved']} players"
                    if info.get("saved") else f"ownership: not saved ({info.get('reason')})"
                )
                try:
                    pi = ProjectionModel().save_slate(
                        pool, by_id, by_name, name,
                        load_cached_logs(settings.nba_cache_dir), n_slots,
                    )
                    own_note += (
                        f"; projections: saved {pi['saved']}"
                        if pi.get("saved") else f"; projections: not saved ({pi.get('reason')})"
                    )
                except Exception as e:
                    own_note += f"; projections: not saved ({e})"

            row = {
                "slate_id": name, "fd_slate": fdid, "run": os.path.basename(run),
                "lineups": len(scores), "best": max(scores),
                "avg": float(np.mean(scores)), "unknown_lineups": unknown_lineups,
                **{k: st.get(k) for k in
                   ("field", "top_score", "best_rank", "best_pct", "top1", "top5", "top20")},
            }
            rows.append(row)
            lines.append(
                f"- {name}: best {row['best']:.1f} / avg {row['avg']:.1f} "
                f"(top entry {st.get('top_score', float('nan')):.1f}); "
                f"best finish {st.get('best_rank', '?')} of {st.get('field', '?')} "
                f"(top {st.get('best_pct', float('nan')):.2%}); "
                f"top1%/5%/20%: {st.get('top1')}/{st.get('top5')}/{st.get('top20')}; "
                f"{own_note}"
                + (f"  [{unknown_lineups} lineups had players missing from results]"
                   if unknown_lineups else "")
            )
        except Exception as e:
            lines.append(f"- {name}: FAILED ({e})")

    if rows:
        os.makedirs(ARCHIVE_ROOT, exist_ok=True)
        new = pd.DataFrame(rows)
        if os.path.exists(BACKTEST_HISTORY):
            old = pd.read_csv(BACKTEST_HISTORY)
            old = old[~old["slate_id"].isin(new["slate_id"])]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(BACKTEST_HISTORY, index=False)

    return "\n".join(lines)


def backtest_history_report():
    if not os.path.exists(BACKTEST_HISTORY):
        return "No backtest history yet."
    h = pd.read_csv(BACKTEST_HISTORY)
    if not len(h):
        return "No backtest history yet."
    return "\n".join([
        f"=== BACKTEST HISTORY ({len(h)} slates) ===",
        f"Mean best-lineup finish:   top {h['best_pct'].mean():.2%} "
        f"(median top {h['best_pct'].median():.2%})",
        f"Mean best / avg lineup:    {h['best'].mean():.1f} / {h['avg'].mean():.1f}",
        f"Lineups in top 1% / 5% / 20%:  "
        f"{int(h['top1'].sum())} / {int(h['top5'].sum())} / {int(h['top20'].sum())} "
        f"of {int(h['lineups'].sum())}",
        f"Slates with a top-1% lineup: {(h['top1'] > 0).sum()}/{len(h)}",
    ])


# ==================================================
# GUI
# ==================================================

def make_sortable(tree):
    def sort_by(col, reverse):
        rows = [(tree.set(k, col), k) for k in tree.get_children("")]

        def conv(v):
            return float(str(v).replace("%", "").replace("$", "").replace(",", ""))

        try:
            rows.sort(key=lambda r: conv(r[0]), reverse=reverse)
        except ValueError:
            rows.sort(key=lambda r: str(r[0]).lower(), reverse=reverse)
        for i, (_, k) in enumerate(rows):
            tree.move(k, "", i)
        tree.heading(col, command=lambda: sort_by(col, not reverse))

    for col in tree["columns"]:
        tree.heading(col, command=lambda c=col: sort_by(c, False))


SETTING_FIELDS = [
    ("Lineups", "lineups", int, 1),
    ("Simulations", "sims", int, 1),
    ("Field size", "field_size", int, 1),
    ("Candidate multiplier", "candidate_mult", float, 1),
    ("Max overlap", "max_overlap", int, 1),
    ("Max exposure %", "max_exposure", float, 100),
    ("Max per team", "max_per_team", int, 1),
    ("Min stack (0=off)", "min_stack", int, 1),
    ("Salary cap", "salary_cap", int, 1),
    ("Salary floor", "salary_floor", int, 1),
    ("Leverage", "leverage", float, 1),
    ("Randomness", "randomness", float, 1),
    ("Min projection", "min_projection", float, 1),
    ("Seed (blank=random)", "seed", int, 1),
    ("Construction weight", "construction_weight", float, 1),
    ("Ownership error (log sd)", "ownership_error", float, 1),
    ("Optimal-lineup sims", "optimal_sims", int, 1),
    ("NBA half-life (games)", "nba_halflife", float, 1),
    ("News weight (0-1)", "news_weight", float, 1),
    ("News ownership boost (0-1)", "news_own_boost", float, 1),
    ("PBP share in blend (0-1)", "pbp_blend", float, 1),
    ("Proj source 2 weight", "proj_source2_weight", float, 1),
]


class OptimizerGUI:

    def __init__(self):
        if tk is None:
            raise RuntimeError("tkinter is not available in this Python install")

        self.root = tk.Tk()
        self.root.title("Optimizer V3")
        self.root.geometry("1400x900")

        self.settings = Settings()
        self.pool = None
        self.results = None
        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None

        self.build_ui()
        self.root.after(100, self.poll_queue)

    def build_ui(self):
        self.build_header()
        self.build_tabs()

    def build_header(self):
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=10, pady=10)

        self.load_button = ttk.Button(header, text="Load Slate", command=self.on_load_slate)
        self.load_button.pack(side="left", padx=5)

        self.template_button = ttk.Button(header, text="Load FD Template", command=self.on_load_template)
        self.template_button.pack(side="left", padx=5)

        self.start_button = ttk.Button(header, text="Generate Portfolio", command=self.on_generate)
        self.start_button.pack(side="left", padx=5)

        self.stop_button = ttk.Button(header, text="Stop", command=self.stop_event.set, state="disabled")
        self.stop_button.pack(side="left", padx=5)

        self.export_button = ttk.Button(header, text="Export FanDuel File", command=self.on_export, state="disabled")
        self.export_button.pack(side="left", padx=5)

        self.progress = ttk.Progressbar(header, mode="determinate", length=400)
        self.progress.pack(side="right")

    def build_backtest_tab(self):
        self.backtest_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.backtest_tab, text="Backtest Evaluation")

        self.btn_load_results = ttk.Button(
            self.backtest_tab,
            text="Select FanDuel Contest Results CSV",
            command=self.run_historical_backtest,
        )
        self.btn_load_results.pack(pady=10, padx=10)

        own_bar = ttk.Frame(self.backtest_tab)
        own_bar.pack(fill="x", padx=10)

        ttk.Button(own_bar, text="Add slate + results to ownership data",
                   command=self.add_ownership_slate).pack(side="left", padx=5)
        ttk.Button(own_bar, text="Batch backtest (select results files)",
                   command=self.run_batch_backtest).pack(side="left", padx=5)
        ttk.Button(own_bar, text="Evaluate projection model",
                   command=self.evaluate_projection_model).pack(side="left", padx=5)
        ttk.Button(own_bar, text="Backtest history",
                   command=self.show_backtest_history).pack(side="left", padx=5)
        ttk.Button(own_bar, text="Evaluate ownership model",
                   command=self.evaluate_ownership_model).pack(side="left", padx=5)

        self.backtest_output = tk.Text(self.backtest_tab, wrap="word", height=25)
        self.backtest_output.pack(fill="both", expand=True, padx=10, pady=10)

    def build_tabs(self):
        self.alert_banner = tk.Label(
            self.root, text="", bg="#c62828", fg="white",
            font=("Segoe UI", 11, "bold"), anchor="w", padx=10, pady=6, cursor="hand2",
        )
        self.alert_banner.bind("<Button-1>", lambda e: self.alert_banner.pack_forget())

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True)

        self.build_dashboard_tab()
        self.build_exposure_tab()
        self.build_player_pool_tab()
        self.build_portfolio_tab()
        self.build_backtest_tab()
        self.build_game_results_tab()
        self.build_learning_tab()
        self.build_simulation_tab()
        self.build_news_tab()
        self.build_log_tab()

    def build_dashboard_tab(self):
        self.dashboard_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.dashboard_tab, text="Dashboard")

        stats = ttk.LabelFrame(self.dashboard_tab, text="Build Statistics")
        stats.pack(fill="x", padx=10, pady=10)

        self.lineups_label = ttk.Label(stats, text="Candidates: 0")
        self.lineups_label.grid(row=0, column=0, padx=10, pady=5)
        self.accepted_label = ttk.Label(stats, text="Selected: 0")
        self.accepted_label.grid(row=0, column=1, padx=10)
        self.rejected_label = ttk.Label(stats, text="Rejected: 0")
        self.rejected_label.grid(row=0, column=2, padx=10)
        self.status_label = ttk.Label(stats, text="Idle")
        self.status_label.grid(row=0, column=3, padx=10)

        files = ttk.LabelFrame(self.dashboard_tab, text="Files")
        files.pack(fill="x", padx=10, pady=5)

        self.slate_var = tk.StringVar()
        self.template_var = tk.StringVar()
        self.proj2_var = tk.StringVar()
        self.output_var = tk.StringVar(value=self.settings.output_path)

        for r, (label, var, cmd) in enumerate([
            ("Slate CSV", self.slate_var, self.on_load_slate),
            ("FD template CSV", self.template_var, self.on_load_template),
            ("Proj source 2 CSV", self.proj2_var, self.on_pick_proj2),
            ("Output CSV", self.output_var, self.on_pick_output),
        ]):
            ttk.Label(files, text=label, width=16).grid(row=r, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(files, textvariable=var, width=90).grid(row=r, column=1, padx=5)
            ttk.Button(files, text="Browse", command=cmd).grid(row=r, column=2, padx=5)

        sf = ttk.LabelFrame(self.dashboard_tab, text="Settings")
        sf.pack(fill="x", padx=10, pady=5)

        self.setting_vars = {}
        for k, (label, attr, typ, scale) in enumerate(SETTING_FIELDS):
            r, c = divmod(k, 4)
            ttk.Label(sf, text=label).grid(row=r, column=c * 2, sticky="w", padx=8, pady=3)
            val = getattr(self.settings, attr)
            var = tk.StringVar(
                value="" if val is None else
                (f"{val * scale:g}" if typ is float else str(val))
            )
            self.setting_vars[attr] = var
            ttk.Entry(sf, textvariable=var, width=10).grid(row=r, column=c * 2 + 1, padx=4)

        r = len(SETTING_FIELDS) // 4 + 1

        ttk.Label(sf, text="Roster format").grid(row=r, column=0, sticky="w", padx=8, pady=3)

        self.format_var = tk.StringVar(value=self.settings.roster_format)
        self.feed_var = tk.BooleanVar(value=self.settings.use_nba_feed)
        ttk.Checkbutton(
            sf, text="Measure sim parameters from NBA stats (nba_api)",
            variable=self.feed_var,
        ).grid(row=r + 1, column=0, columnspan=6, sticky="w", padx=8, pady=3)

        self.gb_var = tk.BooleanVar(value=self.settings.use_gb_ownership)
        self.gb_override_var = tk.BooleanVar(value=self.settings.gb_override_file)
        self.proj_corr_var = tk.BooleanVar(value=self.settings.use_proj_correction)

        ttk.Checkbutton(
            sf, text="Learned projection correction (off = projections as given)",
            variable=self.proj_corr_var,
        ).grid(row=r + 4, column=0, columnspan=4, sticky="w", padx=8, pady=3)

        ttk.Checkbutton(
            sf, text="Gradient-boosted ownership model (off = original estimator)",
            variable=self.gb_var,
        ).grid(row=r + 3, column=0, columnspan=4, sticky="w", padx=8, pady=3)

        ttk.Checkbutton(
            sf, text="Also override ownership from slate file",
            variable=self.gb_override_var,
        ).grid(row=r + 3, column=4, columnspan=3, sticky="w", padx=8, pady=3)

        ttk.Label(sf, text="Contest type").grid(row=r + 2, column=0, sticky="w", padx=8, pady=3)
        self.contest_var = tk.StringVar(value=self.settings.contest_type)
        ttk.Combobox(
            sf, textvariable=self.contest_var,
            values=["Single Entry", "3-Max", "150-Max"],
            state="readonly", width=16,
        ).grid(row=r + 2, column=1, sticky="w", padx=4)

        ttk.Button(sf, text="Tune recency half-life",
                   command=self.on_tune_halflife).grid(row=r + 2, column=3, columnspan=2, sticky="w", padx=8)
        ttk.Button(sf, text="Refresh NBA data",
                   command=self.on_refresh_nba).grid(row=r + 2, column=5, columnspan=2, sticky="w", padx=8)

        self.nba_status_label = ttk.Label(sf, text="")
        self.nba_status_label.grid(row=r + 5, column=0, columnspan=8, sticky="w", padx=8, pady=3)
        try:
            import nba_feed as _nf
            self.nba_status_label.config(text=_nf.data_status(self.settings.nba_cache_dir)["text"])
        except Exception:
            self.nba_status_label.config(text="NBA data: nba_feed.py not found")

        ttk.Combobox(
            sf, textvariable=self.format_var, values=list(ROSTER_FORMATS),
            state="readonly", width=44,
        ).grid(row=r, column=1, columnspan=5, sticky="w", padx=4)

        self.news_var = tk.BooleanVar(value=self.settings.use_news)
        self.news_lines_var = tk.BooleanVar(value=self.settings.news_refresh_lines)

        ttk.Checkbutton(
            sf, text="Live injury news + Vegas lines (free ESPN feed)",
            variable=self.news_var,
        ).grid(row=r + 6, column=0, columnspan=4, sticky="w", padx=8, pady=3)

        ttk.Checkbutton(
            sf, text="Use live lines instead of slate-file spread/total",
            variable=self.news_lines_var,
        ).grid(row=r + 6, column=4, columnspan=4, sticky="w", padx=8, pady=3)

        ttk.Label(sf, text="Simulation engine").grid(row=r + 7, column=0, sticky="w", padx=8, pady=3)
        _engine_names = {"game": "Original game sim", "pbp": "Play-by-play", "blend": "Blend (both)"}
        self.engine_var = tk.StringVar(value=_engine_names.get(self.settings.sim_engine, "Original game sim"))
        ttk.Combobox(
            sf, textvariable=self.engine_var, values=list(_engine_names.values()),
            state="readonly", width=20,
        ).grid(row=r + 7, column=1, columnspan=2, sticky="w", padx=4)

    def build_exposure_tab(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Exposure")

        bar = ttk.Frame(tab)
        bar.pack(fill="x", padx=5, pady=5)
        ttk.Label(
            bar,
            text="Double-click Min % / Max % to edit.  "
                 "Min 100 = lock, Max 0 = ban, blank Max = global default.",
        ).pack(side="left")
        ttk.Button(bar, text="Reset limits", command=self.reset_limits).pack(side="right")

        columns = ("player", "team", "projection", "ownership", "min", "max", "exposure")
        self.exposure_tree = ttk.Treeview(tab, columns=columns, show="headings")
        for c in columns:
            self.exposure_tree.heading(
                c, text={"min": "Min %", "max": "Max %",
                         "exposure": "Portfolio %"}.get(c, c.title())
            )
            self.exposure_tree.column(c, width=120, anchor="center")
        self.exposure_tree.column("player", width=220, anchor="w")
        make_sortable(self.exposure_tree)
        self.exposure_tree.bind("<Double-1>", self.edit_limit_cell)
        self.exposure_tree.pack(fill="both", expand=True)

    def build_player_pool_tab(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Player Pool")

        cols = ("name", "pos", "team", "salary", "projection",
                "own", "opt", "ceiling", "floor", "gpp", "value", "leverage")
        self.player_tree = ttk.Treeview(tab, columns=cols, show="headings")
        for c in cols:
            self.player_tree.heading(c, text=c.upper())
            self.player_tree.column(c, width=90, anchor="center")
        self.player_tree.column("name", width=220, anchor="w")
        make_sortable(self.player_tree)
        self.player_tree.pack(fill="both", expand=True)

    def build_portfolio_tab(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Portfolio")

        names = []
        seen = Counter()
        for slot, _ in self.settings.slots():
            seen[slot] += 1
            names.append(slot if seen[slot] == 1 else f"{slot}{seen[slot]}")

        self.portfolio_cols = ["#"] + names + ["Salary", "Proj", "Own%", "Top1%", "ROI", "Build", "Corr"]
        self.portfolio_tree = ttk.Treeview(tab, columns=self.portfolio_cols, show="headings")
        for c in self.portfolio_cols:
            self.portfolio_tree.heading(c, text=c)
            self.portfolio_tree.column(
                c,
                width=45 if c == "#" else (70 if c in ("Salary", "Proj", "Own%", "Top1%", "ROI") else 130),
                anchor="center",
            )
        make_sortable(self.portfolio_tree)
        self.portfolio_tree.pack(fill="both", expand=True)

    def add_ownership_slate(self):
        slate_path = filedialog.askopenfilename(
            title="1) Select the slate CSV (projections/salaries, pre-lock)",
            filetypes=[("CSV Files", "*.csv")],
        )
        if not slate_path:
            return

        res_path = filedialog.askopenfilename(
            title="2) Select the FanDuel contest results CSV (has %Drafted)",
            filetypes=[("CSV Files", "*.csv")],
        )
        if not res_path:
            return

        try:
            self.read_settings()
            n_slots = len(self.settings.slots())
            pool = DataLoader().load_slate(slate_path, self.settings, lambda *a: None)
            res = pd.read_csv(res_path, encoding="utf-8-sig",
                              engine="python", on_bad_lines="skip")
            info = GBOwnershipModel().save_slate(
                pool, parse_drafted(res), os.path.basename(res_path), n_slots,
            )
            by_id, by_name = results_player_scores(res)
            pinfo = ProjectionModel().save_slate(
                pool, by_id, by_name, os.path.basename(res_path),
                load_cached_logs(self.settings.nba_cache_dir), n_slots,
            )
            model = GBOwnershipModel()
            self.backtest_output.delete("1.0", "end")
            self.backtest_output.insert(
                "end",
                f"Ownership data: {info}\n"
                f"Projection data: {pinfo}\n"
                f"Total saved slates: {model.n_slates()}\n",
            )
        except Exception as e:
            self.backtest_output.insert("end", f"Failed to add slate: {e}\n")

    def run_batch_backtest(self):
        paths = filedialog.askopenfilenames(
            title="Select one or more FanDuel contest results CSVs",
            filetypes=[("CSV Files", "*.csv")],
        )
        if not paths:
            return

        self.backtest_output.delete("1.0", "end")
        self.backtest_output.insert("end", f"Backtesting {len(paths)} file(s)...\n")
        self.root.update_idletasks()

        try:
            self.read_settings()
            text = batch_backtest(list(paths), self.settings)
        except Exception as e:
            text = f"Batch backtest failed: {e}"

        self.backtest_output.delete("1.0", "end")
        self.backtest_output.insert("end", text + "\n\n" + backtest_history_report() + "\n")

    def show_backtest_history(self):
        self.backtest_output.delete("1.0", "end")
        self.backtest_output.insert("end", backtest_history_report() + "\n")

    def evaluate_projection_model(self):
        self.backtest_output.delete("1.0", "end")
        self.backtest_output.insert("end", "Cross-validating...\n")
        self.root.update_idletasks()
        try:
            text = ProjectionModel().report()
        except Exception as e:
            text = f"Evaluation failed: {e}"
        self.backtest_output.delete("1.0", "end")
        self.backtest_output.insert("end", text + "\n")

    def evaluate_ownership_model(self):
        self.backtest_output.delete("1.0", "end")
        self.backtest_output.insert("end", "Cross-validating...\n")
        self.root.update_idletasks()
        try:
            text = GBOwnershipModel().cross_validate()
        except Exception as e:
            text = f"Evaluation failed: {e}"
        self.backtest_output.delete("1.0", "end")
        self.backtest_output.insert("end", text + "\n")

    def run_historical_backtest(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        slate_id = os.path.basename(file_path)

        try:
            results_df = read_results_csv(file_path)
            results_df.columns = [str(c).strip() for c in results_df.columns]
            if 'Player' not in results_df.columns or 'FPTS' not in results_df.columns:
                raise ValueError(
                    "Results file needs 'Player' and 'FPTS' columns "
                    f"(found: {list(results_df.columns)})"
                )

            player_df = results_df[['Player', 'FPTS']].dropna()
            player_lookup = {
                name_key(k): float(v)
                for k, v in zip(player_df['Player'], pd.to_numeric(player_df['FPTS'], errors='coerce'))
                if pd.notna(v)
            }

            if not self.results or len(self.results.lineups) == 0:
                self.backtest_output.insert("end", "Error: You must optimize and generate lineups first before backtesting!\n")
                return

            self.backtest_output.delete("1.0", "end")

            _pool_keys = {name_key(p.name) for p in self.results.pool.all_players()}
            _hits = len(_pool_keys & set(player_lookup))
            if _hits == 0:
                self.backtest_output.insert(
                    "end",
                    "Error: none of the pool players matched names in the results file.\n"
                    f"Results sample: {list(player_lookup)[:5]}\n"
                    f"Pool sample: {list(_pool_keys)[:5]}\n",
                )
                return

            scored_lineups = []
            players = self.results.pool.all_players()
            for index, lineup in enumerate(self.results.lineups):
                total_actual_points = 0
                for player_idx in lineup:
                    total_actual_points += player_lookup.get(
                        name_key(players[player_idx].name), 0.0
                    )
                scored_lineups.append(total_actual_points)

            ranked_lineups = sorted(
                zip(self.results.lineups, scored_lineups),
                key=lambda x: x[1],
                reverse=True,
            )

            learning = ContestLearningEngine(self.settings.contest_type)
            learning.remove_slate(slate_id)

            top_learning_lineups = ranked_lineups[:30]
            players = self.results.pool.all_players()
            for lineup, total_actual_points in top_learning_lineups:
                ownership_sum = sum(players[i].ownership * 100 for i in lineup)
                salary_used = sum(players[i].salary for i in lineup)
                stud_count = sum(1 for i in lineup if players[i].salary >= 9000)
                value_count = sum(1 for i in lineup if players[i].salary <= 5000)
                team_counts = Counter(players[i].team for i in lineup)
                max_team_stack = max(team_counts.values())
                game_totals = [players[i].total for i in lineup if players[i].total]
                avg_game_total = np.mean(game_totals) if game_totals else 0

                ContestLearningEngine(self.settings.contest_type).save_profile(
                    HistoricalContestProfile(
                        slate_id=slate_id,
                        salary_used=salary_used,
                        ownership_sum=ownership_sum,
                        stud_count=stud_count,
                        value_count=value_count,
                        max_team_stack=max_team_stack,
                        avg_game_total=avg_game_total,
                        lineup_score=total_actual_points,
                    )
                )

            top_contest_score = (
                results_df['Points'].max() if 'Points' in results_df.columns else "n/a"
            )

            cal_df, cal_info = calibration_rows(self.results.pool, player_lookup, slate_id)
            cal_log = CalibrationLog()
            if len(cal_df):
                cal_log.save(cal_df, slate_id)

            # FIX: updated message - 'active' excludes DNP, 'dnp' shown separately.
            cal_text = calibration_report(
                cal_df,
                f"THIS SLATE ({cal_info['active']} active matched, "
                f"{cal_info['dnp']} DNP, {cal_info['unmatched']} not in results)",
            )

            cal_all = cal_log.load()
            if cal_all["slate_id"].nunique() > 1:
                cal_text += "\n\n" + calibration_report(cal_all, "ALL SAVED SLATES")

            own_text = ""
            try:
                own_info = GBOwnershipModel().save_slate(
                    self.results.pool, parse_drafted(results_df),
                    slate_id, len(self.settings.slots()),
                )
                if own_info.get("saved"):
                    own_text = (
                        f"Ownership data saved: {own_info['saved']} players "
                        f"({GBOwnershipModel().n_slates()} slate(s) total)\n\n"
                    )
                else:
                    own_text = f"Ownership data not saved: {own_info.get('reason')}\n\n"
            except Exception as e:
                own_text = f"Ownership data not saved: {e}\n\n"

            try:
                _by_id, _by_name = results_player_scores(results_df)
                _pi = ProjectionModel().save_slate(
                    self.results.pool, _by_id, _by_name, slate_id,
                    load_cached_logs(self.settings.nba_cache_dir),
                    len(self.settings.slots()),
                )
                own_text += (
                    f"Projection data saved: {_pi['saved']} players "
                    f"({ProjectionModel().n_slates()} slate(s) total)\n\n"
                    if _pi.get("saved") else
                    f"Projection data not saved: {_pi.get('reason')}\n\n"
                )
            except Exception as e:
                own_text += f"Projection data not saved: {e}\n\n"

            self.refresh_learning_tab()

            self.backtest_output.insert("end", "=== HISTORICAL PERFORMANCE DASHBOARD ===\n\n")
            self.backtest_output.insert("end", f"Top Entry Score in This Contest: {top_contest_score} pts\n")
            self.backtest_output.insert("end", f"Your Best Lineup Realized Score: {round(max(scored_lineups), 2)} pts\n")
            self.backtest_output.insert("end", f"Your Average Lineup Score: {round(sum(scored_lineups)/len(scored_lineups), 2)} pts\n\n")

            try:
                _ids = [m.group(1) for m in
                        (_PID_RE.search(str(x)) for x in player_df['Player']) if m]
                SlateArchive().attach_results(SlateArchive.fd_slate_id(_ids), file_path)
                _st = contest_rank_stats(results_df, scored_lineups)
                if _st:
                    self.backtest_output.insert(
                        "end",
                        f"Best lineup would have finished {_st['best_rank']} of "
                        f"{_st['field']:,} (top {_st['best_pct']:.2%}); lineups in "
                        f"top 1%/5%/20%: {_st['top1']}/{_st['top5']}/{_st['top20']}\n\n",
                    )
            except Exception:
                pass

            self.backtest_output.insert("end", cal_text + "\n\n")
            self.backtest_output.insert("end", own_text)
            self.backtest_output.insert("end", "--- Detailed Lineup Breakdown ---\n")
            for i, score in enumerate(scored_lineups):
                self.backtest_output.insert("end", f"Lineup #{i+1}: {round(score, 2)} FPTS\n")

        except Exception as e:
            self.backtest_output.insert("end", f"Failed to parse backtest: {str(e)}\n")

    def build_game_results_tab(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Game Results")

        cols = ("game", "avg_score", "avg_total", "avg_margin",
                "win_pct", "ot_pct", "sims")
        self.game_tree = ttk.Treeview(tab, columns=cols, show="headings")
        headings = {
            "game": "Game", "avg_score": "Avg Score", "avg_total": "Avg Total",
            "avg_margin": "Avg Margin", "win_pct": "Win %",
            "ot_pct": "OT %", "sims": "Sims",
        }
        for col in cols:
            self.game_tree.heading(col, text=headings[col])
            self.game_tree.column(col, width=120, anchor="center")
        self.game_tree.column("game", width=240, anchor="w")
        make_sortable(self.game_tree)
        self.game_tree.pack(fill="both", expand=True)

    def build_learning_tab(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Contest Learning")

        cols = ("metric", "value")
        self.learning_tree = ttk.Treeview(tab, columns=cols, show="headings")
        self.learning_tree.heading("metric", text="Metric")
        self.learning_tree.heading("value", text="Historical Average")
        self.learning_tree.column("metric", width=300, anchor="w")
        self.learning_tree.column("value", width=200, anchor="center")
        self.learning_tree.pack(fill="both", expand=True)

    def build_simulation_tab(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Simulation")
        self.sim_text = ScrolledText(tab, font=("Courier New", 10))
        self.sim_text.pack(fill="both", expand=True)

    def build_news_tab(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="News")
        self.news_tab = tab
        self.news_pool = None
        self.news_snap = None

        bar = ttk.Frame(tab)
        bar.pack(fill="x", padx=5, pady=5)
        ttk.Button(bar, text="Fetch news now", command=self.on_fetch_news).pack(side="left")
        ttk.Label(bar, text="   Play % override (0-100):").pack(side="left")
        self.news_ovr_var = tk.StringVar(value="")
        ttk.Entry(bar, textvariable=self.news_ovr_var, width=6).pack(side="left", padx=4)
        ttk.Button(bar, text="Set on selected", command=self.on_news_override).pack(side="left", padx=2)
        ttk.Button(bar, text="Clear override", command=self.on_news_clear).pack(side="left", padx=2)
        self.news_info = ttk.Label(bar, text="")
        self.news_info.pack(side="right")

        from news_monitor import MonitorController
        self.monitor_ctl = MonitorController(
            self.settings, self._slate_names, self._lineup_counts, self.post, self.log,
        )
        prefs = self._load_monitor_prefs()

        mon = ttk.LabelFrame(tab, text="Late-news watch (alerts + phone access)")
        mon.pack(fill="x", padx=5, pady=3)

        r1 = ttk.Frame(mon)
        r1.pack(fill="x", padx=4, pady=2)
        ttk.Label(r1, text="Check every (sec):").pack(side="left")
        self.mon_interval_var = tk.StringVar(value=str(prefs.get("interval", 90)))
        ttk.Entry(r1, textvariable=self.mon_interval_var, width=5).pack(side="left", padx=4)
        ttk.Label(r1, text="  Phone push topic (ntfy.sh):").pack(side="left")
        self.mon_topic_var = tk.StringVar(value=prefs.get("topic", ""))
        ttk.Entry(r1, textvariable=self.mon_topic_var, width=24).pack(side="left", padx=4)
        self.mon_phone_var = tk.BooleanVar(value=prefs.get("phone", True))
        ttk.Checkbutton(r1, text="Phone page, port", variable=self.mon_phone_var).pack(side="left", padx=6)
        self.mon_port_var = tk.StringVar(value=str(prefs.get("port", 8765)))
        ttk.Entry(r1, textvariable=self.mon_port_var, width=6).pack(side="left")
        self.mon_button = ttk.Button(r1, text="Start watching", command=self.on_toggle_monitor)
        self.mon_button.pack(side="left", padx=10)
        self.mon_status = ttk.Label(r1, text="monitor off")
        self.mon_status.pack(side="left")

        r2 = ttk.Frame(mon)
        r2.pack(fill="x", padx=4, pady=2)
        ttk.Label(r2, text="Open on your phone:").pack(side="left")
        self.mon_url_var = tk.StringVar(value="")
        ttk.Entry(r2, textvariable=self.mon_url_var, width=58, state="readonly").pack(side="left", padx=4)

        r3 = ttk.Frame(mon)
        r3.pack(fill="x", padx=4, pady=2)
        ttk.Label(r3, text="Quick update (e.g. tatum out; brown 50):").pack(side="left")
        self.manual_var = tk.StringVar(value="")
        e = ttk.Entry(r3, textvariable=self.manual_var, width=44)
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda ev: self.on_manual_update())
        ttk.Button(r3, text="Apply", command=self.on_manual_update).pack(side="left", padx=2)
        ttk.Button(r3, text="Rebuild lineups now", command=self.on_remote_rebuild).pack(side="left", padx=10)

        self.manual_msg = ttk.Label(mon, text="")
        self.manual_msg.pack(anchor="w", padx=6)

        acols = ("time", "player", "change", "lineups", "note")
        self.alert_tree = ttk.Treeview(mon, columns=acols, show="headings", height=4)
        for c, w, t in [("time", 70, "Time"), ("player", 160, "Player"), ("change", 220, "Change"),
                        ("lineups", 140, "Your lineups"), ("note", 520, "Note")]:
            self.alert_tree.heading(c, text=t)
            self.alert_tree.column(c, width=w, anchor="w")
        self.alert_tree.pack(fill="x", padx=4, pady=3)

        ttk.Label(
            tab,
            text="Overrides beat the feed. 0 = out (removed from the pool), 100 = plays. "
                 "In-between values are simulated as a chance of sitting, with teammates absorbing his usage "
                 "in exactly those sims.",
        ).pack(anchor="w", padx=8)

        cols = ("player", "team", "status", "play", "override", "slate", "note")
        self.news_tree = ttk.Treeview(tab, columns=cols, show="headings", height=22)
        for c, w, t in [("player", 170, "Player"), ("team", 150, "Team"), ("status", 100, "Status"),
                        ("play", 70, "Play %"), ("override", 80, "Override"),
                        ("slate", 55, "On slate"), ("note", 520, "Latest note")]:
            self.news_tree.heading(c, text=t)
            self.news_tree.column(c, width=w, anchor="w")
        self.news_tree.pack(fill="both", expand=True, padx=5, pady=5)
        make_sortable(self.news_tree)

    def refresh_news_tree(self, snap):
        import nba_news as nn
        self.news_snap = snap
        slate = set()
        if self.news_pool is not None:
            slate = {nn.name_key(p.name) for p in self.news_pool}
        ovr = {nn.name_key(k): v for k, v in self.settings.news_overrides.items()}
        self.news_tree.delete(*self.news_tree.get_children())
        rows = sorted(snap.injuries.items(), key=lambda kv: (kv[0] not in slate, kv[1]["prob"]))
        for k, i in rows:
            o = ovr.get(k)
            self.news_tree.insert("", "end", iid=k, values=(
                i["name"], i["team"], i["status"], f"{i['prob'] * 100:.0f}",
                "" if o is None else f"{o * 100:.0f}",
                "Y" if k in slate else "", i["comment"][:200]))
        age = int((time.time() - snap.fetched_at) / 60) if snap.fetched_at else 0
        self.news_info.config(
            text=f"{snap.source} | {len(snap.injuries)} players | {len(snap.games)} games | {age} min old")

    def on_fetch_news(self):
        def work():
            try:
                import nba_news as nn
                snap = nn.NewsFeed(log=self.log).get(force=True)
                self.post("news_rows", snap)
            except Exception as e:
                self.log(f"News fetch failed ({e})")
        threading.Thread(target=work, daemon=True).start()

    def on_news_override(self):
        try:
            v = float(self.news_ovr_var.get()) / 100.0
        except ValueError:
            messagebox.showerror("News", "Enter a number from 0 to 100")
            return
        v = min(max(v, 0.0), 1.0)
        for iid in self.news_tree.selection():
            name = self.news_tree.item(iid, "values")[0]
            self.settings.news_overrides[name] = v
        if getattr(self, "news_snap", None):
            self.refresh_news_tree(self.news_snap)

    def on_news_clear(self):
        for iid in self.news_tree.selection():
            name = self.news_tree.item(iid, "values")[0]
            self.settings.news_overrides.pop(name, None)
        if getattr(self, "news_snap", None):
            self.refresh_news_tree(self.news_snap)

    def _monitor_prefs_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_prefs.json")

    def _load_monitor_prefs(self):
        try:
            with open(self._monitor_prefs_path(), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_monitor_prefs(self, **kw):
        try:
            with open(self._monitor_prefs_path(), "w", encoding="utf-8") as f:
                json.dump(kw, f)
        except Exception:
            pass

    def _slate_names(self):
        from nba_news import name_key as _nk
        out = {}
        pools = [getattr(self, "pool", None), getattr(self, "news_pool", None),
                 getattr(self.results, "pool", None) if self.results else None]
        for pool in pools:
            if pool is None:
                continue
            try:
                for p in pool.all_players():
                    out[_nk(p.name)] = p.name
                for n in list(getattr(pool, "news_removed", {}) or {}):
                    out[_nk(n)] = n
            except Exception:
                pass
        return out

    def _lineup_counts(self):
        from nba_news import name_key as _nk
        res = self.results
        if not res or not res.lineups:
            return {}, 0
        try:
            pl = res.pool.all_players()
            counts = {}
            for lu in res.lineups:
                for i in lu:
                    k = _nk(pl[i].name)
                    counts[k] = counts.get(k, 0) + 1
            return counts, len(res.lineups)
        except Exception:
            return {}, 0

    def on_toggle_monitor(self):
        ctl = self.monitor_ctl
        if ctl.monitor is not None:
            ctl.stop()
            self.mon_button.config(text="Start watching")
            self.mon_url_var.set("")
            self.mon_status.config(text="monitor off")
            return

        try:
            interval = float(self.mon_interval_var.get())
        except ValueError:
            interval = 90.0
        try:
            port = int(self.mon_port_var.get())
        except ValueError:
            port = 8765

        topic = self.mon_topic_var.get().strip()
        phone = bool(self.mon_phone_var.get())
        url = ctl.start(interval, topic, phone, port)
        self._save_monitor_prefs(interval=interval, topic=topic, phone=phone, port=port)
        self.mon_url_var.set(url)
        self.mon_button.config(text="Stop watching")
        bits = [f"watching every {int(max(interval, 30))}s"]
        if topic:
            bits.append(f"push to ntfy topic '{topic}'")
        self.mon_status.config(text=" | ".join(bits))
        if self.news_snap is not None:
            ctl.set_baseline(self.news_snap, self.settings.news_overrides)

    def on_manual_update(self):
        text = self.manual_var.get().strip()
        if not text:
            return
        if self.news_snap is not None:
            self.monitor_ctl.known.update({k: i["name"] for k, i in self.news_snap.injuries.items()})
        msg = self.monitor_ctl.apply_manual_text(text)
        self.manual_msg.config(text=msg + "   -> press 'Rebuild lineups now' to apply")
        self.manual_var.set("")

    def on_remote_rebuild(self):
        if str(self.start_button["state"]) == "disabled":
            self.log("Rebuild requested but a run is already in progress")
            return
        self.log("Rebuilding lineups with the latest news / overrides")
        self.on_generate()

    def on_alerts(self, fresh):
        for c in fresh:
            n = self._lineup_counts()[0].get(c["key"], 0)
            from news_monitor import label_for
            self.alert_tree.insert("", 0, values=(
                time.strftime("%H:%M:%S", time.localtime(c["t"])), c["name"],
                f"{label_for(c['old'])} -> {label_for(c['new'])}",
                f"{n} lineups" if n else "-", c.get("comment", "")[:160]))
        names = ", ".join(f"{c['name']} ({c['kind'].lower()})" for c in fresh[:4])
        self.alert_banner.config(text=f"LATE NEWS: {names} - open the News tab, then rebuild (click to dismiss)")
        self.alert_banner.pack(fill="x", before=self.tabs)
        try:
            self.tabs.select(self.news_tab)
        except Exception:
            pass

    def build_log_tab(self):
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Log")
        self.log_text = ScrolledText(tab)
        self.log_text.pack(fill="both", expand=True)

    def log(self, msg): self.q.put(("log", str(msg)))
    def update_progress(self, current, total): self.q.put(("progress", (current, total)))
    def set_status(self, status): self.q.put(("status", status))
    def stats(self, **kw): self.q.put(("stats", kw))
    def post(self, kind, payload=None): self.q.put((kind, payload))

    def poll_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log_text.insert(tk.END, payload + "\n")
                    self.log_text.see(tk.END)
                elif kind == "progress":
                    cur, total = payload
                    self.progress["value"] = cur / max(total, 1) * 100
                elif kind == "status":
                    self.status_label.config(text=payload)
                elif kind == "nba_status":
                    self.nba_status_label.config(text=payload)
                elif kind == "stats":
                    self.lineups_label.config(text=f"Candidates: {payload['lineups']}")
                    self.accepted_label.config(text=f"Selected: {payload['accepted']}")
                    self.rejected_label.config(text=f"Rejected: {payload['rejected']}")
                elif kind == "news":
                    snap, rep_, pool_ = payload
                    self.news_pool = pool_
                    self.refresh_news_tree(snap)
                    self.monitor_ctl.set_baseline(snap, self.settings.news_overrides)
                elif kind == "alert":
                    self.on_alerts(payload)
                elif kind == "override_changed":
                    if self.news_snap is not None:
                        self.refresh_news_tree(self.news_snap)
                elif kind == "rebuild":
                    self.on_remote_rebuild()
                elif kind == "news_rows":
                    self.refresh_news_tree(payload)
                elif kind == "pool":
                    self.refresh_player_tree(payload)
                    self.refresh_exposure_tree(payload)
                elif kind == "game_summary":
                    self.refresh_game_results(payload)
                elif kind == "report":
                    self.sim_text.delete("1.0", tk.END)
                    self.sim_text.insert(tk.END, payload)
                elif kind == "done":
                    self.on_done(payload)
                elif kind == "error":
                    self.log_text.insert(tk.END, payload + "\n")
                    self.status_label.config(text="Error - see Log tab")
                    self.set_running(False)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_queue)

    def set_running(self, running):
        self.start_button.config(state="disabled" if running else "normal")
        self.load_button.config(state="disabled" if running else "normal")
        self.stop_button.config(state="normal" if running else "disabled")

    def read_settings(self):
        s = self.settings
        for label, attr, typ, scale in SETTING_FIELDS:
            text = self.setting_vars[attr].get().strip()
            if text == "":
                if attr == "seed":
                    s.seed = None
                    continue
                raise ValueError(f"{label} is empty")
            val = float(text) / scale
            setattr(s, attr, int(round(val)) if typ is int else val)

        s.roster_format = self.format_var.get()
        s.use_nba_feed = bool(self.feed_var.get())
        s.use_news = bool(self.news_var.get())
        s.news_refresh_lines = bool(self.news_lines_var.get())
        s.sim_engine = {"Original game sim": "game", "Play-by-play": "pbp",
                        "Blend (both)": "blend"}.get(self.engine_var.get(), "game")
        s.use_gb_ownership = bool(self.gb_var.get())
        s.use_proj_correction = bool(self.proj_corr_var.get())
        s.proj_source2_path = self.proj2_var.get().strip()
        s.gb_override_file = bool(self.gb_override_var.get())
        s.contest_type = self.contest_var.get()
        s.slate_path = self.slate_var.get()
        s.template_path = self.template_var.get()
        s.output_path = self.output_var.get()

        if s.salary_floor > s.salary_cap:
            raise ValueError("Salary floor is above the cap")

    def on_load_slate(self):
        path = filedialog.askopenfilename(
            title="Slate CSV", filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if not path:
            return
        self.slate_var.set(path)
        try:
            self.read_settings()
            pool = DataLoader().load_slate(path, self.settings, self.log)
            OwnershipEngine().apply(pool, len(self.settings.slots()), log=self.log)
            self.pool = pool
            self.refresh_player_tree(pool)
            self.refresh_exposure_tree(pool)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def on_refresh_nba(self):
        def work():
            try:
                import nba_feed as nf
                self.log("Refreshing NBA game logs...")
                nf.refresh(self.settings.nba_cache_dir, log=self.log)
                self.post("nba_status", nf.data_status(self.settings.nba_cache_dir)["text"])
            except Exception as e:
                self.log(f"NBA refresh failed: {e}")
        threading.Thread(target=work, daemon=True).start()

    def on_tune_halflife(self):
        def work():
            try:
                from nba_feed import NBAFeed, tune_halflife
                self.log("Tuning recency half-life (walk-forward on game logs)...")
                lg = NBAFeed(self.settings.nba_cache_dir, log=self.log).get_logs()
                table, best = tune_halflife(lg)
                self.log("\n" + table.to_string(index=False))
                self.log(f"Lowest fantasy-point error at half-life {best} games - "
                         "set 'NBA half-life' to try it.")
            except Exception as e:
                self.log(f"Half-life tuning failed: {e}")
        threading.Thread(target=work, daemon=True).start()

    def on_load_template(self):
        path = filedialog.askopenfilename(
            title="FanDuel template / player list CSV",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if path:
            self.template_var.set(path)

    def on_pick_proj2(self):
        path = filedialog.askopenfilename(
            title="Second projection source (optional)",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if path:
            self.proj2_var.set(path)

    def on_pick_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
        )
        if path:
            self.output_var.set(path)

    def on_generate(self):
        try:
            self.read_settings()
        except ValueError as e:
            messagebox.showerror("Settings", str(e))
            return

        if not self.settings.slate_path:
            messagebox.showinfo("Slate", "Load a slate CSV first")
            return

        self.stop_event.clear()
        self.set_running(True)
        self.export_button.config(state="disabled")
        pool = self.pool

        def work():
            try:
                Pipeline(self.settings, self, self.stop_event).run(pool)
            except Exception:
                self.post("error", traceback.format_exc())

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def on_done(self, res):
        self.results = res
        try:
            SlateArchive().archive_run(self.settings, res, self.log)
        except Exception as e:
            self.log(f"Slate archive failed: {e}")

        self.pool = res.pool
        self.set_running(False)
        self.export_button.config(state="normal")

        tree = self.portfolio_tree
        tree.delete(*tree.get_children())
        players = res.pool.all_players()
        for k, (lu, st) in enumerate(zip(res.lineups, res.lineup_stats)):
            tree.insert("", "end", values=(
                [k + 1]
                + [players[i].name for i in lu]
                + [f"${st['salary']:,.0f}", f"{st['proj']:.1f}",
                   f"{st['own']:.0f}", f"{st['top1']:.1%}", f"{st['roi']:+.0%}",
                   f"{st['construction']:.0f}", f"{st['corr']:+.2f}"]
            ))

        self.refresh_exposure_tree(res.pool, res.lineups)
        self.tabs.select(3)

    def on_export(self):
        if not self.results:
            return
        try:
            self.settings.output_path = self.output_var.get()
            n = Exporter().export(
                self.results.pool, self.results.lineups, self.settings,
                self.results.id_map, self.settings.output_path,
            )
            self.log(f"Exported {n} lineups to {self.settings.output_path}")
            messagebox.showinfo("Export", f"Saved {n} lineups to\n{self.settings.output_path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def refresh_player_tree(self, pool):
        tree = self.player_tree
        tree.delete(*tree.get_children())
        for p in pool:
            tree.insert("", "end", values=(
                p.name, "/".join(p.positions), p.team, f"${p.salary:,.0f}",
                f"{p.projection:.1f}", f"{p.ownership:.1%}",
                f"{p.optimal_pct:.1%}" if p.optimal_pct else "",
                f"{p.ceiling:.1f}" if p.ceiling else "",
                f"{p.floor:.1f}" if p.floor else "",
                f"{p.gpp_probability:.1%}" if p.gpp_probability else "",
                f"{p.value_score():.2f}",
                f"{p.leverage:.2f}" if p.leverage else "",
            ))

    def refresh_game_results(self, summaries):
        tree = self.game_tree
        tree.delete(*tree.get_children())
        for g in summaries:
            score = f"{g['avg_a_score']:.1f}-{g['avg_b_score']:.1f}"
            win = f"{g['a_win_pct']:.1%}/{g['b_win_pct']:.1%}"
            tree.insert("", "end", values=(
                g["game"], score, f"{g['avg_total']:.1f}", f"{g['avg_margin']:+.1f}",
                win, f"{g['ot_pct']:.1%}", g["simulations"],
            ))

    def refresh_learning_tab(self):
        tree = self.learning_tree
        tree.delete(*tree.get_children())
        summary = ContestLearningEngine(self.settings.contest_type).summarize()
        if not summary:
            return
        rows = [
            ("Historical Contests", summary["contests"]),
            ("Average Salary Used", f"${summary['avg_salary']:,.0f}"),
            ("Average Ownership Sum", f"{summary['avg_ownership']:.1f}%"),
            ("Average Stud Count", f"{summary['avg_studs']:.2f}"),
            ("Average Value Count", f"{summary['avg_values']:.2f}"),
            ("Average Largest Team Stack", f"{summary['avg_stack']:.2f}"),
            ("Average Game Total", f"{summary['avg_game_total']:.1f}"),
        ]
        for metric, value in rows:
            tree.insert("", "end", values=(metric, value))

    def refresh_exposure_tree(self, pool, lineups=None):
        tree = self.exposure_tree
        tree.delete(*tree.get_children())
        counts = Counter(i for lu in (lineups or []) for i in lu)
        n = len(lineups) if lineups else 0
        players = pool.all_players()
        for p in sorted(players, key=lambda p: -p.ownership):
            lim = self.settings.player_limits.get(p.name, [0, None])
            tree.insert("", "end", values=(
                p.name, p.team, f"{p.projection:.1f}", f"{p.ownership:.1%}",
                f"{(lim[0] or 0) * 100:g}",
                "" if lim[1] is None else f"{lim[1] * 100:g}",
                f"{counts.get(p.idx, 0) / n:.0%}" if n else "",
            ))

    def edit_limit_cell(self, event):
        tree = self.exposure_tree
        if tree.identify("region", event.x, event.y) != "cell":
            return
        col = tree.identify_column(event.x)
        name = tree["columns"][int(col[1:]) - 1]
        if name not in ("min", "max"):
            return
        item = tree.identify_row(event.y)
        x, y, w, h = tree.bbox(item, col)
        entry = ttk.Entry(tree)
        entry.place(x=x, y=y, width=w, height=h)
        entry.insert(0, tree.set(item, name))
        entry.focus()
        done = {"v": False}

        def commit(_=None):
            if done["v"]:
                return
            done["v"] = True
            text = entry.get().strip()
            entry.destroy()
            try:
                val = float(text) / 100 if text else None
            except ValueError:
                return
            player = tree.set(item, "player")
            lim = self.settings.player_limits.setdefault(player, [0.0, None])
            if name == "min":
                lim[0] = val or 0.0
            else:
                lim[1] = val
            tree.set(item, name, "" if val is None and name == "max" else f"{(val or 0) * 100:g}")

        def cancel(_=None):
            done["v"] = True
            entry.destroy()

        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)
        entry.bind("<Escape>", cancel)

    def reset_limits(self):
        self.settings.player_limits.clear()
        if self.pool:
            self.refresh_exposure_tree(self.pool)

    def run(self):
        self.root.mainloop()


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

if __name__ == "__main__":
    gui = OptimizerGUI()
    gui.run()
