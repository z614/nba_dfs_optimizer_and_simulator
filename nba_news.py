"""
nba_news.py - live injury news + Vegas lines for the NBA DFS optimizer
======================================================================

Free sources only (no API keys, stdlib urllib):
  * ESPN injuries   site.api.espn.com/.../basketball/nba/injuries
  * ESPN scoreboard site.api.espn.com/.../basketball/nba/scoreboard  (spread / total)

What it does for the optimizer
  1. fetch + cache news (news_cache.json; falls back to the cache if offline)
  2. map every status to a PLAY PROBABILITY (Out 0, Doubtful .08, Questionable .5 ...)
     - you can override any player by name (Settings.news_overrides)
  3. remove late-news OUT players from the pool (feeds usage-shift machinery)
  4. fill / refresh spread + total from live odds
  5. boost ownership of teammates who inherit a late scratch's role
  6. inside the simulation, players who might not play score 0 in the right
     share of sims and their teammates absorb the usage in exactly those sims,
     so lineup selection sees GTD risk instead of ignoring it

Everything fails soft: no internet -> cache -> nothing, never an exception.
"""

import json
import os
import re
import time
import unicodedata
import urllib.request
from dataclasses import dataclass, field

import numpy as np

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
INJURY_URL = ESPN_BASE + "/injuries"
SCOREBOARD_URL = ESPN_BASE + "/scoreboard"

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# status text -> probability the player plays
STATUS_PLAY_PROB = {
    "OUT": 0.0, "O": 0.0, "INJ": 0.0, "SUSPENDED": 0.0, "SUSP": 0.0,
    "OUT FOR SEASON": 0.0, "INJURED RESERVE": 0.0, "IR": 0.0,
    "DOUBTFUL": 0.08, "D": 0.08,
    "QUESTIONABLE": 0.50, "Q": 0.50, "GTD": 0.55, "GAME-TIME DECISION": 0.55,
    "DAY-TO-DAY": 0.60, "DAY TO DAY": 0.60, "DTD": 0.60,
    "PROBABLE": 0.90, "P": 0.90,
    "AVAILABLE": 1.0, "ACTIVE": 1.0, "": 1.0,
}

# comment keywords that move the probability within a status
_UP = ("expected to play", "will play", "cleared", "upgraded", "return", "probable", "full participant")
_DOWN = ("not expected", "unlikely", "ruled out", "will not play", "will miss", "downgraded", "sidelined", "surgery")

TEAM_CANON = {
    "GS": "GSW", "GSW": "GSW", "NY": "NYK", "NYK": "NYK", "NO": "NOP", "NOP": "NOP",
    "SA": "SAS", "SAS": "SAS", "PHO": "PHX", "PHX": "PHX", "BK": "BKN", "BKN": "BKN",
    "UTAH": "UTA", "UTA": "UTA", "WSH": "WAS", "WAS": "WAS", "CHO": "CHA", "CHA": "CHA",
}


def canon_team(t):
    t = str(t or "").strip().upper()
    return TEAM_CANON.get(t, t)


def name_key(name):
    """'Nikola Jokić Jr.' -> 'nikola jokic' (matches slate names to news names)."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[.'’`]", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(t for t in s.split() if t not in SUFFIXES)


def status_to_prob(status, comment=""):
    st = str(status or "").strip().upper()
    p = STATUS_PLAY_PROB.get(st)
    if p is None:                                  # unknown wording
        if "OUT" in st:
            p = 0.0
        elif "DOUBT" in st:
            p = 0.08
        elif "QUEST" in st:
            p = 0.5
        elif "PROB" in st:
            p = 0.9
        elif "DAY" in st:
            p = 0.6
        else:
            p = 1.0
    if 0.0 < p < 1.0:                              # nudge inside the status using the note
        c = str(comment or "").lower()
        if any(k in c for k in _UP):
            p = min(0.95, p + 0.20)
        elif any(k in c for k in _DOWN):
            p = max(0.03, p - 0.25)
    return float(p)


# ==================================================
# FETCH
# ==================================================

def _get_json(url, timeout=8):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (dfs-optimizer)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def parse_injuries(data):
    """ESPN injuries JSON -> {name_key: dict(name, team, status, comment, date)}."""
    out = {}
    for team in (data or {}).get("injuries", []) or []:
        tname = team.get("displayName", "")
        for it in team.get("injuries", []) or []:
            ath = it.get("athlete") or {}
            name = ath.get("displayName") or ath.get("fullName") or ""
            if not name:
                continue
            status = it.get("status") or (it.get("type") or {}).get("description") or ""
            comment = it.get("shortComment") or it.get("longComment") or ""
            out[name_key(name)] = dict(
                name=name, team=tname, status=str(status), comment=str(comment),
                date=str(it.get("date", "")),
                prob=status_to_prob(status, comment),
            )
    return out


def parse_odds(data):
    """ESPN scoreboard JSON -> list of dict(home, away, spread_home, total)."""
    games = []
    for ev in (data or {}).get("events", []) or []:
        try:
            comp = ev["competitions"][0]
            teams = {c.get("homeAway"): canon_team((c.get("team") or {}).get("abbreviation"))
                     for c in comp.get("competitors", [])}
            odds = (comp.get("odds") or [None])[0]
            if not odds or "home" not in teams or "away" not in teams:
                continue
            total = odds.get("overUnder")
            spread_home = None
            det = str(odds.get("details", "")).strip().upper()
            m = re.match(r"^([A-Z]{2,4})\s+([+-]?\d+(?:\.\d+)?)$", det)
            if m:
                fav, val = canon_team(m.group(1)), float(m.group(2))
                spread_home = val if fav == teams["home"] else -val if fav == teams["away"] else None
            elif det in ("EVEN", "PK", "PICK", "PICKEM"):
                spread_home = 0.0
            if spread_home is None and odds.get("spread") is not None:
                spread_home = float(odds["spread"])
            games.append(dict(home=teams["home"], away=teams["away"],
                              spread_home=spread_home,
                              total=None if total is None else float(total),
                              start=str(ev.get("date") or comp.get("date") or "")))
        except Exception:
            continue
    return games


@dataclass
class NewsSnapshot:
    injuries: dict = field(default_factory=dict)
    games: list = field(default_factory=list)
    fetched_at: float = 0.0
    source: str = "none"          # live | cache | none


class NewsFeed:

    def __init__(self, cache_path="news_cache.json", max_age_min=10, log=print):
        self.cache_path = cache_path
        self.max_age = max_age_min * 60
        self.log = log

    def _load_cache(self):
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return NewsSnapshot(d.get("injuries", {}), d.get("games", []),
                                d.get("fetched_at", 0.0), "cache")
        except Exception:
            return None

    def _save_cache(self, snap):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(dict(injuries=snap.injuries, games=snap.games,
                               fetched_at=snap.fetched_at), f)
        except Exception:
            pass

    def get(self, force=False):
        cached = self._load_cache()
        if cached and not force and time.time() - cached.fetched_at < self.max_age:
            self.log(f"News: using cache ({int((time.time() - cached.fetched_at) / 60)} min old, "
                     f"{len(cached.injuries)} injury rows, {len(cached.games)} games)")
            return cached
        snap = NewsSnapshot(fetched_at=time.time(), source="live")
        errs = []
        try:
            snap.injuries = parse_injuries(_get_json(INJURY_URL))
        except Exception as e:
            errs.append(f"injuries: {e}")
        try:
            snap.games = parse_odds(_get_json(SCOREBOARD_URL))
        except Exception as e:
            errs.append(f"odds: {e}")
        if not snap.injuries and not snap.games:
            if cached:
                self.log(f"News: live fetch failed ({'; '.join(errs)}); using stale cache")
                return cached
            self.log(f"News: unavailable ({'; '.join(errs)}); continuing without it")
            return NewsSnapshot(source="none")
        if errs:
            self.log("News: partial fetch (" + "; ".join(errs) + ")")
            if cached:                                   # keep the half that failed
                snap.injuries = snap.injuries or cached.injuries
                snap.games = snap.games or cached.games
        self._save_cache(snap)
        self.log(f"News: live ({len(snap.injuries)} injury rows, {len(snap.games)} games)")
        return snap


# ==================================================
# APPLY TO POOL
# ==================================================

@dataclass
class NewsReport:
    removed: list = field(default_factory=list)    # (name, status) taken out of pool
    risky: list = field(default_factory=list)      # (name, status, prob) kept with play_prob<1
    resurrected: list = field(default_factory=list)  # slate file dropped them but news says healthy
    line_updates: list = field(default_factory=list)
    vacated: dict = field(default_factory=dict)    # team -> projection points vacated by removed outs
    source: str = "none"


def _game_lookup(games):
    lk = {}
    for g in games:
        lk[frozenset((g["home"], g["away"]))] = g
    return lk


def apply_news(pool, snap, overrides=None, out_threshold=0.10,
               update_lines=False, log=print):
    """Mutates pool. Call after the pool is built, BEFORE ownership/sim."""
    rep = NewsReport(source=snap.source)
    overrides = {name_key(k): float(v) for k, v in (overrides or {}).items()}

    for p in pool:
        p.play_prob = 1.0
        p.news_note = ""

    # ---- availability
    to_remove = []
    for p in pool:
        k = name_key(p.name)
        info = snap.injuries.get(k)
        prob = overrides.get(k)
        status = info["status"] if info else ""
        if prob is None and info:
            prob = info["prob"]
        if prob is None:
            continue
        p.play_prob = float(np.clip(prob, 0.0, 1.0))
        p.news_note = (info["comment"][:140] if info else "manual override")
        if info:
            p.injury = status
        if p.play_prob <= out_threshold:
            to_remove.append(p)
        elif p.play_prob < 0.995:
            rep.risky.append((p.name, status or "override", p.play_prob))

    for p in to_remove:
        rep.removed.append((p.name, p.injury or "override"))
        rep.vacated[p.team] = rep.vacated.get(p.team, 0.0) + p.projection
        pool.players.pop(p.name, None)
        outs = getattr(pool, "out_names", None)
        if outs is not None:
            outs.append(p.name)
    pool.news_removed = {p.name: p for p in to_remove}   # kept for ownership boost

    # slate file dropped them but news says they're fine
    for nm in getattr(pool, "out_names", []) or []:
        info = snap.injuries.get(name_key(nm))
        if info is None:
            continue
        if info["prob"] >= 0.5:
            rep.resurrected.append((nm, info["status"], info["prob"]))

    # ---- lines
    if snap.games:
        lk = _game_lookup(snap.games)
        done = set()
        for p in pool:
            g = lk.get(frozenset((canon_team(p.team), canon_team(p.opponent))))
            if not g:
                continue
            team_is_home = canon_team(p.team) == g["home"]
            live_spread = None
            if g["spread_home"] is not None:
                live_spread = g["spread_home"] if team_is_home else -g["spread_home"]
            live_total = g["total"]
            key = (frozenset((p.team, p.opponent)))
            changed = False
            if live_spread is not None and (p.spread is None or
                                            (update_lines and abs(p.spread - live_spread) >= 0.5)):
                old, p.spread, changed = p.spread, live_spread, True
            if live_total is not None and (p.total is None or
                                           (update_lines and abs(p.total - live_total) >= 1.0)):
                old_t, p.total, changed = p.total, live_total, True
            if changed and key not in done:
                done.add(key)
                rep.line_updates.append((p.team, p.opponent, p.spread, p.total))
            elif not update_lines and key not in done and live_total is not None \
                    and p.total is not None and abs(p.total - live_total) >= 3:
                done.add(key)
                log(f"News: slate total for {p.team}/{p.opponent} is {p.total:g} but live is "
                    f"{live_total:g} (enable 'refresh lines' to use live)")

    pool.finalize()
    log_report(rep, log)
    return rep


def log_report(rep, log=print):
    if rep.removed:
        log("News: removed from pool - " + ", ".join(f"{n} ({s})" for n, s in rep.removed))
    if rep.risky:
        log("News: availability risk - " + ", ".join(f"{n} {s} {p:.0%}" for n, s, p in rep.risky))
    if rep.resurrected:
        log("News WARNING: slate file marks these OUT but news says likely to play - "
            + ", ".join(f"{n} ({s})" for n, s, _ in rep.resurrected)
            + ". Re-export your slate file if you want them included.")
    if rep.line_updates:
        log("News: lines updated for " + ", ".join(
            f"{a}/{b} (spr {s if s is None else format(s, '+g')}, tot {t})"
            for a, b, s, t in rep.line_updates[:8]))


# ==================================================
# OWNERSHIP
# ==================================================

def boost_ownership(pool, rep, strength=0.5, log=print):
    """Late scratches move the field. Teammates who inherit the role get more
    ownership (cheap ones the most); the slate's total ownership is preserved."""
    if not rep or not rep.vacated or strength <= 0:
        return
    own0 = np.array([p.ownership for p in pool], dtype=float)
    total0 = own0.sum()
    if total0 <= 0:
        return
    boosted = []
    for team, vac in rep.vacated.items():
        mates = [p for p in pool if p.team == team and p.projection > 0]
        if not mates:
            continue
        w = np.array([p.projection ** 1.5 for p in mates])
        w = w / w.sum()
        for p, wi in zip(mates, w):
            gain = vac * wi / max(p.projection, 1.0)           # fraction of own projection gained
            cheap = 1.25 if p.salary <= 5500 else (1.10 if p.salary <= 6500 else 1.0)
            f = (1.0 + strength * 2.0 * gain) * (cheap if gain > 0.05 else 1.0)
            f = float(np.clip(f, 1.0, 2.5))
            if f > 1.04:
                p.ownership = min(0.85, max(p.ownership, 0.005) * f)
                boosted.append((p.name, f))
    own1 = np.array([p.ownership for p in pool], dtype=float)
    if own1.sum() > 0:
        scale = total0 / own1.sum()
        touched = {n for n, _ in boosted}
        # take the added ownership proportionally from players who were not boosted
        un = np.array([p.name not in touched for p in pool])
        extra = own1.sum() - total0
        if un.any() and own1[un].sum() > extra > 0:
            k = 1 - extra / own1[un].sum()
            for p, u in zip(pool, un):
                if u:
                    p.ownership *= k
        else:
            for p in pool:
                p.ownership *= scale
    if boosted:
        boosted.sort(key=lambda t: -t[1])
        log("News: ownership boosted for " + ", ".join(f"{n} x{f:.2f}" for n, f in boosted[:8]))


# ==================================================
# SIMULATION
# ==================================================

def apply_availability(pool, sims, rng, redistribute=0.55, log=print):
    """In-place on sims (S, N). Players with play_prob<1 score 0 in the right
    share of sims; teammates absorb part of the vacated production in exactly
    those sims (concentrated on the highest-projected teammates)."""
    players = pool.all_players()
    risky = [p for p in players if getattr(p, "play_prob", 1.0) < 0.995]
    if not risky:
        return {}
    S, N = sims.shape
    proj = np.array([p.projection for p in players])
    team = pool.team_idx
    out_mask = np.zeros((S, N), dtype=bool)
    for p in risky:
        out_mask[:, p.idx] = rng.random(S) > p.play_prob          # True = sits
    for p in risky:
        k = p.idx
        sit = out_mask[:, k]
        if not sit.any():
            continue
        mates = np.where((team == team[k]) & (np.arange(N) != k))[0]
        if len(mates) == 0:
            continue
        w = proj[mates] ** 1.5
        w = w / w.sum()
        factor = 1.0 + redistribute * proj[k] * w / np.maximum(proj[mates], 1.0)
        factor = np.minimum(factor, 1.6)
        sims[np.ix_(sit, mates)] *= factor[None, :].astype(sims.dtype)
    sims[out_mask] = 0.0
    log("Availability: " + ", ".join(
        f"{p.name} plays {p.play_prob:.0%}" for p in risky[:10]))
    return {p.name: float(1 - out_mask[:, p.idx].mean()) for p in risky}


if __name__ == "__main__":
    import sys
    snap = NewsFeed(log=print).get(force="--refresh" in sys.argv or True)
    outs = [(v["name"], v["status"]) for v in snap.injuries.values() if v["prob"] <= 0.1]
    risky = [(v["name"], v["status"], v["prob"]) for v in snap.injuries.values() if 0.1 < v["prob"] < 0.95]
    print(f"{len(outs)} out, {len(risky)} questionable/day-to-day, {len(snap.games)} games with lines")
    for g in snap.games:
        print(f"  {g['away']} @ {g['home']}  home spread {g['spread_home']}  total {g['total']}")
