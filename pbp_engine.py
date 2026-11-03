"""
pbp_engine.py - possession-by-possession NBA game simulator for DFS
===================================================================

Each simulated game is played out possession by possession (vectorised over
thousands of sims at once):

  * rotations: 16 three-minute segments, a coach re-picks the 5 on the floor
    from "minutes owed" (target minutes vs minutes played), with stickiness,
    foul trouble, foul-outs, in-game injuries, starters back at halftime /
    in close endgames, and garbage time when the margin is big
  * every possession: turnover (steal) -> shooter by usage -> 2PA/3PA ->
    foul / and-one / free throws -> make or miss -> block -> assist ->
    offensive or defensive rebound -> put-back
  * game environment: pace shock, per-team shooting shock, favourites win more
    (from the spread), overtime when tied
  * player-game random effects (hot shooter, usage, rebounding, passing) so
    fantasy-point variance is realistic, not just binomial noise

Your projections stay the MEAN of every player: shot volume / rebound / assist
propensities are calibrated in a pilot run until each player's expected stat
line matches his projection, then a final tiny rescale makes it exact. What
the engine adds is the SHAPE: minutes, blowouts, foul trouble and teammate
correlations emerge from the game flow instead of being assumed.

Optional: pass nba_api logs and shot mix / FT rate / shooting % / stat shares
are measured from each player's recent games instead of position templates.
"""

import numpy as np

F32 = np.float32

# share of a player's gross FD points that comes from (pts, reb, ast, stl, blk)
ARCH_SHARES = {
    "PG": (0.545, 0.090, 0.280, 0.070, 0.015),
    "SG": (0.640, 0.115, 0.170, 0.060, 0.015),
    "SF": (0.620, 0.170, 0.120, 0.060, 0.030),
    "PF": (0.575, 0.250, 0.085, 0.035, 0.055),
    "C":  (0.500, 0.320, 0.075, 0.025, 0.080),
}
# 3PA share of attempts, 2P%, 3P%, FT%, free-throw rate mult, foul tendency mult
ARCH_SHOT = {
    "PG": (0.42, 0.52, 0.360, 0.82, 0.90, 0.90),
    "SG": (0.46, 0.52, 0.360, 0.82, 0.90, 0.90),
    "SF": (0.40, 0.54, 0.355, 0.79, 1.00, 1.00),
    "PF": (0.30, 0.56, 0.350, 0.76, 1.05, 1.10),
    "C":  (0.12, 0.62, 0.330, 0.68, 1.20, 1.25),
}
POS_ORDER = ["PG", "SG", "SF", "PF", "C"]

PPP = 1.12                 # natural points / possession used to turn a total into pace
SEG_MIN = 3.0
N_SEG = 16


def _primary(positions):
    for p in positions or []:
        p = str(p).upper()
        if p in ARCH_SHARES:
            return p
        if p in ("G",):
            return "SG"
        if p in ("F",):
            return "SF"
        if p in ("UTIL",):
            continue
    return "SF"


def _pick(w, u):
    """Row-wise categorical draw. w (S,k) non-negative, u (S,) uniform -> (S,) index."""
    c = np.cumsum(w, axis=1)
    t = u * c[:, -1]
    return np.minimum((t[:, None] >= c).sum(axis=1), w.shape[1] - 1)


# ==================================================
# PROFILES FROM nba_api LOGS (optional)
# ==================================================

def compute_pbp_profiles(logs, halflife=14, window=40, min_games=6):
    """key -> dict of measured shot mix / shooting % / stat shares / foul rate.
    Needs the raw leaguegamelog columns (fga, fg3a, fta, oreb, pf ...)."""
    need = {"fga", "fg3a", "fgm", "fg3m", "fta", "ftm", "pf", "oreb"}
    if logs is None or not need.issubset(set(logs.columns)):
        return {}
    out = {}
    for key, g in logs.groupby("key"):
        g = g.sort_values("game_date", ascending=False).head(window)
        n = len(g)
        if n < min_games:
            continue
        w = 0.5 ** (np.arange(n) / halflife)

        def s(c):
            return float((pd_num(g[c]) * w).sum())

        fga, fg3a, fgm, fg3m = s("fga"), s("fg3a"), s("fgm"), s("fg3m")
        fta, ftm, pf, mins = s("fta"), s("ftm"), s("pf"), float((g["min"].values * w).sum())
        pts, reb, ast, stl, blk, tov = (s("pts"), s("reb"), s("ast"), s("stl"), s("blk"), s("tov"))
        gross = pts + 1.2 * reb + 1.5 * ast + 3 * stl + 3 * blk
        if fga < 20 or gross <= 0 or mins <= 0:
            continue
        lam = n / (n + 12.0)
        out[key] = dict(
            n=n, lam=lam,
            shares=np.array([pts, 1.2 * reb, 1.5 * ast, 3 * stl, 3 * blk]) / gross,
            r3=fg3a / fga, fg3a=fg3a, fga2=fga - fg3a,
            fg2=(fgm - fg3m) / max(fga - fg3a, 1e-9),
            fg3=fg3m / max(fg3a, 1e-9),
            ft=ftm / max(fta, 1e-9), fta=fta,
            ftr=fta / fga,
            pf_min=pf / mins,
            tov_frac=tov / max(gross, 1e-9),
        )
    return out


def pd_num(series):
    return np.nan_to_num(np.asarray(series, dtype=float))


# ==================================================
# ONE GAME
# ==================================================

class _Team:
    """Static per-team arrays (local player index space is shared by the game)."""


class GameSim:

    def __init__(self, rows, total, margin, rng, env):
        """
        rows: list of dicts per player, in game-local order, with keys
              team (0/1), proj, minutes, pos, starter, ghost, + optional profile
        total: Vegas total; margin: expected margin for team 0
        env: dict of environment sds
        """
        self.rng = rng
        self.env = env
        self.rows = rows
        self.n = len(rows)
        self.total = float(total)
        self.margin = float(margin)

        self.team = np.array([r["team"] for r in rows])
        self.proj = np.array([r["proj"] for r in rows], dtype=float)
        self.tgt_min = np.array([r["minutes"] for r in rows], dtype=float)
        self.starter = np.array([r["starter"] for r in rows], dtype=bool)
        self.ghost = np.array([r["ghost"] for r in rows], dtype=bool)
        self.idx_t = [np.where(self.team == t)[0] for t in (0, 1)]

        self._init_rates()
        self.poss_team = max(80.0, self.total / (2 * PPP))
        self.margin_adj = 0.0

    # ------------------------------------------------------------
    def _init_rates(self):
        n = self.n
        sh = np.zeros((n, 5))
        r3 = np.zeros(n); fg2 = np.zeros(n); fg3 = np.zeros(n); ft = np.zeros(n)
        ftr = np.zeros(n); fouls = np.zeros(n); tovf = np.zeros(n)
        for i, r in enumerate(self.rows):
            a = ARCH_SHARES[r["pos"]]
            b = ARCH_SHOT[r["pos"]]
            shares = np.array(a)
            r3_i, fg2_i, fg3_i, ft_i, ftr_i, pf_i = b
            tov_i = 0.07 * (0.7 + 2.0 * a[2])
            prof = r.get("prof")
            if prof:
                lam = prof["lam"]
                shares = lam * prof["shares"] + (1 - lam) * shares
                shares = shares / shares.sum()
                l3 = prof["fg3a"] / (prof["fg3a"] + 100.0)
                l2 = prof["fga2"] / (prof["fga2"] + 150.0)
                lf = prof["fta"] / (prof["fta"] + 60.0)
                r3_i = lam * prof["r3"] + (1 - lam) * r3_i
                fg2_i = l2 * prof["fg2"] + (1 - l2) * fg2_i
                fg3_i = l3 * prof["fg3"] + (1 - l3) * fg3_i
                ft_i = lf * prof["ft"] + (1 - lf) * ft_i
                ftr_i = lam * (prof["ftr"] / 0.25) + (1 - lam) * ftr_i
                pf_i = lam * (prof["pf_min"] / 0.075) + (1 - lam) * pf_i
                tov_i = lam * prof["tov_frac"] + (1 - lam) * tov_i
            sh[i] = shares
            r3[i] = np.clip(r3_i, 0.0, 0.8)
            fg2[i] = np.clip(fg2_i, 0.40, 0.72)
            fg3[i] = np.clip(fg3_i, 0.26, 0.45)
            ft[i] = np.clip(ft_i, 0.45, 0.95)
            ftr[i] = np.clip(ftr_i, 0.4, 2.0)
            fouls[i] = np.clip(pf_i, 0.5, 1.8)
            tovf[i] = np.clip(tov_i, 0.02, 0.14)
        self.r3, self.fg2, self.fg3, self.ft, self.ftr, self.foulw = r3, fg2, fg3, ft, ftr, fouls

        gross = self.proj / np.maximum(1.0 - tovf, 0.5)       # gross = P + tov_target
        tov_t = gross * tovf
        gross = self.proj + tov_t
        m = np.maximum(self.tgt_min, 1.0)
        # stat targets over the player's expected minutes
        self.t_pts = gross * sh[:, 0]
        self.t_reb = gross * sh[:, 1] / 1.2
        self.t_ast = gross * sh[:, 2] / 1.5
        self.t_stl = gross * sh[:, 3] / 3.0
        self.t_blk = gross * sh[:, 4] / 3.0
        self.t_tov = tov_t
        # initial propensities (per minute on the floor)
        self.u = np.maximum(self.t_pts / m, 0.03)
        self.rb = np.maximum(self.t_reb / m, 0.01)
        self.a = np.maximum(self.t_ast / m, 0.005)
        self.s = np.maximum(self.t_stl / m, 0.003)
        self.b = np.maximum(self.t_blk / m, 0.002)
        self.t = np.maximum(self.t_tov / m, 0.004)

    # ------------------------------------------------------------
    def _draw_effects(self, S):
        """Per-sim, per-player random effects and per-game environment."""
        rng, env, n = self.rng, self.env, self.n

        def ln(sd, shape):
            return np.exp(sd * rng.standard_normal(shape, dtype=F32) - 0.5 * sd * sd).astype(F32)

        self.m_shoot = ln(env["sd_shoot"], (S, n))
        self.m_use = ln(env["sd_use"], (S, n))
        self.m_reb = ln(env["sd_reb"], (S, n))
        self.m_ast = ln(env["sd_ast"], (S, n))
        self.m_stl = ln(env["sd_def"], (S, n))
        self.m_tov = ln(env["sd_tov"], (S, n))
        self.pace = np.clip(1 + env["sd_pace"] * rng.standard_normal(S), 0.88, 1.12)
        # team shooting shock (shared by that team's players)
        shock = env["sd_team"] * rng.standard_normal((S, 2))
        gshock = env["sd_game"] * rng.standard_normal(S)
        half = (self.margin + self.margin_adj) / (2 * self.poss_team * PPP)
        eff0 = 1 + half
        eff1 = 1 - half
        self.eff = np.stack([eff0 * (1 + shock[:, 0] + gshock), eff1 * (1 + shock[:, 1] + gshock)],
                            axis=1).astype(F32)
        # per-game minutes multiplier (coach / matchup noise on a player's target)
        self.min_mult = np.exp(env["sd_min"] * rng.standard_normal((S, n)) - 0.5 * env["sd_min"] ** 2).astype(F32)
        # coach garbage-time threshold multiplier
        self.coach = rng.uniform(0.80, 1.25, (S, 2)).astype(F32)
        # in-game injury: segment after which the player is gone (99 = never)
        hit = rng.random((S, n)) < env["inj_rate"]
        self.exit_seg = np.where(hit, rng.integers(1, N_SEG, (S, n)), 99).astype(np.int16)

    # ------------------------------------------------------------
    def _reset(self, S):
        n = self.n
        self.S = S
        self.ar = np.arange(S)
        z = lambda: np.zeros((S, n), dtype=F32)
        self.pts, self.reb, self.ast, self.stl, self.blk, self.tov, self.pf = (z() for _ in range(7))
        self.played = z()
        self.score = np.zeros((S, 2), dtype=np.int32)
        self.on = np.zeros((S, 10), dtype=np.int32)
        self.on_prev = np.zeros((S, n), dtype=bool)
        # per-row propensity arrays (S, n)
        self.uw = (self.u[None, :] * self.m_use).astype(F32)
        self.rw = (self.rb[None, :] * self.m_reb).astype(F32)
        self.aw = (self.a[None, :] * self.m_ast).astype(F32)
        self.sw = (self.s[None, :] * self.m_stl).astype(F32)
        self.bw = (self.b[None, :] * self.m_stl).astype(F32)
        self.tw = (self.t[None, :] * self.m_tov).astype(F32)
        self.foulw_full = np.broadcast_to(self.foulw.astype(F32), (S, n))
        # lineup-normalisers (team average weight, minute-weighted)
        self.ref = {}
        for nm, arr in (("t", self.t), ("a", self.a), ("s", self.s), ("b", self.b)):
            self.ref[nm] = [
                float(np.average(arr[ix], weights=np.maximum(self.tgt_min[ix], 1.0)))
                for ix in self.idx_t
            ]

    # ------------------------------------------------------------
    def _priority(self, seg, rows_mask=None):
        """(S, n) priority; the top 5 per team take the floor."""
        S, n = self.S, self.n
        elapsed = seg * SEG_MIN
        owed = (self.tgt_min[None, :] * self.min_mult) * (min(elapsed + SEG_MIN, 48.0) / 48.0) - self.played
        pr = owed + 1.0 * self.rng.standard_normal((S, n)).astype(F32) + 0.8 * self.on_prev
        if seg == 0:
            pr = pr + 100.0 * self.starter[None, :]
        elif seg == 8:
            pr = pr + 3.0 * self.starter[None, :]
        # foul trouble
        pf = self.pf
        pen = np.zeros((S, n), dtype=F32)
        if seg < 8:
            pen -= 6.0 * (pf >= 3)
        elif seg < 12:
            pen -= 6.0 * (pf >= 4)
        else:
            pen -= 4.0 * (pf >= 5)
        pr = pr + pen
        # garbage time
        if seg >= 12:
            marg = np.abs(self.score[:, 0] - self.score[:, 1]).astype(F32)
            k = seg - 12
            thr = np.array([20.0, 17.0, 14.0, 12.0])[min(k, 3)]
            for t in (0, 1):
                lead = (self.score[:, t] > self.score[:, 1 - t])
                th = (thr + np.where(lead, 0.0, 3.0)) * self.coach[:, t]
                blow = marg >= th
                sel = self.starter & (self.team == t)
                pr[:, sel] -= 12.0 * blow[:, None]
            if seg >= 14:
                close = marg <= 8
                pr[:, self.starter] += 1.5 * close[:, None]
        # unavailable: fouled out or hurt
        gone = (pf >= 6) | (self.exit_seg <= seg)
        pr = np.where(gone, -1e4, pr)
        return pr

    def _set_lineups(self, seg, rows=None):
        pr = self._priority(seg)
        for t in (0, 1):
            ix = self.idx_t[t]
            sub = pr[:, ix]
            top = np.argpartition(-sub, 4, axis=1)[:, :5]
            gi = ix[top]
            if rows is None:
                self.on[:, t * 5:(t + 1) * 5] = gi
            else:
                self.on[rows, t * 5:(t + 1) * 5] = gi[rows]
        self.on_prev[:] = False
        self.on_prev[self.ar[:, None], self.on] = True

    # ------------------------------------------------------------
    def _attempt(self, o, mask, second=False, rebber=None):
        """One shot attempt (+ fouls, FTs, block, assist, rebound).
        Returns (oreb_mask, rebounder_game_idx)."""
        rng, ar = self.rng, self.ar
        d = 1 - o
        oi = self.on[:, o * 5:(o + 1) * 5]
        di = self.on[:, d * 5:(d + 1) * 5]
        uw_o = np.take_along_axis(self.uw, oi, axis=1)
        sp = _pick(uw_o, rng.random(self.S))
        if second and rebber is not None:
            # put-back: the rebounder shoots more often than usage says
            reb_pos = (oi == rebber[:, None]).argmax(axis=1)
            take = rng.random(self.S) < 0.55
            sp = np.where(take, reb_pos, sp)
        shooter = oi[ar, sp]

        three = rng.random(self.S) < self.r3[shooter]
        if second:
            three &= rng.random(self.S) < 0.15

        # fouls: shooting foul / and-one
        p_sf = np.where(three, 0.035, np.clip(0.125 * self.ftr[shooter], 0.02, 0.45))
        foul = mask & (rng.random(self.S) < p_sf)
        andone = foul & (rng.random(self.S) < 0.16) & ~three
        shoot_foul = foul & ~andone
        # who fouled
        fdef = np.take_along_axis(self.foulw_full, di, axis=1)
        fd = di[ar, _pick(fdef, rng.random(self.S))]
        self.pf[ar, fd] += foul.astype(F32)

        # make / miss
        base = np.where(three, self.fg3[shooter], self.fg2[shooter])
        lead = np.clip(self.score[:, o] - self.score[:, d], -25, 25)
        pm = base * self.m_shoot[ar, shooter] * self.eff[:, o] * (1.0 - self.env["reversion"] * lead)
        if second:
            pm = pm + 0.07
        pm = np.clip(pm, 0.05, 0.88)
        made = (mask & ~shoot_foul & (rng.random(self.S) < pm)) | andone
        pts_fg = np.where(three, 3, 2) * made
        # free throws
        n_ft = np.where(andone, 1, np.where(shoot_foul, np.where(three, 3, 2), 0))
        ftp = np.clip(self.ft[shooter] * self.m_shoot[ar, shooter] ** 0.3, 0.4, 0.97)
        ft_made = rng.binomial(n_ft, ftp)
        tot = (pts_fg + ft_made) * mask
        self.pts[ar, shooter] += tot
        self.score[:, o] += tot.astype(np.int32)

        # assist
        aw_o = np.take_along_axis(self.aw, oi, axis=1).copy()
        aref = self.ref["a"][o]
        p_ast = np.clip(np.where(three, 0.80, 0.52) * aw_o.mean(axis=1) / aref, 0.15, 0.92)
        aw_o[ar, sp] = 0.0
        has = made if not second else (made & (rng.random(self.S) < 0.25))
        has = has & (rng.random(self.S) < p_ast)
        asst = oi[ar, _pick(aw_o + 1e-6, rng.random(self.S))]
        self.ast[ar, asst] += has.astype(F32)

        # miss -> block -> rebound
        miss = mask & ~made & ~shoot_foul
        bw_d = np.take_along_axis(self.bw, di, axis=1)
        p_blk = np.clip(0.115 * bw_d.mean(axis=1) / self.ref["b"][d], 0.0, 0.45)
        blocked = miss & ~three & (rng.random(self.S) < p_blk)
        blk_by = di[ar, _pick(bw_d, rng.random(self.S))]
        self.blk[ar, blk_by] += blocked.astype(F32)

        rw_o = np.take_along_axis(self.rw, oi, axis=1)
        rw_d = np.take_along_axis(self.rw, di, axis=1)
        if second:
            oreb = np.zeros(self.S, dtype=bool)
        else:
            oreb = miss & (rng.random(self.S) < 0.265)
        rb_o = oi[ar, _pick(rw_o, rng.random(self.S))]
        rb_d = di[ar, _pick(rw_d, rng.random(self.S))]
        rebounder = np.where(oreb, rb_o, rb_d)
        self.reb[ar, rebounder] += miss.astype(F32)
        return oreb, rebounder

    def _possession(self, o, act):
        rng, ar = self.rng, self.ar
        d = 1 - o
        oi = self.on[:, o * 5:(o + 1) * 5]
        di = self.on[:, d * 5:(d + 1) * 5]

        # non-shooting fouls (defence)
        nf = act & (rng.random(self.S) < 0.075)
        if nf.any():
            fw = np.take_along_axis(self.foulw_full, di, axis=1)
            fd = di[ar, _pick(fw, rng.random(self.S))]
            self.pf[ar, fd] += nf.astype(F32)

        tw_o = np.take_along_axis(self.tw, oi, axis=1)
        p_tov = np.clip(0.135 * tw_o.mean(axis=1) / self.ref["t"][o], 0.05, 0.30)
        tov = act & (rng.random(self.S) < p_tov)
        if tov.any():
            tv = oi[ar, _pick(tw_o, rng.random(self.S))]
            self.tov[ar, tv] += tov.astype(F32)
            sw_d = np.take_along_axis(self.sw, di, axis=1)
            p_st = np.clip(0.58 * sw_d.mean(axis=1) / self.ref["s"][d], 0.1, 0.9)
            steal = tov & (rng.random(self.S) < p_st)
            st = di[ar, _pick(sw_d, rng.random(self.S))]
            self.stl[ar, st] += steal.astype(F32)

        go = act & ~tov
        oreb, rebber = self._attempt(o, go)
        if oreb.any():
            self._attempt(o, oreb, second=True, rebber=rebber)

    # ------------------------------------------------------------
    def _foulout_fix(self, seg):
        """Replace anyone who just fouled out (rare; recompute for those rows)."""
        bad = (self.pf[self.ar[:, None], self.on] >= 6).any(axis=1)
        if bad.any():
            rows = np.where(bad)[0]
            self._set_lineups(seg, rows=rows)

    def play(self, S):
        """Play S games; returns dict of (S, n) stat arrays and game info."""
        self._draw_effects(S)
        self._reset(S)
        rng = self.rng
        seg_poss = self.poss_team / N_SEG
        for seg in range(N_SEG):
            self._set_lineups(seg)
            n_row = np.clip(np.round(seg_poss * self.pace + 0.7 * rng.standard_normal(S)), 3, 12).astype(int)
            for k in range(int(n_row.max())):
                act = n_row > k
                self._possession(0, act)
                self._possession(1, act)
                if k % 3 == 2:
                    self._foulout_fix(seg)
            self.played[self.ar[:, None], self.on] += SEG_MIN
        # overtime
        ot = np.zeros(S, dtype=np.int8)
        live = self.score[:, 0] == self.score[:, 1]
        for _ in range(4):
            if not live.any():
                break
            ot += live
            self._set_lineups(N_SEG - 1)
            n_row = np.clip(np.round(self.poss_team / 9.6 * self.pace + 0.5 * rng.standard_normal(S)), 6, 14).astype(int)
            for k in range(int(n_row.max())):
                act = (n_row > k) & live
                self._possession(0, act)
                self._possession(1, act)
            self.played[self.ar[:, None], self.on] += 5.0 * live[:, None]
            live = live & (self.score[:, 0] == self.score[:, 1])
        fp = (self.pts + 1.2 * self.reb + 1.5 * self.ast + 3 * self.stl
              + 3 * self.blk - self.tov).astype(F32)
        return dict(fp=fp, pts=self.pts, reb=self.reb, ast=self.ast, stl=self.stl,
                    blk=self.blk, tov=self.tov, pf=self.pf, minutes=self.played,
                    score=self.score.copy(), ot=ot > 0)

    # ------------------------------------------------------------
    def calibrate(self, iters=3, S=300, log=None):
        """Scale propensities until expected stats match each player's targets."""
        for it in range(iters):
            r = self.play(S)
            mean = {k: r[k].mean(axis=0) for k in ("pts", "reb", "ast", "stl", "blk", "tov", "minutes")}

            def upd(cur, tgt, act, floor, damp=0.8):
                ratio = np.clip(tgt / np.maximum(act, 1e-3), 0.5, 2.0) ** damp
                return np.maximum(cur * ratio, floor)

            mm = float((r["score"][:, 0] - r["score"][:, 1]).mean())
            self.margin_adj += 0.8 * (self.margin - mm)
            self.u = upd(self.u, self.t_pts, mean["pts"], 0.02)
            self.rb = upd(self.rb, self.t_reb, mean["reb"], 0.005)
            self.a = upd(self.a, self.t_ast, mean["ast"], 0.003)
            self.s = upd(self.s, self.t_stl, mean["stl"], 0.002)
            self.b = upd(self.b, self.t_blk, mean["blk"], 0.001)
            self.t = upd(self.t, self.t_tov, mean["tov"], 0.003)
            if log:
                fp = (mean["pts"] + 1.2 * mean["reb"] + 1.5 * mean["ast"] + 3 * mean["stl"]
                      + 3 * mean["blk"] - mean["tov"])
                err = np.abs(fp - self.proj)[~self.ghost]
                log(f"  calibration pass {it + 1}: mean |FP error| {err.mean():.2f}")


# ==================================================
# ENGINE OVER A WHOLE SLATE
# ==================================================

DEFAULT_ENV = dict(
    sd_shoot=0.06, sd_use=0.20, sd_reb=0.15, sd_ast=0.18, sd_def=0.30, sd_tov=0.25,
    sd_pace=0.045, sd_team=0.012, sd_game=0.025, inj_rate=0.012, sd_min=0.09,
    reversion=0.004,
)


class PBPEngine:

    def __init__(self, pool, rng, profiles=None, env=None, calib_sims=300, calib_iters=3):
        self.pool = pool
        self.rng = rng
        self.profiles = profiles or {}
        self.env = dict(DEFAULT_ENV, **(env or {}))
        self.calib_sims = calib_sims
        self.calib_iters = calib_iters

    # ------------------------------------------------------------
    def _team_rows(self, players, t, key_fn):
        """Players of one team -> row dicts, plus ghosts so the rotation is real."""
        rows = []
        mins = []
        for p in players:
            m = p.minutes if p.minutes > 0 else float(38 * (1 - np.exp(-p.projection / 18)))
            mins.append(m)
        mins = np.array(mins, dtype=float)
        if mins.sum() > 240:
            mins = mins * 240.0 / mins.sum()
        order = np.argsort(-mins)
        starters = set(order[:5].tolist())
        for i, p in enumerate(players):
            rows.append(dict(team=t, proj=p.projection, minutes=float(mins[i]),
                             pos=_primary(p.positions), starter=i in starters,
                             ghost=False, prof=self.profiles.get(key_fn(p.name)), player=p))
        # ghosts: fill to 9 bodies and 240 minutes
        short = max(0.0, 240.0 - mins.sum())
        n_g = max(0, 9 - len(rows))
        if short > 4 and n_g == 0:
            n_g = 1
        if len(rows) < 5:
            n_g = max(n_g, 5 - len(rows) + 2)
        if n_g:
            each = short / n_g if short > 0 else 8.0
            for j in range(n_g):
                st = len(rows) < 5
                rows.append(dict(team=t, proj=each * 0.62, minutes=each,
                                 pos=POS_ORDER[(len(rows) + j) % 5], starter=st,
                                 ghost=True, prof=None, player=None))
        # make sure exactly 5 starters
        real_order = sorted(range(len(rows)), key=lambda i: -rows[i]["minutes"])
        st = set(real_order[:5])
        for i, r in enumerate(rows):
            r["starter"] = i in st
        return rows

    def run(self, S, log=print):
        from nba_news import name_key
        pool, rng = self.pool, self.rng
        players = pool.all_players()
        N, G = len(players), len(pool.game_teams)
        sims = np.zeros((S, N), dtype=F32)
        starters = np.zeros(N, dtype=bool)
        team_scores = np.zeros((S, G, 2), dtype=F32)
        ot_flags = np.zeros((S, G), dtype=bool)
        game_factor = np.zeros((S, G), dtype=F32)

        for g in range(G):
            members = pool.game_teams[g]
            tA = members[0]
            gp = [p for p in players if pool.game_idx[p.idx] == g]
            A = [p for p in gp if p.team == tA]
            B = [p for p in gp if p.team != tA]
            rows = self._team_rows(A, 0, name_key) + self._team_rows(B, 1, name_key)
            total = float(pool.game_total[g]) if np.isfinite(pool.game_total[g]) else 225.0
            margin = float(pool.game_margin[g])
            gs = GameSim(rows, total, margin, rng, self.env)
            log(f"PBP game {g + 1}/{G}: {'/'.join(members)}  (total {total:g}, margin {margin:+.1f})")
            gs.calibrate(self.calib_iters, self.calib_sims)
            r = gs.play(S)
            fp = r["fp"]
            # final exact anchor to projections (tiny after calibration)
            for j, row in enumerate(rows):
                p = row["player"]
                if p is None:
                    continue
                mu = float(fp[:, j].mean())
                if mu > 0:
                    fp[:, j] *= float(np.clip(p.projection / mu, 0.85, 1.15))
                sims[:, p.idx] = fp[:, j]
                starters[p.idx] = row["starter"]
            team_scores[:, g, 0] = r["score"][:, 0]
            team_scores[:, g, 1] = r["score"][:, 1]
            ot_flags[:, g] = r["ot"]
            tot = r["score"].sum(axis=1).astype(float)
            game_factor[:, g] = (tot - tot.mean()) / max(tot.mean(), 1.0)
        return dict(sims=sims, game_factor=game_factor, starters=starters,
                    team_scores=team_scores, overtime_flags=ot_flags)
