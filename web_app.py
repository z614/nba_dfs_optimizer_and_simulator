"""
web_app.py - run the optimizer from your phone
==============================================

    python web_app.py            (on the PC that has the optimizer)

Then open the printed address on your phone (Tailscale address works from
anywhere) and use "Add to Home Screen" - it behaves like an app.

The PC does all the work (simulation, optimizer, news feed); the phone is a
remote control with: slate upload, settings, run/stop with live progress,
lineups, exposures, player pool, live injury news with alerts, quick
"tatum out" updates, one-tap rebuild, and CSV download for FanDuel.

No tkinter and no extra packages needed (standard library + your existing
numpy/pandas/scipy). Access needs the private key that is part of the
address; keep the address to yourself and run it only on a private network
(home wifi / Tailscale).
"""

import argparse
import csv
import importlib
import io
import json
import math
import os
import re
import secrets
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import zlib
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
UPLOADS = os.path.join(HERE, "uploads")
PREFS = os.path.join(HERE, "web_prefs.json")
MAX_UPLOAD = 25 * 1024 * 1024

# (attr, label, type, extra)   pct = stored as 0-1, shown as 0-100
FIELDS = [
    ("lineups", "Lineups", "int", (1, 500)),
    ("sims", "Simulations", "int", (500, 50000)),
    ("field_size", "Field size", "int", (200, 20000)),
    ("max_exposure", "Max exposure %", "pct", (1, 100)),
    ("max_overlap", "Max overlap", "int", (1, 8)),
    ("max_per_team", "Max per team", "int", (1, 8)),
    ("min_stack", "Min stack", "int", (0, 8)),
    ("leverage", "Leverage", "float", (0, 1)),
    ("randomness", "Randomness", "float", (0, 1)),
    ("candidate_mult", "Candidate multiple", "float", (1, 20)),
    ("sim_engine", "Simulation engine", "choice", ["game", "pbp", "blend"]),
    ("pbp_blend", "Play-by-play share (blend)", "float", (0, 1)),
    ("use_news", "Live injury news + lines", "bool", None),
    ("news_refresh_lines", "Use live lines over slate file", "bool", None),
    ("news_own_boost", "News ownership boost", "float", (0, 1)),
    ("use_nba_feed", "NBA game-log feed", "bool", None),
    ("contest_type", "Contest type", "choice", ["Single Entry", "3-Max", "150-Max"]),
    ("roster_format", "Roster format", "choice", None),     # filled from the engine
]


def clean(o):
    """Make anything JSON-safe (numpy, NaN, inf)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def jdump(o):
    return json.dumps(clean(o))


# ==================================================
# ICONS (pure python PNG, no imaging library)
# ==================================================

def make_png(size):
    y, x = np.mgrid[0:size, 0:size].astype(float)
    c = (size - 1) / 2
    r = np.hypot(x - c, y - c) / (size / 2)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (17, 17, 20)
    ball = r < 0.78
    img[ball] = (232, 119, 34)
    line_w = 0.035
    seams = (np.abs(x - c) < line_w * size) | (np.abs(y - c) < line_w * size)
    seams |= np.abs(np.hypot(x - c * 2.1, y - c) / (size / 2) - 0.95) < line_w * 1.6
    seams |= np.abs(np.hypot(x + c * 0.1, y - c) / (size / 2) - 0.95) < line_w * 1.6
    img[ball & seams] = (30, 20, 12)
    raw = b"".join(b"\x00" + img[i].tobytes() for i in range(size))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


# ==================================================
# UI OBJECT HANDED TO THE PIPELINE
# ==================================================

class WebUI:
    """Same interface as the Tk GUI / ConsoleUI."""

    def __init__(self, app):
        self.app = app
        self.lock = threading.Lock()
        self.status = "Idle"
        self.cur = 0
        self.total = 0
        self.logs = deque(maxlen=600)
        self.run_stats = {}

    def log(self, msg):
        msg = str(msg)
        print(msg)
        with self.lock:
            self.logs.append(msg)

    def update_progress(self, cur, total):
        self.cur, self.total = cur, total

    def set_status(self, s):
        self.status = str(s)
        print(f"[{s}]")

    def stats(self, **kw):
        self.run_stats.update(kw)

    def post(self, kind, payload=None):
        self.app.on_event(kind, payload)

    def tail(self, n=40):
        with self.lock:
            return list(self.logs)[-n:]


# ==================================================
# APP
# ==================================================

class WebApp:

    def __init__(self, opt, port=8765, key=None):
        self.opt = opt
        self.port = int(port)
        prefs = self._read_prefs()
        self.key = key or prefs.get("key") or secrets.token_urlsafe(12)
        self.prefs = prefs
        self.prefs["key"] = self.key
        self.settings = opt.Settings()
        self.ui = WebUI(self)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.running = False
        self.run_id = 0
        self.results = None
        self.run_slots = []
        self.run_format = ""
        self.preview = None
        self.news_snap = None
        self.news_pool = None
        self.last_error = ""
        self.last_run_secs = 0.0
        os.makedirs(UPLOADS, exist_ok=True)

        self.fields = [(a, l, t, (list(opt.ROSTER_FORMATS) if a == "roster_format" else e)) for a, l, t, e in FIELDS]
        self._apply_saved_settings()

        from news_monitor import MonitorController
        self.monitor = MonitorController(self.settings, self.slate_names, self.lineup_counts,
                                         self.post_event, self.ui.log)

    # ---------------- prefs
    def _read_prefs(self):
        try:
            with open(PREFS, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_prefs(self):
        s = self.settings
        saved = {}
        for a, _, t, _e in self.fields:
            saved[a] = getattr(s, a)
        saved.update(slate_path=s.slate_path, template_path=s.template_path,
                     news_overrides=s.news_overrides)
        self.prefs["settings"] = saved
        try:
            with open(PREFS, "w", encoding="utf-8") as f:
                json.dump(clean(self.prefs), f)
        except Exception:
            pass

    def _apply_saved_settings(self):
        saved = self.prefs.get("settings", {})
        s = self.settings
        for a, _, t, _e in self.fields:
            if a in saved:
                try:
                    setattr(s, a, saved[a])
                except Exception:
                    pass
        for a in ("slate_path", "template_path"):
            p = saved.get(a, "")
            if p and os.path.exists(p):
                setattr(s, a, p)
        s.news_overrides = dict(saved.get("news_overrides", {}) or {})
        if s.roster_format not in self.opt.ROSTER_FORMATS:
            s.roster_format = self.opt.DEFAULT_FORMAT
        # lineups saved from a previous session are not restored; the run is cheap to repeat

    # ---------------- events
    def on_event(self, kind, payload):
        if kind == "news":
            snap, _rep, pool = payload
            self.news_snap, self.news_pool = snap, pool
            self.monitor.set_baseline(snap, self.settings.news_overrides)

    def post_event(self, kind, payload=None):
        if kind == "rebuild":
            ok, msg = self.start_run()
            self.ui.log("Rebuild requested: " + msg)

    # ---------------- names for the monitor
    def _pools(self):
        res = self.results
        return [self.preview, self.news_pool, getattr(res, "pool", None) if res else None]

    def slate_names(self):
        from nba_news import name_key
        out = {}
        for pool in self._pools():
            if pool is None:
                continue
            try:
                for p in pool.all_players():
                    out[name_key(p.name)] = p.name
                for n in list(getattr(pool, "news_removed", {}) or {}):
                    out[name_key(n)] = n
            except Exception:
                pass
        return out

    def lineup_counts(self):
        from nba_news import name_key
        res = self.results
        if not res or not res.lineups:
            return {}, 0
        try:
            pl = res.pool.all_players()
            counts = {}
            for lu in res.lineups:
                for i in lu:
                    k = name_key(pl[i].name)
                    counts[k] = counts.get(k, 0) + 1
            return counts, len(res.lineups)
        except Exception:
            return {}, 0

    # ---------------- files
    def list_files(self):
        seen, out = set(), []
        for folder in (UPLOADS, HERE):
            try:
                for n in os.listdir(folder):
                    p = os.path.join(folder, n)
                    if n.lower().endswith(".csv") and os.path.isfile(p) and p not in seen:
                        seen.add(p)
                        out.append(dict(name=n, where="uploads" if folder == UPLOADS else "app folder",
                                        path=p, size=os.path.getsize(p), mtime=os.path.getmtime(p)))
            except Exception:
                pass
        out.sort(key=lambda d: -d["mtime"])
        return out

    def select_file(self, kind, path):
        allowed = {d["path"] for d in self.list_files()}
        if path not in allowed:
            return False, "file not found"
        if kind == "slate":
            self.settings.slate_path = path
            ok, msg = self.load_preview()
            self.save_prefs()
            return ok, msg
        if kind == "template":
            self.settings.template_path = path
            self.save_prefs()
            return True, f"template: {os.path.basename(path)}"
        return False, "unknown kind"

    def load_preview(self):
        s = self.settings
        try:
            pool = self.opt.DataLoader().load_slate(s.slate_path, s, self.ui.log)
            self.opt.OwnershipEngine().apply(pool, len(s.slots()), log=self.ui.log)
            self.preview = pool
            return True, f"slate loaded: {len(pool)} players"
        except Exception as e:
            self.preview = None
            return False, f"could not read slate: {e}"

    def save_upload(self, kind, filename, data):
        name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(filename or "upload.csv"))[:80]
        if not name.lower().endswith(".csv"):
            name += ".csv"
        path = os.path.join(UPLOADS, name)
        with open(path, "wb") as f:
            f.write(data)
        ok, msg = self.select_file(kind, path)
        return ok, msg

    # ---------------- settings
    def apply_settings(self, d):
        s = self.settings
        errs = []
        for a, label, t, extra in self.fields:
            if a not in d:
                continue
            v = d[a]
            try:
                if t == "int":
                    v = int(float(v))
                elif t == "float":
                    v = float(v)
                elif t == "pct":
                    v = float(v) / 100.0
                elif t == "bool":
                    v = bool(v) if isinstance(v, bool) else str(v).lower() in ("1", "true", "on", "yes")
                elif t == "choice":
                    if v not in extra:
                        raise ValueError("not an option")
                if t in ("int", "float") and extra:
                    v = min(max(v, extra[0]), extra[1])
                if t == "pct":
                    v = min(max(v, extra[0] / 100.0), extra[1] / 100.0)
                setattr(s, a, v)
            except Exception:
                errs.append(f"{label}: invalid value")
        # keep the cap/floor sane for the chosen format
        self.save_prefs()
        return errs

    def settings_json(self):
        out = {}
        for a, _l, t, _e in self.fields:
            v = getattr(self.settings, a)
            out[a] = v * 100.0 if t == "pct" else v
        return out

    # ---------------- run control
    def start_run(self):
        s = self.settings
        with self.lock:
            if self.running:
                return False, "a run is already in progress"
            if not s.slate_path or not os.path.exists(s.slate_path):
                return False, "choose a slate file first"
            self.running = True
            self.last_error = ""
            self.stop_event.clear()
            self.run_slots = [n for n, _ in s.slots()]
            self.run_format = s.roster_format
        self.ui.cur = self.ui.total = 0
        self.save_prefs()

        def work():
            t0 = time.time()
            try:
                res = self.opt.Pipeline(s, self.ui, self.stop_event).run()
                self.results = res
                self.run_id += 1
                try:
                    self.opt.SlateArchive().archive_run(s, res, self.ui.log)
                except Exception as e:
                    self.ui.log(f"Slate archive failed: {e}")
                self.ui.set_status(f"Done - {len(res.lineups)} lineups")
            except Exception as e:
                self.last_error = str(e)
                self.ui.log(traceback.format_exc())
                self.ui.set_status("Error - see log")
            finally:
                self.last_run_secs = time.time() - t0
                self.running = False

        threading.Thread(target=work, daemon=True).start()
        return True, "run started"

    def stop_run(self):
        self.stop_event.set()
        return "stopping..."

    # ---------------- JSON views
    def state_json(self):
        st = self.monitor.state.snapshot(lambda k: self.lineup_counts()[0].get(k, 0),
                                         self.lineup_counts()[1], self.settings.news_overrides)
        res = self.results
        return dict(
            running=self.running, status=self.ui.status, cur=self.ui.cur, total=self.ui.total,
            log=self.ui.tail(40), run_id=self.run_id, n_lineups=len(res.lineups) if res else 0,
            error=self.last_error, last_run_secs=self.last_run_secs,
            settings=self.settings_json(),
            schema=[dict(key=a, label=l, type=t, extra=e) for a, l, t, e in self.fields],
            slate=os.path.basename(self.settings.slate_path or ""),
            template=os.path.basename(self.settings.template_path or ""),
            files=[dict(name=d["name"], where=d["where"], path=d["path"], size=d["size"]) for d in self.list_files()],
            alerts=st["alerts"][:10], monitor=dict(running=self.monitor.monitor is not None,
                                                   last_check=st["last_check"], eta=st["eta"],
                                                   error=st["error"], source=st["source"]),
            now=time.time(),
        )

    def lineups_json(self):
        res = self.results
        if not res:
            return dict(lineups=[])
        pl = res.pool.all_players()
        slots = self.run_slots or [n for n, _ in self.settings.slots()]
        out = []
        for k, (lu, st) in enumerate(zip(res.lineups, res.lineup_stats)):
            out.append(dict(
                n=k + 1, salary=st.get("salary"), proj=st.get("proj"), own=st.get("own"),
                top1=st.get("top1"), roi=st.get("roi"), corr=st.get("corr"),
                players=[dict(slot=slots[j] if j < len(slots) else "", name=pl[i].name, team=pl[i].team,
                              salary=pl[i].salary, proj=pl[i].projection, own=pl[i].ownership * 100,
                              risk=getattr(pl[i], "play_prob", 1.0) < 0.995)
                         for j, i in enumerate(lu)]))
        return dict(lineups=out, run_id=self.run_id)

    def exposure_json(self):
        res = self.results
        if not res or not res.lineups:
            return dict(rows=[], n=0)
        pl = res.pool.all_players()
        n = len(res.lineups)
        cnt = {}
        for lu in res.lineups:
            for i in lu:
                cnt[i] = cnt.get(i, 0) + 1
        rows = []
        for i, c in cnt.items():
            p = pl[i]
            exp = 100.0 * c / n
            rows.append(dict(name=p.name, team=p.team, exp=exp, own=p.ownership * 100,
                             lev=exp - p.ownership * 100, proj=p.projection, salary=p.salary,
                             risk=getattr(p, "play_prob", 1.0) < 0.995))
        rows.sort(key=lambda r: -r["exp"])
        return dict(rows=rows, n=n)

    def pool_json(self):
        pool = self.results.pool if self.results else self.preview
        if pool is None:
            return dict(rows=[], source="none")
        rows = []
        for p in pool.all_players():
            rows.append(dict(name=p.name, team=p.team, pos="/".join(p.positions), salary=p.salary,
                             proj=p.projection, own=p.ownership * 100,
                             value=p.projection / max(p.salary, 1) * 1000, minutes=p.minutes,
                             play=getattr(p, "play_prob", 1.0), status=getattr(p, "injury", "") or "",
                             opp=p.opponent))
        return dict(rows=rows, source="results" if self.results else "preview")

    def news_json(self):
        from nba_news import name_key
        snap = self.news_snap
        slate = set(self.slate_names())
        inj = []
        if snap is not None:
            for k, i in snap.injuries.items():
                inj.append(dict(name=i["name"], team=i["team"], status=i["status"], prob=i["prob"],
                                note=i.get("comment", "")[:240], slate=k in slate))
            inj.sort(key=lambda r: (not r["slate"], r["prob"]))
        mon = self.monitor
        return dict(
            injuries=inj, overrides=dict(self.settings.news_overrides),
            age=(time.time() - snap.fetched_at) / 60 if snap and snap.fetched_at else None,
            source=snap.source if snap else "none",
            monitor=dict(running=mon.monitor is not None,
                         interval=self.prefs.get("interval", 90), topic=self.prefs.get("topic", "")),
            messages=mon.state.messages[:4],
        )

    def fetch_news(self):
        import nba_news as nn
        snap = nn.NewsFeed(log=self.ui.log).get(force=True)
        self.news_snap = snap
        self.monitor.known.update({k: i["name"] for k, i in snap.injuries.items()})
        return f"{len(snap.injuries)} injury rows, {len(snap.games)} games ({snap.source})"

    def set_override(self, name, prob):
        self.settings.news_overrides[name] = min(max(float(prob), 0.0), 1.0)
        self.save_prefs()

    def clear_override(self, name):
        self.settings.news_overrides.pop(name, None)
        self.save_prefs()

    def monitor_control(self, d):
        if d.get("action") == "stop":
            self.monitor.stop()
            return "monitor stopped"
        interval = max(30.0, float(d.get("interval") or 90))
        topic = re.sub(r"[^A-Za-z0-9_-]", "", str(d.get("topic") or ""))
        self.prefs.update(interval=interval, topic=topic, monitor=True)
        self.save_prefs()
        self.monitor.start(interval, topic, False, 0, click_url=self.url_hint())
        if self.news_snap is not None:
            self.monitor.set_baseline(self.news_snap, self.settings.news_overrides)
        return f"watching every {int(interval)}s" + (f", push to '{topic}'" if topic else "")

    def export_csv(self):
        res = self.results
        if not res or not res.lineups:
            raise ValueError("no lineups yet - run first")
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            import dataclasses
            st = dataclasses.replace(self.settings, roster_format=self.run_format or self.settings.roster_format)
            self.opt.Exporter().export(res.pool, res.lineups, st, res.id_map, path)
            with open(path, "rb") as f:
                return f.read()
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

    # ---------------- addresses
    def addresses(self):
        out = []
        ts_ips = []
        # 1) ask the tailscale CLI (PATH first, then the usual Windows / Mac install spots)
        for exe in ("tailscale", r"C:\Program Files\Tailscale\tailscale.exe",
                    r"C:\Program Files (x86)\Tailscale\tailscale.exe",
                    "/Applications/Tailscale.app/Contents/MacOS/Tailscale"):
            try:
                r = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=4)
                ts_ips = [ip for ip in r.stdout.split() if ip.startswith("100.")]
                if ts_ips:
                    break
            except Exception:
                continue
        # 2) fall back to any local address in Tailscale's 100.64.0.0/10 range
        if not ts_ips:
            try:
                for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                    ip = info[4][0]
                    a, b = (int(x) for x in ip.split(".")[:2])
                    if a == 100 and 64 <= b <= 127 and ip not in ts_ips:
                        ts_ips.append(ip)
            except Exception:
                pass
        for ip in ts_ips:
            out.append(("Tailscale (use this one on the go)", ip))
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("10.255.255.255", 1))
            out.append(("home wifi", s.getsockname()[0]))
            s.close()
        except Exception:
            pass
        return out

    def url_hint(self):
        a = self.addresses()
        host = a[0][1] if a else "127.0.0.1"
        return f"http://{host}:{self.port}/?k={self.key}"


# ==================================================
# HTTP
# ==================================================

def make_handler(app):

    class H(BaseHTTPRequestHandler):

        def log_message(self, *a):
            pass

        # ---- helpers
        def _q(self):
            return urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

        def _authed(self):
            if self._q().get("k", [""])[0] == app.key:
                return True
            for part in (self.headers.get("Cookie") or "").split(";"):
                if part.strip() == f"dfs_k={app.key}":
                    return True
            return False

        def _send(self, body, ctype="application/json", code=200, cookie=False):
            b = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            if cookie:
                self.send_header("Set-Cookie", f"dfs_k={app.key}; Path=/; Max-Age=31536000; SameSite=Lax")
            self.end_headers()
            self.wfile.write(b)

        def _err(self, msg, code=400):
            self._send(jdump({"error": str(msg)}), code=code)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n > MAX_UPLOAD:
                raise ValueError("file too large")
            return self.rfile.read(n) if n else b""

        # ---- GET
        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path in ("/icon-192.png", "/icon-512.png", "/apple-touch-icon.png"):
                size = 512 if "512" in path else (180 if "apple" in path else 192)
                return self._send(make_png(size), "image/png")
            if path == "/sw.js":
                return self._send("self.addEventListener('fetch',()=>{});", "application/javascript")
            if not self._authed():
                return self._send("forbidden - open the address printed by web_app.py", "text/plain", 403)
            try:
                if path == "/":
                    return self._send(PAGE.replace("__KEY__", app.key), "text/html; charset=utf-8", cookie=True)
                if path == "/manifest.json":
                    return self._send(jdump(dict(
                        name="NBA DFS", short_name="DFS", display="standalone",
                        start_url=f"/?k={app.key}", background_color="#0e0e10", theme_color="#0e0e10",
                        icons=[dict(src="/icon-192.png", sizes="192x192", type="image/png"),
                               dict(src="/icon-512.png", sizes="512x512", type="image/png")])),
                        "application/manifest+json")
                routes = {"/api/state": app.state_json, "/api/lineups": app.lineups_json,
                          "/api/exposure": app.exposure_json, "/api/pool": app.pool_json,
                          "/api/news": app.news_json}
                if path in routes:
                    return self._send(jdump(routes[path]()))
                if path == "/api/export.csv":
                    data = app.export_csv()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv")
                    self.send_header("Content-Disposition", 'attachment; filename="fanduel_upload.csv"')
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                return self._err("not found", 404)
            except Exception as e:
                return self._err(e, 500)

        # ---- POST
        def do_POST(self):
            if not self._authed():
                return self._send("forbidden", "text/plain", 403)
            path = urllib.parse.urlparse(self.path).path
            try:
                raw = self._body()
                if path == "/api/upload":
                    q = self._q()
                    ok, msg = app.save_upload(q.get("kind", ["slate"])[0], q.get("name", ["upload.csv"])[0], raw)
                    return self._send(jdump(dict(ok=ok, message=msg)))
                d = json.loads(raw.decode("utf-8")) if raw else {}
                if path == "/api/settings":
                    errs = app.apply_settings(d)
                    return self._send(jdump(dict(ok=not errs, message="; ".join(errs) or "saved")))
                if path == "/api/select":
                    ok, msg = app.select_file(d.get("kind", "slate"), d.get("path", ""))
                    return self._send(jdump(dict(ok=ok, message=msg)))
                if path == "/api/run":
                    if d.get("settings"):
                        app.apply_settings(d["settings"])
                    ok, msg = app.start_run()
                    return self._send(jdump(dict(ok=ok, message=msg)))
                if path == "/api/stop":
                    return self._send(jdump(dict(ok=True, message=app.stop_run())))
                if path == "/api/override":
                    if d.get("text"):
                        msg = app.monitor.apply_manual_text(d["text"])
                        app.save_prefs()
                    else:
                        app.set_override(d["name"], d["prob"])
                        msg = f"{d['name']} set to {float(d['prob']):.0%}"
                    return self._send(jdump(dict(ok=True, message=msg)))
                if path == "/api/override/clear":
                    app.clear_override(d["name"])
                    return self._send(jdump(dict(ok=True, message="cleared")))
                if path == "/api/monitor":
                    return self._send(jdump(dict(ok=True, message=app.monitor_control(d))))
                if path == "/api/news/fetch":
                    return self._send(jdump(dict(ok=True, message=app.fetch_news())))
                return self._err("not found", 404)
            except Exception as e:
                return self._err(e, 500)

    return H


def serve(app, host="0.0.0.0"):
    httpd = ThreadingHTTPServer((host, app.port), make_handler(app))
    return httpd


# ==================================================
# THE PAGE
# ==================================================

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes"><meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#0e0e10"><title>NBA DFS</title>
<link rel="manifest" href="/manifest.json?k=__KEY__"><link rel="apple-touch-icon" href="/apple-touch-icon.png">
<style>
:root{--bg:#0e0e10;--card:#1a1a1e;--line:#2a2a30;--tx:#f2f2f5;--mu:#9a9aa5;--ac:#ff8a2b;--bad:#ff453a;--ok:#30d158;--warn:#ffb340}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px -apple-system,Segoe UI,Roboto,sans-serif;
padding-top:env(safe-area-inset-top);padding-bottom:calc(64px + env(safe-area-inset-bottom))}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 14px}
h1{font-size:16px;margin:0;display:flex;justify-content:space-between;align-items:center}
.sub{color:var(--mu);font-size:12px;margin-top:3px}
.bar{height:5px;background:var(--line);border-radius:3px;margin-top:6px;overflow:hidden}.bar i{display:block;height:100%;background:var(--ac);width:0}
#banner{display:none;background:var(--bad);color:#fff;padding:10px 14px;font-weight:600}
nav{position:fixed;bottom:0;left:0;right:0;background:#151518;border-top:1px solid var(--line);display:flex;padding-bottom:env(safe-area-inset-bottom);z-index:9}
nav button{flex:1;background:none;border:0;color:var(--mu);padding:11px 0 9px;font-size:12px}
nav button.on{color:var(--ac);font-weight:700}
.tab{display:none;padding:10px}.tab.on{display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;margin-bottom:10px}
.row{display:flex;gap:8px;align-items:center;justify-content:space-between;margin:6px 0}
.mu{color:var(--mu);font-size:12px}.b{font-weight:600}
button.p,button.s,button.d{border:0;border-radius:10px;padding:12px 14px;font-size:15px;font-weight:600;color:#fff}
button.p{background:var(--ac);color:#111}button.s{background:#34343c}button.d{background:var(--bad)}
.full{width:100%}.two{display:flex;gap:8px}.two>*{flex:1}
input,select{width:100%;background:#0b0b0d;color:var(--tx);border:1px solid var(--line);border-radius:9px;padding:10px;font-size:15px}
input[type=checkbox]{width:22px;height:22px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:7px 5px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--mu);font-weight:600;position:sticky;top:0;background:var(--bg)}
.scroll{overflow-x:auto}
.pl{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--line);font-size:14px}.pl:last-child{border:0}
.sl{color:var(--mu);width:38px;display:inline-block}.risk{color:var(--warn)}.neg{color:var(--bad)}.pos{color:var(--ok)}
pre{white-space:pre-wrap;font-size:11px;color:var(--mu);max-height:230px;overflow:auto;margin:0}
.al{border-left:4px solid var(--bad)}.al.up{border-left-color:var(--ok)}.al.dn{border-left-color:var(--warn)}
.fld{margin-bottom:9px}.fld label{display:block;margin-bottom:3px;color:var(--mu);font-size:12px}
.chk{display:flex;justify-content:space-between;align-items:center;margin:9px 0}
</style></head><body>
<header><h1><span>NBA DFS</span><span id="st" class="mu"></span></h1><div class="sub" id="sub"></div>
<div class="bar"><i id="pb"></i></div></header>
<div id="banner" onclick="seen();show('news')"></div>

<div class="tab on" id="t-run">
 <div class="card"><div class="b">Slate</div><div class="mu" id="slname">none chosen</div>
  <select id="slsel" onchange="pick('slate',this.value)"></select>
  <div class="two" style="margin-top:8px"><button class="s" onclick="$('upS').click()">Upload slate CSV</button>
  <button class="s" onclick="$('upT').click()">Upload FD player list</button></div>
  <input type="file" id="upS" accept=".csv" hidden onchange="up('slate',this)"><input type="file" id="upT" accept=".csv" hidden onchange="up('template',this)">
  <div class="mu" id="tpname" style="margin-top:6px"></div>
  <select id="tpsel" onchange="pick('template',this.value)" style="margin-top:6px"></select></div>
 <div class="two"><button class="p" id="runb" onclick="run()">Build lineups</button><button class="d" id="stopb" onclick="stop()" style="display:none">Stop</button></div>
 <div class="mu" id="runmsg" style="margin:8px 2px"></div>
 <div class="card"><div class="b" style="margin-bottom:8px">Settings</div><div id="form"></div>
  <button class="s full" onclick="saveSet()">Save settings</button></div>
 <div class="card"><div class="b" style="margin-bottom:6px">Log</div><pre id="log"></pre></div>
</div>

<div class="tab" id="t-lu"><div class="two" style="margin-bottom:10px">
 <button class="p" onclick="dl()">Download FanDuel CSV</button><button class="s" onclick="load('lu')">Refresh</button></div><div id="lulist"></div></div>

<div class="tab" id="t-ex"><div class="card scroll"><table id="extb"></table></div></div>

<div class="tab" id="t-pl"><input id="plq" placeholder="search player or team" oninput="rPool()" style="margin-bottom:8px">
 <div class="card scroll"><table id="pltb"></table></div></div>

<div class="tab" id="t-news">
 <div class="card"><div class="b">Late-news watch</div><div class="mu" id="mon" style="margin:4px 0 8px"></div>
  <div class="two"><div class="fld"><label>check every (sec)</label><input id="mint" type="number" value="90"></div>
  <div class="fld"><label>ntfy topic (phone push)</label><input id="mtop" placeholder="your-secret-topic"></div></div>
  <div class="two"><button class="s" onclick="mon('start')">Start / apply</button><button class="s" onclick="mon('stop')">Stop</button></div></div>
 <div class="card"><div class="b">Quick update</div><div class="mu">tatum out; brown q; mitchell 50; jokic in</div>
  <input id="qtx" placeholder="player + status" autocapitalize="none" style="margin:6px 0">
  <div class="two"><button class="s" onclick="quick()">Apply</button><button class="p" onclick="run()">Rebuild now</button></div>
  <div class="mu" id="qmsg" style="margin-top:6px"></div></div>
 <div id="alerts"></div>
 <div class="card"><div class="b">Your overrides</div><div id="ovr" class="mu">none</div></div>
 <div class="card"><div class="row"><div class="b">Injury report</div><button class="s" onclick="fetchNews()">Refresh</button></div>
  <div class="mu" id="nwinfo"></div><div id="inj"></div></div>
</div>

<nav><button id="n-run" class="on" onclick="show('run')">Build</button><button id="n-lu" onclick="show('lu')">Lineups</button>
<button id="n-ex" onclick="show('ex')">Exposure</button><button id="n-pl" onclick="show('pl')">Pool</button>
<button id="n-news" onclick="show('news')">News</button></nav>

<script>
const K="__KEY__";const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path,body,raw){const u=path+(path.includes('?')?'&':'?')+'k='+K;
 const o=body===undefined?{}:{method:'POST',body:raw?body:JSON.stringify(body)};
 const r=await fetch(u,o);const t=await r.text();let j={};try{j=JSON.parse(t)}catch(e){}
 if(!r.ok)throw new Error(j.error||t||r.status);return j}
let S=null,tab='run',lastRun=-1,formBuilt=false,poolRows=[],sortKey='proj',sortDir=-1,exRows=[];
const f0=(x,d=0)=>x==null?'-':Number(x).toFixed(d);
function show(t){tab=t;for(const x of['run','lu','ex','pl','news']){$('t-'+x).classList.toggle('on',x==t);$('n-'+x).classList.toggle('on',x==t)}
 window.scrollTo(0,0);load(t)}
function load(t){if(t=='lu')loadLu();else if(t=='ex')loadEx();else if(t=='pl')loadPl();else if(t=='news')loadNews()}
function seen(){localStorage.setItem('seen',String(S?S.now:Date.now()/1000));$('banner').style.display='none'}
async function poll(){try{S=await api('/api/state');render();}catch(e){$('st').textContent='offline'}
 setTimeout(poll,S&&S.running?1500:6000)}
function render(){const s=S;$('st').textContent=s.running?'running':'idle';
 const pct=s.total?Math.round(100*s.cur/s.total):(s.running?8:0);$('pb').style.width=(s.running?pct:0)+'%';
 $('sub').textContent=s.status+(s.monitor.running?' · watching news':'')+(s.monitor.eta?' · '+s.monitor.eta:'');
 $('runb').style.display=s.running?'none':'block';$('stopb').style.display=s.running?'block':'none';
 $('runmsg').textContent=s.error?('Error: '+s.error):(s.running?s.status:(s.n_lineups?s.n_lineups+' lineups ready':''));
 $('slname').textContent=s.slate||'none chosen';$('tpname').textContent=s.template?('FD player list: '+s.template):'FD player list: none (needed for CSV export)';
 fillSel('slsel',s.files,s.slate);fillSel('tpsel',s.files,s.template);
 if(!formBuilt){buildForm();formBuilt=true}
 $('log').textContent=s.log.join('\n');
 const seenT=parseFloat(localStorage.getItem('seen')||'0');const fresh=s.alerts.filter(a=>a.t>seenT);
 if(fresh.length){$('banner').style.display='block';$('banner').textContent='LATE NEWS: '+fresh.slice(0,3).map(a=>a.name+' '+lab(a.new)).join(', ')+' — tap'}
 if(s.run_id!==lastRun){lastRun=s.run_id;load(tab)}
 if(tab=='news')rAlerts()}
function fillSel(id,files,cur){const el=$(id);const v=files.map(f=>f.name+'|'+f.path).join('#');if(el.dataset.v===v+cur)return;el.dataset.v=v+cur;
 el.innerHTML='<option value="">choose a file...</option>'+files.map(f=>`<option value="${esc(f.path)}"${f.name==cur?' selected':''}>${esc(f.name)} (${esc(f.where)})</option>`).join('')}
function buildForm(){let h='';for(const f of S.schema){const v=S.settings[f.key];
 if(f.type=='bool')h+=`<div class="chk"><span>${esc(f.label)}</span><input type="checkbox" id="f_${f.key}"${v?' checked':''}></div>`;
 else if(f.type=='choice')h+=`<div class="fld"><label>${esc(f.label)}</label><select id="f_${f.key}">${f.extra.map(o=>`<option${o==v?' selected':''}>${esc(o)}</option>`).join('')}</select></div>`;
 else h+=`<div class="fld"><label>${esc(f.label)}</label><input id="f_${f.key}" type="number" step="any" value="${v}"></div>`}
 $('form').innerHTML=h}
function readForm(){const o={};for(const f of S.schema){const el=$('f_'+f.key);if(!el)continue;o[f.key]=f.type=='bool'?el.checked:el.value}return o}
async function saveSet(){const r=await api('/api/settings',readForm());$('runmsg').textContent=r.message}
async function pick(kind,path){if(!path)return;const r=await api('/api/select',{kind,path});$('runmsg').textContent=r.message;poll()}
async function up(kind,inp){const f=inp.files[0];if(!f)return;$('runmsg').textContent='uploading...';
 const r=await api('/api/upload?kind='+kind+'&name='+encodeURIComponent(f.name),await f.arrayBuffer(),true);
 $('runmsg').textContent=r.message;inp.value='';poll()}
async function run(){try{const r=await api('/api/run',{settings:readForm()});$('runmsg').textContent=r.message;show('run');poll()}catch(e){$('runmsg').textContent=e.message}}
async function stop(){const r=await api('/api/stop',{});$('runmsg').textContent=r.message}
async function dl(){try{const r=await fetch('/api/export.csv?k='+K);if(!r.ok){const j=await r.json().catch(()=>({}));alert(j.error||'export failed');return}
 const blob=await r.blob();const file=new File([blob],'fanduel_upload.csv',{type:'text/csv'});
 if(navigator.canShare&&navigator.canShare({files:[file]})){try{await navigator.share({files:[file],title:'FanDuel upload'});return}catch(e){if(e.name=='AbortError')return}}
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='fanduel_upload.csv';document.body.appendChild(a);a.click();a.remove()}catch(e){alert(e.message)}}
async function loadLu(){const d=await api('/api/lineups');const el=$('lulist');
 if(!d.lineups.length){el.innerHTML='<div class="card mu">No lineups yet — go to Build.</div>';return}
 el.innerHTML=d.lineups.map(l=>`<div class="card"><div class="row"><span class="b">#${l.n}</span><span class="mu">${f0(l.proj,1)} proj · $${Number(l.salary).toLocaleString()} · own ${f0(l.own)} · top1 ${f0(100*l.top1,1)}% · ROI ${l.roi>=0?'+':''}${f0(100*l.roi)}%</span></div>`+
 l.players.map(p=>`<div class="pl"><span><span class="sl">${esc(p.slot)}</span>${esc(p.name)} <span class="mu">${esc(p.team)}</span>${p.risk?' <span class="risk">⚠</span>':''}</span><span class="mu">$${f0(p.salary/1000,1)}k · ${f0(p.proj,1)} · ${f0(p.own)}%</span></div>`).join('')+'</div>').join('')}
async function loadEx(){const d=await api('/api/exposure');exRows=d.rows;
 $('extb').innerHTML='<tr><th>Player</th><th>Exp%</th><th>Own%</th><th>Lev</th><th>Proj</th><th>Sal</th></tr>'+d.rows.map(r=>`<tr><td>${esc(r.name)}${r.risk?' <span class="risk">⚠</span>':''} <span class="mu">${esc(r.team)}</span></td><td>${f0(r.exp)}</td><td>${f0(r.own)}</td><td class="${r.lev>=0?'pos':'neg'}">${r.lev>=0?'+':''}${f0(r.lev)}</td><td>${f0(r.proj,1)}</td><td>${f0(r.salary/1000,1)}k</td></tr>`).join('')||'<tr><td>No lineups yet</td></tr>'}
async function loadPl(){const d=await api('/api/pool');poolRows=d.rows;rPool()}
function sortBy(k){if(sortKey==k)sortDir=-sortDir;else{sortKey=k;sortDir=-1}rPool()}
function rPool(){const q=($('plq').value||'').toLowerCase();let rows=poolRows.filter(r=>!q||(r.name+' '+r.team).toLowerCase().includes(q));
 rows.sort((a,b)=>sortDir*((a[sortKey]>b[sortKey])-(a[sortKey]<b[sortKey])));
 const th=(k,t)=>`<th onclick="sortBy('${k}')">${t}</th>`;
 $('pltb').innerHTML='<tr>'+th('name','Player')+th('pos','Pos')+th('salary','Sal')+th('proj','Proj')+th('own','Own%')+th('value','Val')+th('play','Play')+'</tr>'+
 rows.map(r=>`<tr><td>${esc(r.name)} <span class="mu">${esc(r.team)}</span></td><td>${esc(r.pos)}</td><td>${f0(r.salary/1000,1)}k</td><td>${f0(r.proj,1)}</td><td>${f0(r.own)}</td><td>${f0(r.value,1)}</td><td class="${r.play<0.995?'risk':''}">${r.play<0.995?Math.round(r.play*100)+'%':''}</td></tr>`).join('')||'<tr><td>No slate loaded</td></tr>'}
function lab(p){return p<=0.1?'OUT':p>=0.95?'ACTIVE':Math.round(p*100)+'%'}
function rAlerts(){$('alerts').innerHTML=S.alerts.map(a=>`<div class="card al ${a.new>a.old?'up':(a.new>0.1?'dn':'')}"><div class="b">${esc(a.name)}: ${lab(a.old)} → ${lab(a.new)}</div><div class="mu">${a.lineups?a.lineups+' of '+S.n_lineups+' lineups · ':''}${Math.max(0,Math.round((S.now-a.t)/60))} min ago${a.comment?' · '+esc(a.comment):''}</div></div>`).join('')}
async function loadNews(){const d=await api('/api/news');rAlerts();seen();
 $('mon').textContent=d.monitor.running?'watching the feed (alerts here and by push)':'monitor is off';
 if(d.monitor.topic&&!$('mtop').value)$('mtop').value=d.monitor.topic;if(d.monitor.interval)$('mint').value=d.monitor.interval;
 const ov=Object.entries(d.overrides);$('ovr').innerHTML=ov.length?ov.map(([n,v])=>`${esc(n)}: ${Math.round(v*100)}% <a href="#" onclick="clr(${JSON.stringify(n).replace(/"/g,'&quot;')});return false" style="color:var(--bad)">remove</a>`).join('<br>'):'none';
 $('nwinfo').textContent=d.source+(d.age!=null?' · '+Math.round(d.age)+' min old':'')+' · tap a player to set play %';
 $('inj').innerHTML=d.injuries.map((r,i)=>`<div class="pl" onclick="setP(${i})"><span>${esc(r.name)} <span class="mu">${esc(r.team)}</span>${r.slate?' <span class="pos">●</span>':''}</span><span class="${r.prob<0.995?'risk':''}">${esc(r.status)} · ${Math.round(r.prob*100)}%</span></div>`).join('')||'<div class="mu">no data</div>';
 window._inj=d.injuries}
async function setP(i){const r=window._inj[i];const v=prompt('Play % for '+r.name+' (0-100)',Math.round(r.prob*100));if(v===null)return;
 await api('/api/override',{name:r.name,prob:parseFloat(v)/100});loadNews()}
async function clr(n){await api('/api/override/clear',{name:n});loadNews()}
async function quick(){const t=$('qtx').value.trim();if(!t)return;const r=await api('/api/override',{text:t});$('qmsg').textContent=r.message+' — tap Rebuild now to apply';$('qtx').value='';loadNews()}
async function mon(a){const r=await api('/api/monitor',{action:a,interval:$('mint').value,topic:$('mtop').value});$('qmsg').textContent=r.message;loadNews();poll()}
async function fetchNews(){$('nwinfo').textContent='fetching...';try{await api('/api/news/fetch',{})}catch(e){}loadNews()}
if('serviceWorker' in navigator&&window.isSecureContext)navigator.serviceWorker.register('/sw.js').catch(()=>{});
poll();
</script></body></html>"""


# ==================================================
# MAIN
# ==================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--engine", default="new_1", help="module name of the optimizer file")
    ap.add_argument("--no-monitor", action="store_true", help="don't auto-start the news watch")
    args = ap.parse_args()

    opt = importlib.import_module(args.engine)
    app = WebApp(opt, args.port)
    httpd = serve(app)

    print("\nNBA DFS web app is running.  Open on your phone:")
    addrs = app.addresses() or [("this PC", "127.0.0.1")]
    for label, ip in addrs:
        print(f"  {label:38s} http://{ip}:{args.port}/?k={app.key}")
    if not any(l.startswith("Tailscale") for l, _ in addrs):
        print("\n  Tailscale address not found. Open the Tailscale app on this PC, copy the 100.x.x.x")
        print(f"  address shown for this device, and use  http://100.x.x.x:{args.port}/?k={app.key}")
    print("\nThen use 'Add to Home Screen'. Keep this window open. Ctrl+C to quit.\n")

    if not args.no_monitor:
        try:
            app.monitor_control(dict(action="start", interval=app.prefs.get("interval", 90),
                                     topic=app.prefs.get("topic", "")))
        except Exception as e:
            print(f"(news watch not started: {e})")
    app.save_prefs()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.monitor.stop()
        httpd.server_close()


if __name__ == "__main__":
    main()
