"""
news_monitor.py - late-breaking news watchdog + phone access
============================================================

Runs in the background while the optimizer is open:

  * polls the injury feed every N seconds (min 30) and diffs it against the news
    your LAST lineup build used
  * for every status change on a slate player: banner in the app, beep, and a
    PUSH NOTIFICATION to your phone (free ntfy.sh - install the ntfy app, subscribe
    to your topic, nothing else to sign up for)
  * says how many of your built lineups contain the player and how long until
    the first tip
  * phone page: open http://<pc-ip>:8765/?k=<key> on the phone (same wifi, or any
    network via Tailscale) to see alerts, type "Tatum out" / "Brown 50", and hit
    REBUILD - the PC does the work
  * manual updates from Twitter/beat writers ("tatum out", "brown q", "mitchell 50%")
    beat the feed immediately

Everything fails soft; a network error never stops the optimizer.
"""

import difflib
import json
import re
import secrets
import socket
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nba_news import name_key, status_to_prob

MIN_INTERVAL = 30
CHANGE_THRESHOLD = 0.15

STATUS_WORDS = {
    "out": 0.0, "o": 0.0, "ruled out": 0.0, "inactive": 0.0, "dnp": 0.0, "scratched": 0.0,
    "doubtful": 0.08, "d": 0.08,
    "questionable": 0.5, "q": 0.5, "gtd": 0.55, "dtd": 0.6, "day-to-day": 0.6,
    "probable": 0.9, "p": 0.9,
    "in": 1.0, "active": 1.0, "plays": 1.0, "play": 1.0, "available": 1.0, "back": 1.0,
    "cleared": 1.0, "starting": 1.0, "starts": 1.0,
}


# ==================================================
# DIFFING
# ==================================================

def label_for(prob):
    if prob <= 0.1:
        return "OUT"
    if prob >= 0.95:
        return "ACTIVE"
    return f"{prob:.0%} to play"


def diff_injuries(baseline, injuries, slate_keys, overrides=None, threshold=CHANGE_THRESHOLD):
    """baseline: key -> prob used by the last build. injuries: key -> info dict.
    Returns a list of change dicts for players on the slate."""
    overrides = {name_key(k) for k in (overrides or {})}
    changes = []
    for k, info in injuries.items():
        if k in overrides or k not in slate_keys:
            continue
        old = baseline.get(k, 1.0)
        new = info["prob"]
        if abs(new - old) >= threshold:
            changes.append(dict(key=k, name=info["name"], team=info.get("team", ""),
                                old=old, new=new, status=info["status"],
                                comment=info.get("comment", ""), kind=_kind(old, new)))
    for k, old in baseline.items():                      # dropped off the report
        if k in injuries or k in overrides or old >= 0.95:
            continue
        if k not in slate_keys:
            continue
        changes.append(dict(key=k, name=k.title(), team="", old=old, new=1.0,
                            status="no longer listed", comment="removed from the injury report",
                            kind="CLEARED"))
    changes.sort(key=lambda c: (c["new"], c["name"]))
    return changes


def _kind(old, new):
    if new <= 0.1 < old:
        return "NOW OUT"
    if old <= 0.1 < new:
        return "BACK IN"
    return "UPGRADE" if new > old else "DOWNGRADE"


def first_tip(games, now=None):
    """Seconds until the earliest not-yet-started game, or None."""
    now = now or time.time()
    best = None
    for g in games or []:
        s = g.get("start")
        if not s:
            continue
        try:
            t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            t = t.timestamp()
        except Exception:
            continue
        if t > now and (best is None or t < best):
            best = t
    return None if best is None else best - now


def fmt_eta(sec):
    if sec is None:
        return "no upcoming tip found"
    m = int(sec // 60)
    return f"{m // 60}h {m % 60:02d}m to first tip" if m >= 60 else f"{m} min to first tip"


# ==================================================
# MANUAL UPDATES  ("tatum out", "brown q", "mitchell 50%")
# ==================================================

_STATUS_RE = "|".join(sorted((re.escape(w) for w in STATUS_WORDS), key=len, reverse=True))
_TAIL = re.compile(rf"^(?P<n>.+?)\s*[:\-]?\s+(?:is\s+|are\s+|will\s+be\s+|now\s+)?"
                   rf"(?P<s>{_STATUS_RE}|\d{{1,3}}\s*%?)\s*$")
_HEAD = re.compile(rf"^(?P<s>{_STATUS_RE})\s*[:\-]?\s+(?P<n>.+)$")


def parse_manual(text):
    """-> list of (name_query, prob). Unparseable lines come back as (line, None)."""
    out = []
    for line in re.split(r"[\n;]+", str(text or "")):
        line = line.strip().strip(".")
        if not line:
            continue
        low = line.lower()
        m = _TAIL.match(low) or _HEAD.match(low)
        if not m:
            out.append((line, None))
            continue
        s = m.group("s").strip()
        if s in STATUS_WORDS:
            prob = STATUS_WORDS[s]
        else:
            prob = min(max(float(s.rstrip("% ")) / 100.0, 0.0), 1.0)
        out.append((m.group("n").strip(), prob))
    return out


def resolve_name(query, candidates):
    """candidates: key -> display name. Returns (display_name, None) or (None, error)."""
    q = name_key(query)
    if not q:
        return None, "empty name"
    if q in candidates:
        return candidates[q], None
    toks = q.split()
    hits = [k for k in candidates if all(t in k.split() for t in toks)]
    if len(hits) == 1:
        return candidates[hits[0]], None
    if not hits:
        hits = [k for k in candidates if q in k]
        if len(hits) == 1:
            return candidates[hits[0]], None
    if len(hits) > 1:
        return None, f"'{query}' matches " + ", ".join(candidates[k] for k in hits[:5])
    if len(toks) == 2 and len(toks[0]) == 1:                       # "a davis"
        hits = [k for k in candidates
                if k.split()[0].startswith(toks[0]) and k.split()[-1] == toks[1]]
        if len(hits) == 1:
            return candidates[hits[0]], None
    # typo: compare against full names and against last names
    close = set(difflib.get_close_matches(q, list(candidates), n=3, cutoff=0.72))
    if len(toks) == 1:
        lasts = {}
        for k in candidates:
            lasts.setdefault(k.split()[-1], []).append(k)
        for ln in difflib.get_close_matches(q, list(lasts), n=3, cutoff=0.75):
            close.update(lasts[ln])
    close = sorted(close)
    if len(close) == 1:
        return candidates[close[0]], None
    if close:
        return None, f"'{query}' - did you mean " + ", ".join(candidates[k] for k in close[:4]) + "?"
    return None, f"no player matching '{query}'"


# ==================================================
# NOTIFICATIONS
# ==================================================

class Notifier:

    def __init__(self, topic="", click_url="", log=print):
        self.topic = re.sub(r"[^A-Za-z0-9_-]", "", topic or "")
        self.click_url = click_url
        self.log = log

    def beep(self):
        try:
            import winsound
            for _ in range(3):
                winsound.Beep(1200, 220)
        except Exception:
            try:
                print("\a", end="", flush=True)
            except Exception:
                pass

    def push(self, title, body, urgent=False):
        if not self.topic:
            return False
        try:
            h = {"Title": title.encode("ascii", "ignore").decode()[:120] or "NBA news",
                 "Priority": "5" if urgent else "4",
                 "Tags": "rotating_light" if urgent else "basketball"}
            if self.click_url:
                h["Click"] = self.click_url
            req = urllib.request.Request(f"https://ntfy.sh/{self.topic}",
                                         data=body.encode("utf-8"), headers=h, method="POST")
            urllib.request.urlopen(req, timeout=8).read()
            return True
        except Exception as e:
            self.log(f"Monitor: push notification failed ({e})")
            return False


# ==================================================
# SHARED STATE
# ==================================================

class MonitorState:

    def __init__(self):
        self.lock = threading.Lock()
        self.alerts = []                 # newest first
        self.last_check = 0.0
        self.source = "-"
        self.error = ""
        self.baseline_time = 0.0
        self.games = []
        self.running = False
        self.messages = []               # recent manual-update / action messages

    def add_alerts(self, items):
        with self.lock:
            self.alerts = (items + self.alerts)[:60]

    def note(self, msg):
        with self.lock:
            self.messages = ([dict(t=time.time(), msg=msg)] + self.messages)[:12]

    def snapshot(self, count_fn=None, total_lineups=0, overrides=None):
        with self.lock:
            alerts = []
            for a in self.alerts:
                d = dict(a)
                d["lineups"] = count_fn(a["key"]) if count_fn else 0
                alerts.append(d)
            return dict(
                now=time.time(), running=self.running, last_check=self.last_check,
                source=self.source, error=self.error, baseline_time=self.baseline_time,
                eta=fmt_eta(first_tip(self.games)), total_lineups=total_lineups,
                alerts=alerts, messages=list(self.messages),
                overrides={k: v for k, v in (overrides or {}).items()},
            )


# ==================================================
# THE WATCHDOG
# ==================================================

class NewsMonitor(threading.Thread):

    def __init__(self, feed, state, interval, notifier, slate_keys, overrides,
                 lineup_count, on_alerts=None, log=print):
        super().__init__(daemon=True)
        self.feed, self.state, self.notifier = feed, state, notifier
        self.interval = max(MIN_INTERVAL, float(interval))
        self.slate_keys, self.overrides, self.lineup_count = slate_keys, overrides, lineup_count
        self.on_alerts, self.log = on_alerts, log
        self._stop_evt = threading.Event()
        self.baseline = None             # key -> prob used by the last build
        self.seen = set()

    def set_baseline(self, injuries, overrides=None):
        ov = {name_key(k): v for k, v in (overrides or {}).items()}
        base = {k: ov.get(k, i["prob"]) for k, i in injuries.items()}
        for k, v in ov.items():
            base[k] = v
        self.baseline = base
        self.state.baseline_time = time.time()
        self.seen.clear()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        self.state.running = True
        self.log(f"Monitor: watching the injury feed every {int(self.interval)}s")
        while not self._stop_evt.is_set():
            try:
                self.check_once()
            except Exception as e:                       # never die
                self.state.error = str(e)
            self._stop_evt.wait(self.interval)
        self.state.running = False
        self.log("Monitor: stopped")

    def check_once(self):
        snap = self.feed.get(force=True)
        st = self.state
        st.last_check = time.time()
        st.source = snap.source
        st.games = snap.games or st.games
        if snap.source != "live":
            st.error = "feed unreachable - showing last known news"
            return []
        st.error = ""
        if self.baseline is None:                        # no build yet: watch from now on
            self.set_baseline(snap.injuries, self.overrides())
            return []
        changes = diff_injuries(self.baseline, snap.injuries, self.slate_keys(), self.overrides())
        fresh = []
        for c in changes:
            sig = (c["key"], round(c["new"], 2))
            if sig in self.seen:
                continue
            self.seen.add(sig)
            c["t"] = time.time()
            c["id"] = secrets.token_hex(4)
            fresh.append(c)
        if fresh:
            st.add_alerts(fresh)
            self._notify(fresh, snap)
            if self.on_alerts:
                self.on_alerts(fresh)
        return fresh

    def _notify(self, fresh, snap):
        eta = fmt_eta(first_tip(snap.games))
        urgent = any(c["kind"] in ("NOW OUT", "BACK IN") for c in fresh)
        lines = []
        for c in fresh[:6]:
            n = self.lineup_count(c["key"])
            lines.append(f"{c['name']}: {label_for(c['old'])} -> {label_for(c['new'])}"
                         + (f" ({n} of your lineups)" if n else ""))
        title = ("LATE NEWS: " if urgent else "News: ") + ", ".join(c["name"] for c in fresh[:3])
        body = "\n".join(lines) + f"\n{eta}"
        self.log("Monitor ALERT - " + "; ".join(lines) + f" | {eta}")
        self.notifier.beep()
        self.notifier.push(title, body, urgent)


# ==================================================
# PHONE PAGE
# ==================================================

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DFS watch</title><style>
body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;background:#111;color:#eee}
header{padding:12px 14px;background:#1c1c1e;position:sticky;top:0}
h1{font-size:17px;margin:0}.sub{font-size:13px;color:#aaa;margin-top:3px}
.card{margin:10px;padding:12px;border-radius:10px;background:#1c1c1e}
.out{border-left:5px solid #ff453a}.down{border-left:5px solid #ff9f0a}.up{border-left:5px solid #30d158}
.n{font-weight:600}.m{font-size:13px;color:#bbb;margin-top:4px}
input[type=text]{width:100%;box-sizing:border-box;padding:12px;font-size:16px;border-radius:8px;border:1px solid #444;background:#000;color:#fff}
button{width:100%;padding:13px;margin-top:8px;font-size:16px;border:0;border-radius:8px;background:#0a84ff;color:#fff}
button.r{background:#ff453a}.bad{color:#ff9f0a}
</style></head><body>
<header><h1>NBA DFS watch</h1><div class="sub" id="st">loading...</div></header>
<div id="al"></div>
<div class="card"><div class="n">Update a player</div>
<div class="m">e.g. <i>tatum out</i> &nbsp; <i>brown q</i> &nbsp; <i>mitchell 50</i> &nbsp; <i>jokic in</i></div>
<input type="text" id="tx" placeholder="player + status" autocapitalize="none">
<button onclick="send('/override')">Apply update</button>
<div class="m" id="msg"></div></div>
<div class="card"><button class="r" onclick="if(confirm('Rebuild all lineups now?'))send('/rebuild')">Rebuild lineups now</button></div>
<div class="card"><div class="n">Active overrides</div><div class="m" id="ov">none</div></div>
<script>
const K=new URLSearchParams(location.search).get('k')||'';
function ago(t){const s=Math.max(0,Math.round(Date.now()/1000-t));return s<90?s+'s ago':Math.round(s/60)+' min ago'}
function lab(p){return p<=0.1?'OUT':p>=0.95?'ACTIVE':Math.round(p*100)+'% to play'}
async function load(){try{
 const r=await fetch('/state.json?k='+K);const s=await r.json();
 document.getElementById('st').innerHTML=(s.running?'monitor on':'monitor OFF')+' · checked '+ago(s.last_check)+' · '+s.eta+(s.error?'<br><span class="bad">'+s.error+'</span>':'');
 let h='';for(const a of s.alerts){const c=a.new<=0.1?'out':(a.new<a.old?'down':'up');
  h+='<div class="card '+c+'"><div class="n">'+a.name+' · '+lab(a.old)+' → '+lab(a.new)+'</div>'
   +'<div class="m">'+(a.lineups?a.lineups+' of '+s.total_lineups+' lineups affected · ':'')+ago(a.t)+'</div>'
   +(a.comment?'<div class="m">'+a.comment+'</div>':'')+'</div>'}
 document.getElementById('al').innerHTML=h||'<div class="card"><div class="m">No changes since your last build.</div></div>';
 const o=Object.entries(s.overrides).map(([k,v])=>k+': '+Math.round(v*100)+'%').join('<br>');
 document.getElementById('ov').innerHTML=o||'none';
 document.getElementById('msg').innerHTML=s.messages.slice(0,3).map(m=>m.msg).join('<br>');
}catch(e){document.getElementById('st').innerHTML='<span class="bad">cannot reach the PC</span>'}}
async function send(path){const b=new URLSearchParams();b.set('text',document.getElementById('tx').value);
 const r=await fetch(path+'?k='+K,{method:'POST',body:b});const j=await r.json();
 document.getElementById('msg').innerHTML=j.message;if(path=='/override')document.getElementById('tx').value='';load()}
load();setInterval(load,10000);
</script></body></html>"""


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class PhoneServer:

    def __init__(self, controller, port=8765):
        self.controller = controller
        self.port = int(port)
        self.key = secrets.token_urlsafe(9)
        self.httpd = None

    @property
    def url(self):
        return f"http://{lan_ip()}:{self.port}/?k={self.key}"

    def start(self):
        ctl, key = self.controller, self.key

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _ok(self):
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                if q.get("k", [""])[0] != key:
                    self.send_response(403)
                    self.end_headers()
                    self.wfile.write(b"forbidden")
                    return False
                return True

            def _send(self, body, ctype):
                b = body.encode("utf-8") if isinstance(body, str) else body
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(b)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if not self._ok():
                    return
                path = urllib.parse.urlparse(self.path).path
                if path == "/state.json":
                    self._send(json.dumps(ctl.state_json()), "application/json")
                else:
                    self._send(PAGE, "text/html; charset=utf-8")

            def do_POST(self):
                if not self._ok():
                    return
                n = int(self.headers.get("Content-Length", 0) or 0)
                form = urllib.parse.parse_qs(self.rfile.read(min(n, 4096)).decode("utf-8", "ignore"))
                text = form.get("text", [""])[0]
                path = urllib.parse.urlparse(self.path).path
                if path == "/override":
                    msg = ctl.apply_manual_text(text)
                elif path == "/rebuild":
                    msg = ctl.request_rebuild()
                else:
                    msg = "unknown action"
                self._send(json.dumps({"message": msg}), "application/json")

        self.httpd = ThreadingHTTPServer(("0.0.0.0", self.port), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None


# ==================================================
# CONTROLLER (what the GUI talks to - no Tk in here)
# ==================================================

class MonitorController:
    """
    slate_names()   -> {name_key: display name} of every player on the slate
    lineup_counts() -> ({name_key: n lineups containing him}, total lineups)
    post(kind, payload) -> hands events to the GUI thread
    """

    def __init__(self, settings, slate_names, lineup_counts, post, log=print):
        self.settings = settings
        self.slate_names = slate_names
        self.lineup_counts = lineup_counts
        self.post = post
        self.log = log
        self.state = MonitorState()
        self.monitor = None
        self.server = None
        self.known = {}                  # news-feed names too (for manual matching)

    # ---- lifecycle
    def start(self, interval=90, ntfy_topic="", phone=False, port=8765, click_url=""):
        from nba_news import NewsFeed
        self.stop()
        self.server = None
        if phone:
            try:
                self.server = PhoneServer(self, port)
                self.server.start()
                self.log(f"Monitor: phone page at {self.server.url}")
            except Exception as e:
                self.log(f"Monitor: phone page failed to start ({e})")
                self.server = None
        click = self.server.url if self.server else click_url
        notifier = Notifier(ntfy_topic, click, self.log)
        self.monitor = NewsMonitor(
            NewsFeed(log=lambda *_: None), self.state, interval, notifier,
            slate_keys=lambda: set(self.slate_names()),
            overrides=lambda: self.settings.news_overrides,
            lineup_count=lambda k: self.lineup_counts()[0].get(k, 0),
            on_alerts=lambda fresh: self.post("alert", fresh), log=self.log)
        self.monitor.start()
        if notifier.topic:
            notifier.push("DFS watch started", "You will get late-news alerts here.")
        return self.server.url if self.server else ""

    def stop(self):
        if self.monitor:
            self.monitor.stop()
            self.monitor = None
        if self.server:
            self.server.stop()
            self.server = None
        self.state.running = False

    @property
    def phone_url(self):
        return self.server.url if self.server else ""

    def set_baseline(self, snap, overrides=None):
        """Call after each build with the news snapshot that build used."""
        if self.monitor and snap is not None:
            self.monitor.set_baseline(snap.injuries, overrides)
            self.state.games = snap.games or self.state.games
        if snap is not None:
            for k, i in snap.injuries.items():
                self.known[k] = i["name"]

    # ---- manual updates (GUI box and phone both land here)
    def apply_manual_text(self, text):
        cands = dict(self.known)
        cands.update(self.slate_names())
        msgs, changed = [], False
        for query, prob in parse_manual(text):
            if prob is None:
                msgs.append(f"couldn't read '{query}' (try: tatum out / brown q / mitchell 50)")
                continue
            name, err = resolve_name(query, cands)
            if err:
                msgs.append(err)
                continue
            self.settings.news_overrides[name] = prob
            changed = True
            n = self.lineup_counts()[0].get(name_key(name), 0)
            msgs.append(f"{name} set to {label_for(prob)}" + (f" - in {n} of your lineups" if n else ""))
        msg = "; ".join(msgs) or "nothing to apply"
        self.state.note(msg)
        self.log("Monitor: manual update - " + msg)
        if changed:
            self.post("override_changed", None)
        return msg

    def request_rebuild(self):
        self.post("rebuild", None)
        msg = "rebuild requested on the PC"
        self.state.note(msg)
        return msg

    def state_json(self):
        counts, total = self.lineup_counts()
        return self.state.snapshot(lambda k: counts.get(k, 0), total, self.settings.news_overrides)
