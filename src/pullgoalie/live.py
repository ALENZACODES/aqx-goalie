"""Live game reader and a tiny local JSON server.

    python -m pullgoalie.live --game 2026020150          # one answer, printed as JSON
    python -m pullgoalie.live --serve --port 8787         # JSON API for overlays and chat bots

Endpoints (all GET, CORS open, JSON):
    /today                         games on the NHL scoreboard right now
    /state?game=ID[&mode=win][&at=SECONDS]  score, clock, goalie status and the model's call for both teams
    /vote?game=ID&choice=pull|wait&user=NAME   record one viewer vote (one per user per game)
    /votes?game=ID                 vote tallies, for a stream overlay
The vote endpoints are the hook for a Twitch chat bot: it reads !pull or !wait from chat
and calls /vote, while docs/overlay.html shows the model next to the crowd."""
from __future__ import annotations
import argparse, json, threading, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from .model import recommend

ROOT = Path(__file__).resolve().parents[2]
API = "https://api-web.nhle.com/v1"

def load_rates(path: Path = ROOT / "docs" / "data" / "results.json", team: str | None = None):
    R = json.loads(path.read_text())
    rates = {tuple(int(x) for x in k.split(",")): tuple(v) for k, v in R["rates_js"].items()}
    if team:
        for t in R["teams"]:
            if t["team"] == team:
                for k, v in t["rates_override"].items():
                    rates[tuple(int(x) for x in k.split(","))] = tuple(v)
    return rates

def fetch(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pull-the-goalie"}), timeout=15) as r:
        return json.load(r)

def seconds_left_regulation(period: int, remaining: int) -> int:
    if period > 3:
        return 0
    return (3 - period) * 1200 + remaining

def _replay(d: dict, at: int):
    """Rewind a finished game to `at` seconds left in regulation (for demos and overlay tests)."""
    cut = 3600 - at
    def el(p):
        m, s = p["timeInPeriod"].split(":"); return (p["periodDescriptor"]["number"] - 1) * 1200 + int(m) * 60 + int(s)
    plays = [p for p in d.get("plays", []) if p["periodDescriptor"]["number"] <= 3 and el(p) <= cut]
    hs = as_ = 0
    for p in plays:
        if p.get("typeDescKey") == "goal":
            hs, as_ = p["details"].get("homeScore", hs), p["details"].get("awayScore", as_)
    per = min(3, cut // 1200 + 1)
    return plays, per, 1200 - (cut - (per - 1) * 1200), hs, as_

def game_state(game_id: int, mode: str = "points", feed: dict | None = None, at: int | None = None):
    """Score, clock, goalie status and the model's call for both teams.
    `at` replays a finished game as if `at` seconds were left in regulation."""
    d = feed or fetch(f"{API}/gamecenter/{game_id}/play-by-play")
    if at is not None:
        plays, per, rem, hs, as_ = _replay(d, int(at))
        d = {**d, "gameState": "REPLAY", "clock": {"secondsRemaining": rem, "timeRemaining": f"{rem // 60:02d}:{rem % 60:02d}"}}
    else:
        per = (d.get("periodDescriptor") or {}).get("number", 1)
        rem = (d.get("clock") or {}).get("secondsRemaining", 1200)
        plays = d.get("plays") or []
        hs, as_ = d["homeTeam"].get("score", 0), d["awayTeam"].get("score", 0)
    left = seconds_left_regulation(per, rem)
    sit = plays[-1].get("situationCode", "1551") if plays else "1551"
    out = {"game": game_id, "state": d.get("gameState"), "period": per, "clock": (d.get("clock") or {}).get("timeRemaining"),
           "seconds_left_regulation": left, "situation": sit, "teams": []}
    for side, me, opp, gidx in (("home", hs, as_, 3), ("away", as_, hs, 0)):
        t = d[f"{side}Team"]
        abbrev = t["abbrev"]
        rec = recommend(load_rates(team=abbrev), left, me - opp, mode) if per <= 3 else {"action": "STAY", "reason": "overtime"}
        out["teams"].append({"side": side, "team": abbrev, "score": me, "goalie_in": sit[gidx] == "1",
                             "deficit": me - opp, "call": rec})
    return out

VOTES: dict = {}
LOCK = threading.Lock()

class Handler(BaseHTTPRequestHandler):
    mode = "points"
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def log_message(self, *a):
        pass
    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/today":
                s = fetch(f"{API}/score/now")
                return self._send(200, [{"id": g["id"], "away": g["awayTeam"]["abbrev"], "home": g["homeTeam"]["abbrev"],
                                         "state": g.get("gameState")} for g in s.get("games", [])])
            if u.path == "/state":
                return self._send(200, game_state(int(q["game"]), q.get("mode", self.mode), at=int(q["at"]) if "at" in q else None))
            if u.path == "/vote":
                g, c, user = q["game"], q["choice"].lower(), q.get("user", self.client_address[0])
                if c not in ("pull", "wait"):
                    return self._send(400, {"error": "choice must be pull or wait"})
                with LOCK:
                    VOTES.setdefault(g, {})[user] = c
                return self._send(200, {"ok": True})
            if u.path == "/votes":
                with LOCK:
                    v = list(VOTES.get(q["game"], {}).values())
                return self._send(200, {"pull": v.count("pull"), "wait": v.count("wait")})
            return self._send(404, {"error": "unknown path"})
        except Exception as e:  # keep the overlay alive on a bad request
            return self._send(500, {"error": str(e)})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int); ap.add_argument("--mode", default="points", choices=["points", "win"])
    ap.add_argument("--at", type=int, help="replay a finished game at this many seconds left in regulation")
    ap.add_argument("--serve", action="store_true"); ap.add_argument("--port", type=int, default=8787)
    a = ap.parse_args()
    if a.serve:
        Handler.mode = a.mode
        print(f"Serving on http://localhost:{a.port}  (try /today)")
        ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()
    elif a.game:
        print(json.dumps(game_state(a.game, a.mode, at=a.at), indent=1))
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
