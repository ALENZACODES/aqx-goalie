"""Download NHL regular season play by play from the public feed api-web.nhle.com
and keep only the fields the model needs. One gzipped JSON lines file per season."""
import gzip, json, sys, time, concurrent.futures as cf, urllib.request, urllib.error
from pathlib import Path

BASE = "https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"
KEEP = ("periodDescriptor", "timeInPeriod", "situationCode", "typeDescKey", "sortOrder")

def slim(d):
    plays = []
    for p in d.get("plays", []):
        det = p.get("details", {}) or {}
        plays.append({
            "per": p["periodDescriptor"]["number"],
            "ptype": p["periodDescriptor"].get("periodType"),
            "t": p.get("timeInPeriod"),
            "sit": p.get("situationCode"),
            "ev": p.get("typeDescKey"),
            "team": det.get("eventOwnerTeamId"),
            "zone": det.get("zoneCode"),
            "hs": det.get("homeScore"), "as": det.get("awayScore"),
            "ord": p.get("sortOrder"),
        })
    return {
        "id": d["id"], "season": d["season"], "type": d["gameType"], "date": d["gameDate"],
        "state": d.get("gameState"),
        "home": {"id": d["homeTeam"]["id"], "abbrev": d["homeTeam"]["abbrev"], "score": d["homeTeam"].get("score")},
        "away": {"id": d["awayTeam"]["id"], "abbrev": d["awayTeam"]["abbrev"], "score": d["awayTeam"].get("score")},
        "last": (d.get("gameOutcome") or {}).get("lastPeriodType"),
        "plays": plays,
    }

def get(gid, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(BASE.format(gid=gid), headers={"User-Agent": "pull-the-goalie-research"}), timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (i + 1))
        except Exception:
            time.sleep(2 * (i + 1))
    return None

def fetch_season(start_year, n_games=1312, workers=8):
    out = OUT / f"{start_year}{start_year+1}.jsonl.gz"
    if out.exists() and "--force" not in sys.argv:
        print("skip", out); return
    ids = [int(f"{start_year}02{n:04d}") for n in range(1, n_games + 1)]
    rows = []
    with cf.ThreadPoolExecutor(workers) as ex:
        for d in ex.map(get, ids):
            if d and d.get("gameState") in ("OFF", "FINAL") and d.get("plays"):
                rows.append(slim(d))
    rows.sort(key=lambda r: r["id"])
    with gzip.open(out, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(start_year, len(rows), "games ->", out)

if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:] if a.isdigit()] or [2021, 2022, 2023, 2024, 2025, 2026]
    for y in years:
        fetch_season(y)
