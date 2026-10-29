"""Build rate tables, pull events and backtest states from parsed seasons."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import bisect
import numpy as np, pandas as pd
from .parse import load_season, game_segments, first_pulls, REG_END
from .model import rate_key

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
CHECKPOINTS = tuple(range(10, 601, 10))

def strategic_pull_after(segs, t_elapsed):
    for s in segs:
        if s.t1 > s.t0 and s.t0 >= t_elapsed - 1e-9 and s.own_goalie == 0 and s.diff < 0 and not s.delayed_penalty and s.opp_goalie == 1:
            return REG_END - s.t0
    return 0.0

def state_at(segs, t_elapsed):
    for s in segs:
        if s.t1 > s.t0 and s.t0 <= t_elapsed < s.t1:
            return s
    return None

def season_tables(path: Path):
    games = list(load_season(path))
    exp_rows, pulls, states = [], [], []
    for g in games:
        if g["type"] != 2:
            continue
        segs = game_segments(g)
        went_ot = g.get("last") in ("OT", "SO")
        for p in first_pulls(g, segs):
            pulls.append(p.__dict__)
        for tid, ss in segs.items():
            home = tid == g["home"]["id"]
            me, opp = (g["home"], g["away"]) if home else (g["away"], g["home"])
            pts = 2 if me["score"] > opp["score"] else (1 if went_ot else 0)
            acc = defaultdict(lambda: [0.0, 0, 0])
            for s in ss:
                if s.t1 < 2400 or s.delayed_penalty:
                    continue
                if s.opp_goalie != 1:
                    continue
                pulled = 1 - s.own_goalie
                even = s.opp_sk == 5 and s.own_sk == 5 + pulled
                a = acc[(rate_key(s.diff), pulled, s.t0 >= 3000, even)]
                a[0] += s.t1 - s.t0; a[1] += s.gf; a[2] += s.ga
            for (d, pulled, late, even), (sec, gf, ga) in acc.items():
                exp_rows.append((g["season"], g["id"], me["abbrev"], d, pulled, late, even, sec, gf, ga))
            late = [x for x in ss if x.t1 > x.t0 and x.t1 > REG_END - 610]
            starts = [x.t0 for x in late]
            for c in CHECKPOINTS:
                te = REG_END - c
                i = bisect.bisect_right(starts, te) - 1
                if i < 0:
                    continue
                s = late[i]
                if not (s.t0 <= te < s.t1) or s.diff >= 0 or s.diff < -3 or s.own_goalie == 0:
                    continue
                pr = strategic_pull_after(late[i:], te)
                pull_el = REG_END - pr if pr > 0 else REG_END
                caught = any(x.diff >= 0 and te <= x.t0 < pull_el for x in late[i:])
                states.append(dict(season=g["season"], game=g["id"], date=g["date"], team=me["abbrev"], opp=opp["abbrev"],
                                   checkpoint=c, diff=s.diff, even=(s.own_sk == 5 and s.opp_sk == 5),
                                   pull_remaining=pr, caught_up_first=caught, points=pts,
                                   reached_ot_or_won=int(pts >= 1), won=int(pts == 2)))
    exp = pd.DataFrame(exp_rows, columns=["season", "game", "team", "d", "pulled", "late", "even", "sec", "gf", "ga"])
    return exp, pd.DataFrame(pulls), pd.DataFrame(states)

def load_all(seasons=None):
    files = sorted(RAW.glob("*.jsonl.gz"))
    E, P, S = [], [], []
    for f in files:
        if seasons and int(f.name[:8]) not in seasons:
            continue
        e, p, s = season_tables(f)
        E.append(e); P.append(p); S.append(s)
    return pd.concat(E, ignore_index=True), pd.concat(P, ignore_index=True), pd.concat(S, ignore_index=True)

def rate_table(exp: pd.DataFrame, late_only_trailing=True, strengths="even"):
    """Goals per 60 minutes with exposure, by (d, pulled), third period only.
    strengths="even": 5 on 5, or 6 on 5 with the goalie pulled (the decision model).
    strengths="all": every manpower state, so penalties are folded in at their real frequency.
    When trailing with the goalie in we use the last 10 minutes (score effects grow late)."""
    if strengths == "even":
        exp = exp[exp.even.astype(bool)]
    rows = []
    for (d, pulled), g in exp.groupby(["d", "pulled"]):
        if pulled == 0 and d < 0 and late_only_trailing:
            g = g[g.late]
        sec, gf, ga = g.sec.sum(), g.gf.sum(), g.ga.sum()
        rows.append(dict(d=int(d), pulled=int(pulled), hours=sec / 3600, gf=int(gf), ga=int(ga),
                         gf60=gf / sec * 3600 if sec else np.nan, ga60=ga / sec * 3600 if sec else np.nan))
    return pd.DataFrame(rows)

def rates_dict(rt: pd.DataFrame):
    r = {}
    for _, x in rt.iterrows():
        r[(int(x.d), int(x.pulled))] = (float(x.gf60), float(x.ga60))
    for d in (0, 1, 2):
        r.setdefault((d, 0), r[(0, 0)])
    return r

def posterior_draws(rt: pd.DataFrame, n=1000, seed=7):
    """Gamma(goals + 0.5, hours) posterior draws of every rate (Jeffreys prior)."""
    rng = np.random.default_rng(seed)
    draws = [dict() for _ in range(n)]
    for _, x in rt.iterrows():
        f = rng.gamma(x.gf + 0.5, 1 / x.hours, n); a = rng.gamma(x.ga + 0.5, 1 / x.hours, n)
        for i in range(n):
            draws[i][(int(x.d), int(x.pulled))] = (f[i], a[i])
    return draws
