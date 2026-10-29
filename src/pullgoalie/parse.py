"""Turn slim NHL play by play into time segments and goalie pull events.

situationCode is four digits: away goalie, away skaters, home skaters, home goalie.
"1551" is five on five with both goalies in, "0651" means the away team pulled its goalie.
All times are seconds of regulation elapsed (0 to 3600)."""
from __future__ import annotations
import gzip, json
from dataclasses import dataclass
from pathlib import Path

REG_END = 3600

def load_season(path: Path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)

def clock(per: int, t: str) -> int:
    m, s = t.split(":")
    return (per - 1) * 1200 + int(m) * 60 + int(s)

def parse_sit(sit: str | None):
    """Return (away_goalie, away_sk, home_sk, home_goalie) or None."""
    if not sit or len(sit) != 4 or not sit.isdigit():
        return None
    return int(sit[0]), int(sit[1]), int(sit[2]), int(sit[3])

@dataclass
class Segment:
    """A stretch of regulation time from one team's point of view."""
    game: int
    team: int          # team id whose view this is
    is_home: bool
    t0: float          # seconds elapsed at start
    t1: float          # seconds elapsed at end
    diff: int          # team score minus opponent score
    own_goalie: int    # 1 in net, 0 pulled
    opp_goalie: int
    own_sk: int
    opp_sk: int
    gf: int = 0        # goals for scored at the end of this segment
    ga: int = 0
    delayed_penalty: bool = False

def regulation_plays(g):
    ps = [p for p in g["plays"] if p.get("ptype") == "REG" and p["per"] <= 3 and p.get("t") and parse_sit(p.get("sit"))]
    ps.sort(key=lambda p: p["ord"])
    return ps

def game_segments(g):
    """Yield one list of segments for home and one for away.
    The gap between two plays is split at the midpoint when the situation changes."""
    plays = regulation_plays(g)
    hid, aid = g["home"]["id"], g["away"]["id"]
    out = {hid: [], aid: []}
    hs = as_ = 0
    prev_t, prev_sit = 0, "1551"
    for i, p in enumerate(plays):
        t = clock(p["per"], p["t"])
        sit = p["sit"]
        pieces = []
        if t > prev_t:
            if sit == prev_sit:
                pieces.append((prev_t, t, sit))
            else:
                mid = (prev_t + t) / 2
                pieces += [(prev_t, mid, prev_sit), (mid, t, sit)]
        is_goal = p["ev"] == "goal"
        scorer = p.get("team")
        # delayed penalty: a goalie out stretch that ends with a penalty on the other team
        for (a, b, s) in pieces:
            ag, ask, hsk, hg = parse_sit(s)
            out[hid].append(Segment(g["id"], hid, True, a, b, hs - as_, hg, ag, hsk, ask))
            out[aid].append(Segment(g["id"], aid, False, a, b, as_ - hs, ag, hg, ask, hsk))
        if is_goal:
            ag, ask, hsk, hg = parse_sit(sit)
            # zero length marker segment carrying the goal, in the state it was scored
            for tid, home in ((hid, True), (aid, False)):
                own_g, opp_g = (hg, ag) if home else (ag, hg)
                own_sk, opp_sk = (hsk, ask) if home else (ask, hsk)
                d = hs - as_ if home else as_ - hs
                seg = Segment(g["id"], tid, home, t, t, d, own_g, opp_g, own_sk, opp_sk)
                seg.gf = int(scorer == tid)
                seg.ga = int(scorer is not None and scorer != tid)
                out[tid].append(seg)
            if p.get("hs") is not None:
                hs, as_ = p["hs"], p["as"]
            elif scorer == hid:
                hs += 1
            else:
                as_ += 1
        prev_t, prev_sit = t, sit
    if prev_t < REG_END:
        ag, ask, hsk, hg = parse_sit(prev_sit)
        out[hid].append(Segment(g["id"], hid, True, prev_t, REG_END, hs - as_, hg, ag, hsk, ask))
        out[aid].append(Segment(g["id"], aid, False, prev_t, REG_END, as_ - hs, ag, hg, ask, hsk))
    _flag_delayed_penalties(plays, out, hid, aid)
    return out

def _flag_delayed_penalties(plays, out, hid, aid):
    """Goalie out stretches that end in a penalty called on the opponent within 75 seconds
    are delayed penalty extra attackers, not strategic pulls."""
    for tid, home in ((hid, True), (aid, False)):
        idx = 3 if home else 0
        start = None
        for p in plays:
            s = parse_sit(p["sit"])
            t = clock(p["per"], p["t"])
            if s[idx] == 0 and start is None:
                start = t
            if start is None:
                continue
            if p["ev"] == "penalty" and p.get("team") not in (None, tid):
                if t - start <= 75:
                    for seg in out[tid]:
                        if start - 30 <= seg.t0 <= t and seg.own_goalie == 0:
                            seg.delayed_penalty = True
                start = None
            elif s[idx] != 0:
                start = None

@dataclass
class PullEvent:
    game: int
    season: int
    date: str
    team: str
    opp: str
    is_home: bool
    diff: int                 # deficit when the goalie first came out (negative)
    pull_remaining: float     # seconds left in regulation when pulled
    tied: bool                # did the team tie the game in regulation after this point
    won: bool                 # won the game (any period)
    points: int               # standings points earned (2, 1 or 0)
    went_ot: bool
    pp_at_pull: bool          # had a man advantage at the moment of the pull

def first_pulls(g, segs):
    """First strategic goalie pull for each team while trailing in the third period."""
    res = []
    hid, aid = g["home"]["id"], g["away"]["id"]
    final_h, final_a = g["home"]["score"], g["away"]["score"]
    went_ot = g.get("last") in ("OT", "SO")
    for tid, home in ((hid, True), (aid, False)):
        own_final = final_h if home else final_a
        opp_final = final_a if home else final_h
        won = own_final > opp_final
        pts = 2 if won else (1 if went_ot else 0)
        for s in segs[tid]:
            if s.t1 > s.t0 and s.t0 >= 2400 and s.own_goalie == 0 and s.diff < 0 and not s.delayed_penalty and s.opp_goalie == 1:
                res.append(PullEvent(g["id"], g["season"], g["date"], g["home"]["abbrev"] if home else g["away"]["abbrev"],
                                     g["away"]["abbrev"] if home else g["home"]["abbrev"], home, s.diff, REG_END - s.t0,
                                     _tied_after(segs[tid], s.t0), won, pts, went_ot, s.own_sk - 1 > s.opp_sk))
                break
    return res

def _tied_after(segs, t):
    """True if the score was level at the end of regulation or at any later moment."""
    for s in segs:
        if s.t0 >= t and s.t1 > s.t0 and s.diff >= 0:
            return True
    last = [s for s in segs if s.t1 > s.t0]
    return bool(last) and last[-1].diff >= 0
