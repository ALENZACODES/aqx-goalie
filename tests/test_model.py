import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from pullgoalie.model import solve, optimal_pull_times, policy_grid, value_at, recommend, D_MIN
from pullgoalie.parse import parse_sit, clock, game_segments, first_pulls

R = {(-3, 0): (2.2, 2.4), (-3, 1): (8.5, 21.2), (-2, 0): (2.56, 2.51), (-2, 1): (6.66, 23.5),
     (-1, 0): (2.47, 2.10), (-1, 1): (6.90, 20.3), (0, 0): (2.17, 2.17), (1, 0): (2.25, 2.43), (2, 0): (2.40, 2.37)}

def test_terminal_values():
    V, _ = solve(R, "points", 0.5, horizon=10)
    assert value_at(V, 0, 1) == 2 and value_at(V, 0, 0) == 1.5 and value_at(V, 0, -1) == 0

def test_values_are_probabilities_in_win_mode():
    V, _ = solve(R, "win")
    assert V.min() >= 0 and V.max() <= 1 + 1e-12

def test_optimal_beats_every_fixed_policy():
    Vopt, _ = solve(R, "points", horizon=600)
    G = policy_grid(R, range(0, 601, 10), "points", horizon=600)
    for d in (-1, -2):
        assert value_at(Vopt, 600, d) >= G[600, :, d - D_MIN].max() - 1e-9

def test_down_two_pulls_earlier_than_down_one():
    _, P = solve(R)
    o = optimal_pull_times(P)
    assert o[-2] > o[-1] > 60

def test_grid_matches_solver():
    G = policy_grid(R, [120], "points", horizon=600)
    V, _ = solve(R, "points", horizon=600, policy={k: 120 for k in range(-6, 0)})
    assert np.allclose(G[:, 0, :], V)

def test_better_empty_net_defence_means_earlier_pull():
    R2 = dict(R); R2[(-1, 1)] = (6.9, 15.0)
    _, P1 = solve(R); _, P2 = solve(R2)
    assert optimal_pull_times(P2)[-1] > optimal_pull_times(P1)[-1]

def test_recommend():
    assert recommend(R, 600, -1)["action"] == "WAIT"
    assert recommend(R, 60, -1)["action"] == "PULL"
    assert recommend(R, 60, 0)["action"] == "STAY"

def test_parse_helpers():
    assert parse_sit("0651") == (0, 6, 5, 1) and parse_sit(None) is None
    assert clock(3, "18:00") == 3480

def _play(per, t, sit, ev, team=None, hs=None, as_=None, o=0):
    return {"per": per, "ptype": "REG", "t": t, "sit": sit, "ev": ev, "team": team, "hs": hs, "as": as_, "ord": o}

def test_segments_and_pull_detection():
    g = {"id": 1, "season": 20252026, "date": "2026-01-01", "type": 2, "last": "OT",
         "home": {"id": 10, "abbrev": "TOR", "score": 2}, "away": {"id": 20, "abbrev": "MTL", "score": 1},
         "plays": [
             _play(1, "05:00", "1551", "goal", 20, 0, 1, 1),
             _play(3, "17:00", "1551", "faceoff", o=2),
             _play(3, "17:10", "1560", "shot-on-goal", 10, o=3),   # TOR pulls, down 1
             _play(3, "18:30", "1560", "goal", 10, 1, 1, 4),        # TOR ties
             _play(3, "18:31", "1551", "faceoff", o=5),
         ]}
    segs = game_segments(g)
    pulls = first_pulls(g, segs)
    assert len(pulls) == 1 and pulls[0].team == "TOR" and pulls[0].diff == -1
    assert 170 <= pulls[0].pull_remaining <= 180 and pulls[0].tied and pulls[0].points == 2
    six = sum(s.t1 - s.t0 for s in segs[10] if s.own_goalie == 0)
    assert 80 <= six <= 90

def test_delayed_penalty_is_not_a_pull():
    g = {"id": 2, "season": 20252026, "date": "2026-01-01", "type": 2, "last": "REG",
         "home": {"id": 10, "abbrev": "TOR", "score": 0}, "away": {"id": 20, "abbrev": "MTL", "score": 1},
         "plays": [
             _play(1, "05:00", "1551", "goal", 20, 0, 1, 1),
             _play(3, "15:00", "1560", "shot-on-goal", 10, o=2),
             _play(3, "15:20", "1560", "penalty", 20, o=3),
             _play(3, "15:20", "1451", "faceoff", o=4),
         ]}
    segs = game_segments(g)
    assert first_pulls(g, segs) == []
