"""Markov chain model of the last period of an NHL game.

State: seconds left in regulation, score difference d from the trailing team's view,
and whether that team's goalie is on the bench. Goals arrive as Poisson processes with
rates (per 60 minutes) that depend on d and on the pull. Backward induction gives the
best policy and the expected standings points (or win chance) for any pull policy."""
from __future__ import annotations
import numpy as np

D_MIN, D_MAX = -6, 4            # score differences tracked
DS = np.arange(D_MIN, D_MAX + 1)
NS = len(DS)
PERIOD = 1200                   # seconds in the third period

def rate_key(d: int) -> int:
    """Rates are estimated for d in {-3, -2, -1, 0, 1, 2}; larger gaps reuse the edge."""
    return int(max(-3, min(2, d)))

def terminal(mode: str, p_ot: float):
    v = np.zeros(NS)
    if mode == "points":          # NHL regular season: 2 for a win, 1 for reaching overtime
        v[DS > 0] = 2.0
        v[DS == 0] = 1.0 + p_ot
    elif mode == "win":           # playoffs: overtime is a coin flip at p_ot
        v[DS > 0] = 1.0
        v[DS == 0] = p_ot
    elif mode == "tie":           # chance of tying or leading at the end of regulation
        v[DS >= 0] = 1.0
    else:
        raise ValueError(mode)
    return v

def rate_arrays(rates: dict):
    """rates[(d, pulled)] = (gf_per60, ga_per60). Returns per second arrays (2, NS)."""
    gf = np.zeros((2, NS)); ga = np.zeros((2, NS))
    for i, d in enumerate(DS):
        k = rate_key(d)
        for pulled in (0, 1):
            key = (k, pulled) if (k < 0 and pulled) else (k, 0)
            f, a = rates[key]
            gf[pulled, i] = f / 3600.0
            ga[pulled, i] = a / 3600.0
    return gf, ga

def solve(rates: dict, mode: str = "points", p_ot: float = 0.5, horizon: int = PERIOD, policy=None):
    """Backward induction.
    policy None means optimal; otherwise a dict {deficit: seconds_left_to_pull} (deficit < 0)
    where deficits beyond the dict use the most negative key.
    Returns V (horizon+1, NS), the value of each state with the goalie IN at that moment,
    and PULL (horizon+1, NS) booleans of the action taken."""
    gf, ga = rate_arrays(rates)
    V = np.zeros((horizon + 1, NS)); PULL = np.zeros((horizon + 1, NS), bool)
    V[0] = terminal(mode, p_ot)
    up = np.r_[np.arange(1, NS), NS - 1]       # index after a goal for
    dn = np.r_[0, np.arange(0, NS - 1)]        # index after a goal against
    trailing = DS < 0
    if policy is not None:
        keys = sorted(policy)
        thr = np.array([policy.get(int(d), policy[keys[0]]) if d < 0 else -1 for d in DS], float)
    for t in range(1, horizon + 1):
        prev = V[t - 1]
        cont = np.empty((2, NS))
        for a in (0, 1):
            cont[a] = gf[a] * prev[up] + ga[a] * prev[dn] + (1 - gf[a] - ga[a]) * prev
        if policy is None:
            pull = trailing & (cont[1] >= cont[0] - 1e-15)
        else:
            pull = trailing & (t <= thr)
        PULL[t] = pull
        V[t] = np.where(pull, cont[1], cont[0])
    return V, PULL

def optimal_pull_times(PULL):
    """Seconds left at which pulling first becomes optimal, for each deficit."""
    out = {}
    for i, d in enumerate(DS):
        if d < 0:
            col = PULL[:, i]
            # largest t such that pulling is optimal for every s <= t
            bad = np.where(~col[1:])[0]
            out[int(d)] = int(bad[0]) if len(bad) else len(col) - 1
    return out

def value_at(V, t, d):
    return float(V[int(round(t)), int(d) - D_MIN])

def optimal_thresholds(rates: dict, mode: str = "points", p_ot: float = 0.5) -> dict:
    """{deficit: seconds left at which pulling becomes optimal} from the full solve."""
    _, PULL = solve(rates, mode, p_ot)
    return optimal_pull_times(PULL)

def curve(rates, d, mode="points", p_ot=0.5, start=600, step=5):
    """Expected value at `start` seconds left, trailing by d, for each pull time T.
    The team pulls at T while trailing by d and follows the optimal rule at every
    other deficit (for example, down 2 it pulls at T; if it scores it puts the goalie
    back in until the down-1 threshold). This makes the curve's maximum coincide with
    the optimal pull time for d."""
    opt = optimal_thresholds(rates, mode, p_ot)
    xs, ys = [], []
    for T in range(0, start + 1, step):
        V, _ = solve(rates, mode, p_ot, horizon=start, policy=_policy_for(d, T, opt))
        xs.append(T); ys.append(value_at(V, start, d))
    return xs, ys

def _policy_for(d, T, opt=None):
    """Pull at T seconds left when trailing by d. Other deficits use the optimal
    thresholds `opt` when given, otherwise T as well (the old behaviour)."""
    pol = {k: T for k in range(D_MIN, 0)}
    if opt is not None:
        pol.update({int(k): int(v) for k, v in opt.items() if k < 0})
        pol[int(d)] = T
    return pol

def policy_grid(rates: dict, Ts, mode: str = "points", p_ot: float = 0.5, horizon: int = 600,
                d: int | None = None, opt: dict | None = None):
    """Values for many fixed policies at once. Policy j pulls at Ts[j] seconds left.
    With d and opt given, Ts[j] applies only when trailing by d and the other deficits
    follow the optimal thresholds opt; otherwise Ts[j] applies whenever trailing.
    Returns array (horizon+1, len(Ts), NS) of values with the goalie in."""
    gf, ga = rate_arrays(rates)
    Ts = np.asarray(Ts, float)[:, None]
    up = np.r_[np.arange(1, NS), NS - 1]; dn = np.r_[0, np.arange(0, NS - 1)]
    trailing = (DS < 0)[None, :]
    if d is not None and opt is not None:
        thr = np.repeat(Ts, NS, axis=1)                       # (len(Ts), NS)
        for i, dd in enumerate(DS):
            if dd < 0 and int(dd) != int(d):
                thr[:, i] = opt.get(int(dd), opt[min(opt)])
    else:
        thr = Ts
    V = np.zeros((horizon + 1, len(Ts), NS))
    V[0] = terminal(mode, p_ot)[None, :]
    for t in range(1, horizon + 1):
        prev = V[t - 1]
        c0 = gf[0] * prev[:, up] + ga[0] * prev[:, dn] + (1 - gf[0] - ga[0]) * prev
        c1 = gf[1] * prev[:, up] + ga[1] * prev[:, dn] + (1 - gf[1] - ga[1]) * prev
        V[t] = np.where(trailing & (t <= thr), c1, c0)
    return V

def recommend(rates: dict, seconds_left: float, deficit: int, mode: str = "points", p_ot: float = 0.5):
    """Answer for a live game state: pull now or wait, and the value of each choice."""
    if deficit >= 0:
        return {"action": "STAY", "reason": "not trailing"}
    d = max(int(deficit), D_MIN)
    _, PULL = solve(rates, mode, p_ot)
    t_star = optimal_pull_times(PULL)[d]
    h = max(1, int(round(seconds_left)))
    opt = optimal_pull_times(PULL)
    pol = lambda T: _policy_for(d, T, opt)
    if h > PERIOD:
        return {"action": "WAIT", "pull_at": t_star, "seconds_until": h - t_star}
    v_now = value_at(solve(rates, mode, p_ot, horizon=h, policy=pol(h))[0], h, d)
    v_best = value_at(solve(rates, mode, p_ot, horizon=h, policy=pol(min(t_star, h)))[0], h, d)
    return {"action": "PULL" if h <= t_star else "WAIT", "pull_at": int(t_star),
            "seconds_until": max(0, h - t_star), "value_pull_now": v_now, "value_best": v_best, "mode": mode}
