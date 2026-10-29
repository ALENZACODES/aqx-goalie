// Pull the Goalie Sooner? Core model in plain JavaScript (no dependencies).
// Same Markov chain as src/pullgoalie/model.py so a web page, a stream overlay
// or a chat bot can ask "pull now or wait?" for any live game state.
// rates: { "d,pulled": [goalsFor60, goalsAgainst60] } for d in -3..2, pulled 0/1.

export const D_MIN = -6, D_MAX = 4;
const NS = D_MAX - D_MIN + 1;
const idx = (d) => d - D_MIN;
const key = (d) => Math.max(-3, Math.min(2, d));

function perSecond(rates) {
  const gf = [new Float64Array(NS), new Float64Array(NS)];
  const ga = [new Float64Array(NS), new Float64Array(NS)];
  for (let d = D_MIN; d <= D_MAX; d++) {
    const k = key(d);
    for (const p of [0, 1]) {
      const r = rates[(k < 0 && p ? `${k},1` : `${k},0`)];
      gf[p][idx(d)] = r[0] / 3600; ga[p][idx(d)] = r[1] / 3600;
    }
  }
  return { gf, ga };
}

function terminal(mode, pOT) {
  const v = new Float64Array(NS);
  for (let d = D_MIN; d <= D_MAX; d++) {
    if (mode === "points") v[idx(d)] = d > 0 ? 2 : d === 0 ? 1 + pOT : 0;
    else if (mode === "win") v[idx(d)] = d > 0 ? 1 : d === 0 ? pOT : 0;
    else v[idx(d)] = d >= 0 ? 1 : 0;
  }
  return v;
}

// policy: null for optimal, or a number T meaning "pull at T seconds left while trailing".
export function solve(rates, { mode = "points", pOT = 0.5, horizon = 1200, policy = null } = {}) {
  const { gf, ga } = perSecond(rates);
  const V = [terminal(mode, pOT)], PULL = [new Uint8Array(NS)];
  for (let t = 1; t <= horizon; t++) {
    const prev = V[t - 1], cur = new Float64Array(NS), pull = new Uint8Array(NS);
    for (let i = 0; i < NS; i++) {
      const up = Math.min(i + 1, NS - 1), dn = Math.max(i - 1, 0);
      const c = [0, 1].map((a) => gf[a][i] * prev[up] + ga[a][i] * prev[dn] + (1 - gf[a][i] - ga[a][i]) * prev[i]);
      const d = i + D_MIN;
      let p = 0;
      if (d < 0) p = policy === null ? (c[1] >= c[0] - 1e-15 ? 1 : 0) : (t <= policy ? 1 : 0);
      pull[i] = p; cur[i] = c[p];
    }
    V.push(cur); PULL.push(pull);
  }
  return { V, PULL, value: (t, d) => V[Math.round(t)][idx(d)] };
}

// Seconds left at which pulling becomes the better choice, for deficit d.
export function optimalPullTime(rates, d, opts = {}) {
  const { PULL } = solve(rates, { ...opts, policy: null });
  for (let t = 1; t < PULL.length; t++) if (!PULL[t][idx(d)]) return t - 1;
  return PULL.length - 1;
}

// Expected value at `start` seconds left, trailing by d, for each pull time.
export function pullCurve(rates, d, { start = 600, step = 5, ...opts } = {}) {
  const out = [];
  for (let T = 0; T <= start; T += step) {
    const { value } = solve(rates, { ...opts, horizon: start, policy: T });
    out.push([T, value(start, d)]);
  }
  return out;
}

// One call answer for a live game state. Used by the dashboard and a future chat overlay.
export function recommend(rates, { secondsLeft, deficit, mode = "points", pOT = 0.5 }) {
  if (deficit >= 0) return { action: "STAY", reason: "Not trailing" };
  const Tstar = optimalPullTime(rates, Math.max(deficit, D_MIN), { mode, pOT });
  const h = Math.max(secondsLeft, 1);
  const now = solve(rates, { mode, pOT, horizon: h, policy: h }).value(h, deficit);
  const best = solve(rates, { mode, pOT, horizon: h, policy: Math.min(Tstar, h) }).value(h, deficit);
  return {
    action: secondsLeft <= Tstar ? "PULL" : "WAIT",
    pullAt: Tstar,
    valueIfPullNow: now,
    valueIfOptimal: best,
  };
}

export const fmtClock = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
