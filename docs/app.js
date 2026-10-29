import { solve, optimalPullTime, fmtClock } from "./pullgoalie.js";

const $ = (id) => document.getElementById(id);
const R = await (await fetch("data/results.json")).json();
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;

// hero numbers
const opt = R.optimal.points, med = R.median_pull;
$("nGames").textContent = R.meta.games.toLocaleString();
$("c1").textContent = fmtClock(opt["-1"]); $("m1").textContent = fmtClock(med["-1"]);
$("c2").textContent = fmtClock(opt["-2"]); $("m2").textContent = fmtClock(med["-2"]);
$("c3").textContent = Math.round(R.points_left.league_per_full_season);
$("c3b").textContent = `${Math.round(R.points_left.late_share * 100)}%`;
const n1 = R.trend.reduce((a, t) => a + t.n_down1, 0);
const s1 = R.trend.reduce((a, t) => a + t.share_optimal_down1 * t.n_down1, 0) / n1;
$("c4").textContent = s1 < 0.01 ? "<1%" : `${Math.round(s1 * 100)}%`; $("c4b").textContent = n1.toLocaleString();
const rate = (d, p) => R.rates.find((r) => r.d === d && r.pulled === p);
$("r1").textContent = rate(-1, 0).gf60.toFixed(1); $("r2").textContent = rate(-1, 1).gf60.toFixed(1);
$("r3").textContent = rate(-1, 0).ga60.toFixed(1); $("r4").textContent = rate(-1, 1).ga60.toFixed(1);
{
  const c = R.curves["-1"], best = Math.max(...c.points);
  const i = c.T.findIndex((t) => t >= med["-1"]);
  $("costMed").textContent = ((best - c.points[i]) * 100).toFixed(1);
}

// bench tool
const teamSel = $("team");
for (const t of [...R.teams].sort((a, b) => a.team.localeCompare(b.team))) teamSel.add(new Option(t.team, t.team));
const ratesFor = (team) => {
  const r = { ...R.rates_js };
  const t = R.teams.find((x) => x.team === team);
  if (t) Object.assign(r, t.rates_override);
  return r;
};
const cache = new Map();
function curveFor(rates, key, d, start, mode) {
  const k = `${key}|${d}|${start}|${mode}`;
  if (cache.has(k)) return cache.get(k);
  const out = [];
  for (let T = 0; T <= start; T += 5) out.push([T, solve(rates, { mode, horizon: start, policy: T }).value(start, d)]);
  cache.set(k, out); return out;
}
function fmtVal(v, mode) { return mode === "points" ? `${v.toFixed(3)} pts` : pct(v); }

function update() {
  const d = +$("deficit").value, t = +$("tLeft").value, mode = $("mode").value, team = teamSel.value;
  $("tLabel").textContent = fmtClock(t);
  const rates = ratesFor(team);
  const Tstar = Math.min(optimalPullTime(rates, d, { mode }), 1200);
  const pullNow = t <= Tstar;
  const v = $("verdict");
  v.textContent = pullNow ? "PULL NOW" : "WAIT"; v.className = `verdict ${pullNow ? "pull" : "wait"}`;
  const tsLabel = Tstar >= 1200 ? "any time in the third" : `${fmtClock(Tstar)} left`;
  $("verdictSub").textContent = pullNow ? `Best time to pull here is ${tsLabel}. You are already inside it.` : `Pull at ${tsLabel}, ${fmtClock(t - Tstar)} from now.`;
  const value = (T) => solve(rates, { mode, horizon: t, policy: Math.min(T, t) }).value(t, d);
  const medT = d === -1 ? med["-1"] : d === -2 ? med["-2"] : 254;
  $("vNow").textContent = fmtVal(value(t), mode);
  $("vBest").textContent = fmtVal(value(Tstar), mode);
  $("vMed").textContent = medT > t ? "already passed" : fmtVal(value(medT), mode);
  drawCurve(curveFor(rates, team, d, t, mode), Tstar, medT, t, mode);
}

function drawCurve(pts, Tstar, medT, start, mode) {
  const svg = $("curve"), W = 760, H = 300, L = 70, Rm = 20, T = 20, B = 46;
  const ys = pts.map((p) => p[1]), yMin = Math.min(...ys), yMax = Math.max(...ys);
  const pad = (yMax - yMin) * 0.1 || 0.01;
  const x = (s) => L + (1 - s / start) * (W - L - Rm);
  const y = (v) => T + (1 - (v - (yMin - pad)) / (yMax - yMin + 2 * pad)) * (H - T - B);
  const path = pts.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
  const ticks = []; const step = start > 300 ? 60 : start > 120 ? 30 : 15;
  for (let s = 0; s <= start; s += step) ticks.push(`<line x1="${x(s)}" x2="${x(s)}" y1="${H - B}" y2="${H - B + 5}" stroke="#5B6475"/><text x="${x(s)}" y="${H - B + 20}" font-size="12" text-anchor="middle" fill="#5B6475">${fmtClock(s)}</text>`);
  const yt = [yMin, (yMin + yMax) / 2, yMax].map((v) => `<text x="${L - 8}" y="${y(v) + 4}" font-size="12" text-anchor="end" fill="#5B6475">${mode === "points" ? v.toFixed(3) : pct(v)}</text>`);
  const vline = (s, col, label, dy) => s <= start ? `<line x1="${x(s)}" x2="${x(s)}" y1="${T}" y2="${H - B}" stroke="${col}" stroke-dasharray="5 4" stroke-width="2"/><text x="${x(s) + 6}" y="${T + dy}" font-size="13" font-weight="700" fill="${col}">${label}</text>` : "";
  svg.innerHTML = `<line x1="${L}" x2="${W - Rm}" y1="${H - B}" y2="${H - B}" stroke="#5B6475"/>${ticks.join("")}${yt.join("")}
    <path d="${path}" fill="none" stroke="#0B1B3B" stroke-width="3"/>
    ${vline(Math.min(Tstar, start), "#0B1B3B", `Best ${Tstar >= 1200 ? "now" : fmtClock(Tstar)}`, 16)}
    ${vline(medT, "#C8102E", `NHL median ${fmtClock(medT)}`, 34)}
    <text x="${(L + W) / 2}" y="${H - 6}" font-size="12" text-anchor="middle" fill="#5B6475">Time left when the goalie is pulled</text>`;
  $("curveCap").textContent = `Expected ${mode === "points" ? "standings points" : "win chance"} from ${fmtClock(start)} left for every possible pull time. Higher is better.`;
}
for (const id of ["deficit", "tLeft", "mode", "team"]) $(id).addEventListener("input", update);
update();

// backtest table
const bt = R.backtest;
$("calCap").textContent = `Predicted versus actual chance of getting at least a point, ${bt.n_states.toLocaleString()} game states from ${bt.n_games.toLocaleString()} held out games. Average prediction ${pct(bt.predicted_rate_all)}, reality ${pct(bt.observed_rate)}.`;
const rows = [
  ["Markov model, all manpower states, real pull time", bt.p_model_all],
  ["Markov model, even strength only, real pull time", bt.p_model],
  ["Same model but everyone pulls at the league median", bt.p_model_median_pull],
  ["Lookup table of past comeback rates by score and clock", bt.p_base],
];
$("btTable").innerHTML = `<tr><th>Out of sample, 2025-26 and 2026-27</th><th>Brier score (lower is better)</th><th>Log loss</th><th>AUC</th></tr>` +
  rows.map(([n, m]) => `<tr><td>${n}</td><td>${m.brier.toFixed(4)}</td><td>${m.logloss.toFixed(4)}</td><td>${m.auc.toFixed(3)}</td></tr>`).join("");

// sensitivity
$("sensTable").innerHTML = `<tr><th>Assumption</th><th>Best pull, down 1</th><th>Best pull, down 2</th></tr>` +
  R.sensitivity.map((s) => `<tr><td>${s.case}</td><td>${fmtClock(s.down1)}</td><td>${fmtClock(s.down2)}</td></tr>`).join("") +
  `<tr><td>90% range from goal rate uncertainty</td><td>${fmtClock(R.optimal.points_ci90["-1"][0])} to ${fmtClock(R.optimal.points_ci90["-1"][2])}</td><td>${fmtClock(R.optimal.points_ci90["-2"][0])} to ${fmtClock(R.optimal.points_ci90["-2"][2])}</td></tr>`;

// team table
$("teamSeasons").textContent = `${R.team_seasons[0]} to ${R.team_seasons[R.team_seasons.length - 1]}`;
const cols = [
  ["team", "Team", (v) => v], ["opt_down1", "Best pull down 1", fmtClock], ["med_down1", "Their median down 1", (v) => (v == null ? "n/a" : fmtClock(v))],
  ["opt_down2", "Best pull down 2", fmtClock], ["med_down2", "Their median down 2", (v) => (v == null ? "n/a" : fmtClock(v))],
  ["sixv5_gf60", "6 on 5 GF/60 (blended)", (v) => v.toFixed(2)], ["sixv5_ga60", "6 on 5 GA/60 (blended)", (v) => v.toFixed(2)],
  ["pulls", "Pulls", (v) => v], ["points_left_per_82", "Points lost per 82 games", (v) => v.toFixed(2)],
];
let sortKey = "points_left_per_82", asc = false;
function drawTeams() {
  const data = [...R.teams].sort((a, b) => ((a[sortKey] ?? -1) > (b[sortKey] ?? -1) ? 1 : -1) * (asc ? 1 : -1));
  $("teamTable").innerHTML = `<tr>${cols.map(([k, n]) => `<th data-k="${k}">${n}${k === sortKey ? (asc ? " ▲" : " ▼") : ""}</th>`).join("")}</tr>` +
    data.map((r) => `<tr>${cols.map(([k, , f]) => `<td>${f(r[k])}</td>`).join("")}</tr>`).join("");
  $("teamTable").querySelectorAll("th").forEach((th) => th.addEventListener("click", () => {
    const k = th.dataset.k; asc = k === sortKey ? !asc : k === "team"; sortKey = k; drawTeams();
  }));
}
drawTeams();
