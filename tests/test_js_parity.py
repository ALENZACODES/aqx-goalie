"""The browser model (docs/pullgoalie.js) must agree with the Python model."""
import json, shutil, subprocess, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pullgoalie.model import solve, optimal_pull_times

@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_matches_python():
    R = json.loads((ROOT / "docs/data/results.json").read_text())
    js = ("import {optimalPullTime} from './docs/pullgoalie.js'; import fs from 'fs';"
          "const R=JSON.parse(fs.readFileSync('docs/data/results.json'));"
          "console.log(JSON.stringify([optimalPullTime(R.rates_js,-1),optimalPullTime(R.rates_js,-2)]))")
    out = subprocess.run(["node", "--input-type=module", "-e", js], cwd=ROOT, capture_output=True, text=True, check=True)
    rates = {tuple(int(x) for x in k.split(",")): tuple(v) for k, v in R["rates_js"].items()}
    _, P = solve(rates)
    o = optimal_pull_times(P)
    assert json.loads(out.stdout) == [o[-1], o[-2]]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_curves_match_python_and_peak_at_the_best_time():
    """pullCurve pulls at T for the deficit in question and follows the best rule
    elsewhere, so its maximum must sit at the optimal pull time, in both languages."""
    from pullgoalie.model import curve
    R = json.loads((ROOT / "docs/data/results.json").read_text())
    js = ("import {pullCurve} from './docs/pullgoalie.js'; import fs from 'fs';"
          "const R=JSON.parse(fs.readFileSync('docs/data/results.json'));"
          "console.log(JSON.stringify([-1,-2].map(d=>pullCurve(R.rates_js,d).map(p=>p[1]))))")
    out = json.loads(subprocess.run(["node", "--input-type=module", "-e", js], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout)
    rates = {tuple(int(x) for x in k.split(",")): tuple(v) for k, v in R["rates_js"].items()}
    _, P = solve(rates)
    o = optimal_pull_times(P)
    for d, ys_js in zip((-1, -2), out):
        xs, ys = curve(rates, d)
        assert max(abs(a - b) for a, b in zip(ys, ys_js)) < 1e-9
        assert abs(xs[ys.index(max(ys))] - o[d]) <= 5
