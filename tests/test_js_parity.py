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
