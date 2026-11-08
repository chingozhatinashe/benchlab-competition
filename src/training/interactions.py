"""Interaction and nonlinear terms for the linear model, by forward selection on the yearly folds.

    python -m training.interactions --pilot --folds 2024   # step 1 on one fold, timed
    python -m training.interactions [--max-seconds N]      # forward selection; resumable
    python -m training.interactions --summary

Item (iv) of the follow-up plan to docs/model_diagnostics.md. The XGBoost hybrid's trees
didn't beat the linear model (docs/hybrid_pfss.md), so couplings go into the linear model as
a few explicit terms (models.linear_model Config.terms; computed from the last speed and
the standardised regressors, so nothing extra live; terms with the optional coronal-hole
strip are left out of its submodel).

Candidates (z = standardised regressor at each lead; last = (last speed - mean) / 100), over
the operands ch, dhcs, bt, xray, density and last: every pair (15 int:a*b), every square
(6 sq:a), and hinge:ch>0, hinge:dhcs>0. The one-fold pilot used a hand-picked subset of
13 (8 pairs, 3 squares, 2 hinges); all pairs were chosen after the pilot so the candidates
don't depend on that choice (user decision, 2026-11-08).
Forward selection: each step adds the candidate with the best six-fold score; a step is
kept only if it gains >= 0.3% over the current set and wins >= 4 of 6 folds; at most 3
terms. Decision rule for the final set, set before running: keep if > 1% better than the
reference and winning >= 4 of 6 folds, without the slow runs getting worse; otherwise a
set gaining >= 0.3% on >= 4 folds goes forward to the combination step. Score: official
S_final / persistence, geometric mean over folds (training.slow_wind.score_fold). The
eval year is not used. Results: src/experiments/interactions.csv
(cache: src/data/interactions_cache/).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from training import slow_wind as sw
from training import xgboost_model as tx
from training.folds import YEARLY

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "interactions_cache"
STEP_GAIN, STEP_FOLDS, MAX_TERMS = 0.003, 4, 3
KEEP_GAIN, KEEP_FOLDS = 0.01, 4
CH, DHCS, BT = "ch_cm_lead_72h", "pfss_d_hcs_lead_72h", "bt_mean_1h"
XRAY, DENS = "xray_peak_max_72h", "density_mean_24h"
OPERANDS = (CH, DHCS, BT, XRAY, DENS, "last")
CANDIDATES = (tuple(f"int:{a}*{b}" for i, a in enumerate(OPERANDS) for b in OPERANDS[i + 1:])
              + tuple(f"sq:{a}" for a in OPERANDS) + (f"hinge:{CH}>0", f"hinge:{DHCS}>0"))
SHORT = {CH: "ch", DHCS: "dhcs", BT: "bt", XRAY: "xray", DENS: "density"}


def log(msg):
    print(msg, flush=True)


def short(term):
    for long_name, s in SHORT.items():
        term = term.replace(long_name, s)
    return term


def key(terms):
    return " + ".join(short(t) for t in terms) or "reference"


class Scores:
    """Per-fold scores of term sets, cached on disk."""

    def __init__(self, data, years, pilot):
        self.data, self.pilot = data, pilot
        self.folds = [(s, e) for s, e in YEARLY if years is None or s.year in years]
        self.rows = {}
        for s, _ in self.folds:
            path = CACHE_DIR / f"{s:%Y}.json"
            self.rows[f"{s:%Y}"] = ({} if pilot or not path.exists()
                                    else json.loads(path.read_text("utf-8")))
        self.timer = {}

    def get(self, sets):
        """{set key: {year: row}} for each term set, computing what's missing."""
        specs = {key(ts): {"terms": tuple(ts)} for ts in sets}
        specs.setdefault("reference", {})
        for s, e in self.folds:
            year = f"{s:%Y}"
            todo = [k for k in specs if k not in self.rows[year]]
            if todo:
                t = time.perf_counter()
                self.rows[year].update(sw.score_fold(self.data, s, e, todo, self.timer, specs))
                self.timer["fold"] = self.timer.get("fold", 0.0) + time.perf_counter() - t
                if not self.pilot:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    (CACHE_DIR / f"{year}.json").write_text(json.dumps(self.rows[year]),
                                                            encoding="utf-8")
        return {k: {y: self.rows[y][k] for y in self.rows} for k in specs}

    def rel(self, a, b="reference"):
        ys = sorted(self.rows)
        return np.array([self.rows[y][a]["all"] / self.rows[y][b]["all"] for y in ys])


def run(years=None, pilot=False, max_seconds=None):
    started = time.perf_counter()
    data = sw.Data()
    log(f"data ready in {time.perf_counter() - started:.0f} s")
    scores = Scores(data, years, pilot)
    chosen, path = [], [{"step": 0, "terms": "reference", "vs reference": 0.0}]
    for step in range(1, MAX_TERMS + 1):
        if max_seconds and time.perf_counter() - started > max_seconds:
            log("time limit reached; run the same command again to continue")
            return None
        t = time.perf_counter()
        cands = [c for c in CANDIDATES if c not in chosen]
        scores.get([chosen + [c] for c in cands])
        current = key(chosen)
        rows = []
        for c in cands:
            k = key(chosen + [c])
            r = scores.rel(k, current)
            rows.append({"term": short(c), "set": k,
                         "vs current": float(np.exp(np.log(r).mean()) - 1),
                         "folds won": int((r < 1).sum()),
                         "vs reference": float(np.exp(np.log(scores.rel(k)).mean()) - 1)})
        step_table = pd.DataFrame(rows).sort_values("vs current")
        log(f"\nstep {step} ({time.perf_counter() - t:.0f} s), against {current}:\n"
            + step_table.round(4).to_string(index=False))
        if pilot:
            log(f"  time: {scores.timer.get('fold', 0):.0f} s for {len(cands) + 1} sets on "
                f"{len(scores.folds)} fold(s); peak memory {tx.peak_mb():.0f} MB; full "
                f"selection (up to 3 steps x 6 folds) -> ~"
                f"{3 * 6 * scores.timer.get('fold', 0) / len(scores.folds) / 60:.0f} min")
            return step_table
        best = step_table.iloc[0]
        ok = best["vs current"] <= -STEP_GAIN and best["folds won"] >= STEP_FOLDS
        path.append({"step": step, "terms": best["set"], "vs reference": best["vs reference"],
                     "vs previous": best["vs current"], "folds won": int(best["folds won"]),
                     "kept": bool(ok)})
        if not ok:
            break
        chosen.append(next(c for c in cands if short(c) == best["term"]))
    return summarize(scores, chosen, path)


def summarize(scores, chosen, path):
    final = key(chosen)
    r = scores.rel(final)
    gain = float(np.exp(np.log(r).mean()) - 1)
    won = int((r < 1).sum())
    ys = sorted(scores.rows)
    slow = np.array([scores.rows[y][final]["slow wind"] / scores.rows[y]["reference"]["slow wind"]
                     for y in ys])
    slow_gain = float(np.exp(np.log(slow).mean()) - 1)
    keep = gain < -KEEP_GAIN and won >= KEEP_FOLDS and slow_gain <= 0
    forward = gain <= -STEP_GAIN and won >= KEEP_FOLDS
    long = [{"fold": y, "set": k, "persistence": scores.rows[y]["persistence"], **v}
            for y in ys for k, v in scores.rows[y].items() if isinstance(v, dict)]
    pd.DataFrame(long).to_csv(EXPERIMENTS / "interactions.csv", index=False, float_format="%.6g")
    pd.DataFrame(path).to_csv(EXPERIMENTS / "interactions_path.csv", index=False,
                              float_format="%.6g")
    log(f"\nselection path:\n{pd.DataFrame(path).round(4).to_string(index=False)}")
    log(f"final set: {final}: {100 * gain:+.2f}% vs reference, {won} of {len(ys)} folds, "
        f"slow runs {100 * slow_gain:+.2f}%; keep {keep}, forward {forward}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args(argv)
    run(args.folds, args.pilot, args.max_seconds)


if __name__ == "__main__":
    sys.exit(main())
