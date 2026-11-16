"""Fits and scores the Bayes model (models.bayes_model), an option behind SWP_MODEL=bayes.

    python -m training.bayes_model train      # latent NUTS fit on every training run (~1-1.5 h),
                                              # then the linear model -> src/assets/bayes_model.json
    python -m training.bayes_model evaluate   # the eval year, once (descriptive)

train: the latent Bayes stream fit (training.bayes_stream: FITS["latent"], STUDY_NUTS and
LATENT_DRAWS, float64) on the training windows of every training run (2016-10 .. 2025-10,
as the shipped linear model), cached in src/data/bayes_model_cache/ (--refit to redo it);
N_DRAWS posterior draws kept (the same choice as training.bayes_features); the parr and
mean regressors for every training run; the submitted linear config refitted with them.
evaluate: the asset on the eval year (2025-10 .. 2026-10) through the live imputation,
against the shipped linear model, a reference refitted on the same table as the Bayes
model's linear part, and persistence, with the official metric, overall and by run regime.
Results: src/experiments/bayes_model_eval_year.csv. Descriptive only: the model was chosen
on the yearly folds (docs/bayes_features.md) and is not the submitted model.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from models import bayes_model as bm
from models.linear_model import LinearModel
from training import bayes_stream as tbs
from training import competition_score as cs
from training import model_diagnostics as dg
from training.datasets import DATA_START, TRAIN_END

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "bayes_model_cache"
FIT = "latent"


def log(msg):
    print(msg, flush=True)


def latent_fit(ff, refit=False):
    """(draws, info) of the latent fit on the training windows, cached."""
    path, info_path = CACHE_DIR / "latent_draws.npz", CACHE_DIR / "latent_fit.json"
    if path.exists() and info_path.exists() and not refit:
        log(f"latent fit from the cache ({path})")
        return dict(np.load(path)), json.loads(info_path.read_text("utf-8"))
    import numpyro

    numpyro.enable_x64()  # as the study (STUDY_NUTS)
    t = time.perf_counter()
    draws, info = tbs.study_fit(ff["d"], tbs.FITS[FIT])
    info = {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
            for k, v in info.items()}
    info["fit s"] = time.perf_counter() - t
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **draws)
    info_path.write_text(json.dumps(info), encoding="utf-8")
    return draws, info


def keep_draws(draws, n_draws=bm.N_DRAWS, seed=0):
    """The n_draws posterior draws models.bayes_model.features would pick (same rng)."""
    rng = np.random.default_rng(seed)
    n_all = len(draws["rho_up"])
    idx = rng.choice(n_all, size=min(n_draws, n_all), replace=False)
    return {k: np.asarray(draws[k])[idx] for k in tbs.bs.GLOBALS}


def build(data, ff, draws):
    """The BayesModel: the fit's input statistics and draws, the linear model refitted with
    the Bayes regressors on every training run."""
    stream = ff["model"]
    stream.bayes, stream.draws, stream.table = tbs.FITS[FIT], keep_draws(draws), None
    if tuple(stream.pre.config.input_features) != tuple(data.cfg.input_features):
        raise RuntimeError("the stream model's regressors differ from the linear config's")
    cfg = replace(data.cfg, features=(*data.cfg.features,
                                      *(f"{bm.signal_name(s)}_lead_0h" for s in bm.SIGNALS)))
    model = bm.BayesModel(stream, LinearModel(cfg))
    x = model.design(data.values, ff["pos_fit"], ff["x_fit"])
    model.linear = LinearModel(cfg).fit(data.values, ff["pos_fit"], x, end=ff["start_pos"])
    return model


def train(out=bm.ASSET, refit=False):
    started = time.perf_counter()
    data = dg.Data()
    ff = tbs.fold_fit(data, TRAIN_END)
    log(f"data ready in {time.perf_counter() - started:.0f} s: {ff['wininfo']}, "
        f"{len(ff['fit6'])} training runs")
    draws, info = latent_fit(ff, refit)
    log(f"latent fit: {info['fit s']:.0f} s, R-hat max {info['r_hat max']:.3f}, "
        f"ESS min {info['ess min']:.0f}, divergences {info['divergences']:.0f}")
    t = time.perf_counter()
    model = build(data, ff, draws)
    model.fit_info = {"start": f"{DATA_START:%Y-%m-%dT%H:%MZ}",
                      "end": f"{TRAIN_END:%Y-%m-%dT%H:%MZ}", "runs": int(len(ff["fit6"])),
                      **ff["wininfo"], "draws kept": bm.N_DRAWS,
                      **{k: info[k] for k in ("fit s", "r_hat max", "ess min", "divergences")}}
    model.save(out)
    log(f"linear part fitted in {time.perf_counter() - t:.0f} s; wrote {out} "
        f"({Path(out).stat().st_size / 1e3:.0f} kB)")
    return model


def evaluate(path=bm.ASSET):
    t = time.perf_counter()
    data = dg.Data()
    model = bm.BayesModel.load(path)
    t0s = data.eval_t0s
    pos = data.full.index.get_indexer(t0s)
    names = list(model.regressor_names)
    xv = data.eval_table[names].to_numpy(dtype=float)
    obs, pers = cs.verifying(data.truth, t0s), cs.persistence(data.truth, t0s)
    regime = data.run_regimes(t0s)
    ff = tbs.fold_fit(data, TRAIN_END)
    ref = LinearModel(data.cfg).fit(data.values, ff["pos_fit"], ff["x_fit"], end=ff["start_pos"])
    shipped = LinearModel.load()
    forecasts = {
        "bayes model": model.forecast(data.full_values, pos, xv),
        "reference (same table)": ref.forecast(data.full_values, pos, xv),
        "shipped linear model": shipped.forecast(
            data.full_values, pos,
            data.eval_table[list(shipped.config.input_features)].to_numpy(dtype=float)),
        "persistence": np.repeat(pers[:, None], cs.HORIZON, axis=1),
    }
    rows = {k: dg.regime_scores(f, obs, pers, regime) for k, f in forecasts.items()}
    table = pd.DataFrame(rows).T
    base = table.loc["reference (same table)"]
    rel = (table / base - 1).add_suffix(" vs reference")
    out = table.join(rel[["all vs reference"]])
    out.to_csv(EXPERIMENTS / "bayes_model_eval_year.csv", float_format="%.6g",
               lineterminator="\n")
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format,
                           "display.max_columns", 30):
        log(f"eval year {t0s[0]:%Y-%m-%d} .. {t0s[-1]:%Y-%m-%d}, {len(t0s)} runs "
            f"({time.perf_counter() - t:.0f} s):\n{out.to_string()}\n\n"
            f"vs the reference by regime:\n{rel.to_string()}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--refit", action="store_true", help="redo the cached latent fit")
    args = parser.parse_args(argv)
    if args.command == "train":
        train(refit=args.refit)
    else:
        evaluate()


if __name__ == "__main__":
    sys.exit(main())
