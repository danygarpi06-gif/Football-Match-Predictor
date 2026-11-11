"""Reusable pipeline: load data -> train once -> predict any match.

Count targets (goals, shots on target, corners) all share the same machinery:
5 regressors -> top-2 ensemble -> negative-binomial over/under probabilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LEAGUES, load_league
from .features import build_training_frame, feature_matrix
from .models import (CORNER_LINES, RANK_LINES, corner_ensemble_backtest, evaluate_corner_models,
                     evaluate_outcome_models, fit_corner_ensemble, fit_outcome_ensemble, outcome_metrics_table,
                     predict_corners, predict_outcome, prob_over, time_split)

TARGETS = {
    "goals": {"label": "Goles", "unit": "goles", "col": "y_goals", "home": "y_hg", "away": "y_ag",
              "lines": [0.5, 1.5, 2.5, 3.5, 4.5], "rank": [1.5, 2.5, 3.5]},
    "sot": {"label": "Tiros al arco", "unit": "tiros al arco", "col": "y_sot", "home": "y_hst", "away": "y_ast",
            "lines": [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5], "rank": [6.5, 7.5, 8.5, 9.5, 10.5]},
    "corners": {"label": "Tiros de esquina", "unit": "córners", "col": "y_corners", "home": "y_hc", "away": "y_ac",
                "lines": CORNER_LINES, "rank": RANK_LINES},
}


def load_data(start: str = "2022-07", refresh: bool = False) -> dict[str, pd.DataFrame]:
    return {lg: load_league(lg, start=start, refresh=refresh, verbose=False) for lg in LEAGUES}


# --------------------------------------------------------------------------- #
# count targets
# --------------------------------------------------------------------------- #
def fit_count_target(frame: pd.DataFrame, leagues: list[str], key: str, seed: int = 42) -> dict | None:
    t = TARGETS[key]
    f = frame[frame[t["col"]].notna()].reset_index(drop=True)
    if len(f) < 300:
        return None  # the data source does not provide this statistic
    X = feature_matrix(f, leagues)
    y = f[t["col"]].astype(float)
    split = time_split(len(f))
    table, info = evaluate_corner_models(X, y, split, seed=seed, lines=t["lines"], rank_lines=t["rank"])
    top2 = table["modelo"].head(2).tolist()
    return {
        "key": key, "table": table, "top2": top2, "columns": list(X.columns),
        "info": {n: {"mu_test": info[n]["mu_test"], "oof": info[n]["oof"]} for n in top2},
        "y_train": y.iloc[:split].to_numpy(), "y_test": y.iloc[split:].to_numpy(),
        "test_leagues": f["league"].iloc[split:].to_numpy(),
        "models": fit_corner_ensemble(top2, X, y, seed=seed),
        "home_models": fit_corner_ensemble(top2, X, f[t["home"]].astype(float), seed=seed),
        "away_models": fit_corner_ensemble(top2, X, f[t["away"]].astype(float), seed=seed),
    }


def count_view(ct: dict, league: str, conf: float) -> dict:
    """Hold-out over/under metrics for one league (all leagues if the league sample is small)."""
    mask = ct["test_leagues"] == league
    use = mask if mask.sum() >= 80 else None
    bt, alpha = corner_ensemble_backtest(ct["top2"], ct["info"], ct["y_train"], ct["y_test"], conf, mask=use,
                                         lines=TARGETS[ct["key"]]["lines"])
    y = ct["y_test"][use] if use is not None else ct["y_test"]
    return {"bt": bt, "alpha": alpha, "n": len(y), "mean_real": float(y.mean()), "league_specific": use is not None}


def count_predict(ct: dict, x: pd.DataFrame, league: str, conf: float) -> dict:
    x = x.reindex(columns=ct["columns"])
    mu = predict_corners(ct["models"], x)
    mh, ma = max(predict_corners(ct["home_models"], x), 0.01), max(predict_corners(ct["away_models"], x), 0.01)
    share = mh / (mh + ma)
    view = count_view(ct, league, conf)
    rows, best_over, best_under = [], None, None
    for L in TARGETS[ct["key"]]["lines"]:
        po = float(prob_over(np.array([mu]), view["alpha"], L)[0])
        r = view["bt"][view["bt"]["linea"] == L].iloc[0]
        over = po >= 0.5
        rows.append({"Línea": L, "P(Over)": po, "P(Under)": 1 - po, "Veredicto": "OVER" if over else "UNDER",
                     "Accuracy histórica": r["accuracy"],
                     "Precisión histórica": r["precision_over"] if over else r["precision_under"],
                     "Recall histórico": r["recall_over"] if over else r["recall_under"]})
        if po >= conf:
            best_over = {"linea": L, "prob": po, "precision": r["senal_precision"], "n": int(r["senal_n"]),
                         "accuracy": r["accuracy"], "recall": r["recall_over"]}
        if (1 - po) >= conf and best_under is None:
            best_under = {"linea": L, "prob": 1 - po, "precision": r["precision_under"],
                          "accuracy": r["accuracy"], "recall": r["recall_under"]}
    return {"mu": mu, "mu_home": mu * share, "mu_away": mu * (1 - share), "lines": pd.DataFrame(rows),
            "best_over": best_over, "best_under": best_under, "view": view}


# --------------------------------------------------------------------------- #
# full training + match prediction (web app)
# --------------------------------------------------------------------------- #
def train(data: dict[str, pd.DataFrame], seed: int = 42) -> dict:
    """Evaluate all models on the most recent 20% and refit the top-2 of each task on everything."""
    leagues = list(data)
    frame, builders = build_training_frame(pd.concat(data.values(), ignore_index=True))
    X = feature_matrix(frame, leagues)
    y = frame["y_outcome"].astype(int)
    split = time_split(len(frame))
    table, preds = evaluate_outcome_models(X, y, split, seed=seed)
    top2 = table["modelo"].head(2).tolist()
    return {
        "leagues": leagues, "builders": builders, "n_matches": len(frame), "n_test": len(frame) - split,
        "test_from": frame["date"].iloc[split], "last_match": frame["date"].max(),
        "columns": list(X.columns), "table": table, "preds": preds, "yte": y.iloc[split:].to_numpy(),
        "test_leagues": frame["league"].iloc[split:].to_numpy(), "top2": top2,
        "models": fit_outcome_ensemble(top2, X, y, seed=seed),
        "counts": {k: fit_count_target(frame, leagues, k, seed) for k in TARGETS},
    }


def predict(tr: dict, league: str, home: str, away: str, kickoff: pd.Timestamp, neutral: bool, conf: float) -> dict:
    fb = tr["builders"][league]
    if not fb.hist[home] or not fb.hist[away]:
        raise ValueError("No hay historial suficiente para uno de los equipos.")
    feats = fb.features(home, away, kickoff, neutral)
    x = feature_matrix(pd.DataFrame([{**feats, "league": league}]), tr["leagues"])
    proba = predict_outcome(tr["models"], x.reindex(columns=tr["columns"]))
    mask = tr["test_leagues"] == league
    specific = mask.sum() >= 30
    return {"proba": proba, "feats": feats, "elo_home": fb.elo[home], "elo_away": fb.elo[away],
            "outcome_table": outcome_metrics_table(tr["preds"], tr["yte"], mask) if specific else tr["table"],
            "n_outcome": int(mask.sum()) if specific else len(mask),
            "counts": {k: count_predict(ct, x, league, conf) for k, ct in tr["counts"].items() if ct}}
