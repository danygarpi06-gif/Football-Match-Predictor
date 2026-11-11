"""Reusable pipeline for the web app: load data -> train once -> predict any match."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LEAGUES, load_league
from .features import build_training_frame, feature_matrix
from .models import (CORNER_LINES, corner_ensemble_backtest, evaluate_corner_models, evaluate_outcome_models,
                     fit_corner_ensemble, fit_outcome_ensemble, outcome_metrics_table, predict_corners,
                     predict_outcome, prob_over, time_split)


def load_data(start: str = "2022-07", refresh: bool = False) -> dict[str, pd.DataFrame]:
    return {lg: load_league(lg, start=start, refresh=refresh, verbose=False) for lg in LEAGUES}


def train(data: dict[str, pd.DataFrame], seed: int = 42) -> dict:
    """Evaluate the 5+5 models on the most recent 20% and refit the top-2 of each on everything."""
    leagues = list(data)
    frame, builders = build_training_frame(pd.concat(data.values(), ignore_index=True))
    X = feature_matrix(frame, leagues)
    y = frame["y_outcome"].astype(int)
    split = time_split(len(frame))
    table, preds = evaluate_outcome_models(X, y, split, seed=seed)
    top2 = table["modelo"].head(2).tolist()

    fc = frame[frame["y_corners"].notna()].reset_index(drop=True)
    Xc = feature_matrix(fc, leagues)
    yc = fc["y_corners"].astype(float)
    splitc = time_split(len(fc))
    ctable, cinfo = evaluate_corner_models(Xc, yc, splitc, seed=seed)
    ctop2 = ctable["modelo"].head(2).tolist()
    return {
        "leagues": leagues, "builders": builders, "n_matches": len(frame), "n_test": len(frame) - split,
        "test_from": frame["date"].iloc[split], "last_match": frame["date"].max(),
        "columns": list(X.columns), "ccolumns": list(Xc.columns),
        "table": table, "preds": preds, "yte": y.iloc[split:].to_numpy(),
        "test_leagues": frame["league"].iloc[split:].to_numpy(), "top2": top2,
        "models": fit_outcome_ensemble(top2, X, y, seed=seed),
        "ctable": ctable, "ctop2": ctop2,
        "cinfo": {n: {"mu_test": cinfo[n]["mu_test"], "oof": cinfo[n]["oof"]} for n in ctop2},
        "yc_train": yc.iloc[:splitc].to_numpy(), "yc_test": yc.iloc[splitc:].to_numpy(),
        "ctest_leagues": fc["league"].iloc[splitc:].to_numpy(),
        "cmodels": fit_corner_ensemble(ctop2, Xc, yc, seed=seed),
    }


def league_view(tr: dict, league: str, conf: float) -> dict:
    """Hold-out metrics restricted to one league (falls back to all leagues if the sample is small)."""
    mask = tr["test_leagues"] == league
    out_table = outcome_metrics_table(tr["preds"], tr["yte"], mask) if mask.sum() >= 30 else tr["table"]
    cmask = tr["ctest_leagues"] == league
    use = cmask if cmask.sum() >= 80 else None
    bt, alpha = corner_ensemble_backtest(tr["ctop2"], tr["cinfo"], tr["yc_train"], tr["yc_test"], conf, mask=use)
    return {"outcome_table": out_table, "n_outcome": int(mask.sum()) if mask.sum() >= 30 else len(mask),
            "bt": bt, "alpha": alpha, "n_corners": int(cmask.sum()) if use is not None else len(cmask),
            "league_specific": use is not None}


def predict(tr: dict, league: str, home: str, away: str, kickoff: pd.Timestamp, neutral: bool, conf: float) -> dict:
    fb = tr["builders"][league]
    if not fb.hist[home] or not fb.hist[away]:
        raise ValueError("No hay historial suficiente para uno de los equipos.")
    feats = fb.features(home, away, kickoff, neutral)
    x = feature_matrix(pd.DataFrame([{**feats, "league": league}]), tr["leagues"])
    proba = predict_outcome(tr["models"], x.reindex(columns=tr["columns"]))
    mu = predict_corners(tr["cmodels"], x.reindex(columns=tr["ccolumns"]))
    view = league_view(tr, league, conf)
    rows, best_over, best_under = [], None, None
    for L in CORNER_LINES:
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
    return {"proba": proba, "mu": mu, "lines": pd.DataFrame(rows), "best_over": best_over, "best_under": best_under,
            "feats": feats, "elo_home": fb.elo[home], "elo_away": fb.elo[away], "view": view}
