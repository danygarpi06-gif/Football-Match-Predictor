"""Reusable pipeline: load data -> train once -> predict any match.

Count targets (goals, shots on target, corners) all share the same machinery:
5 regressors -> top-2 ensemble -> negative-binomial over/under probabilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LEAGUES, load_league
from .features import build_training_frame, feature_matrix
from .models import (CORNER_LINES, ENSEMBLE, RANK_LINES, SPLITS, evaluate_count_models, evaluate_outcome_models,
                     fit_corner_ensemble, fit_outcome_ensemble, line_metrics, outcome_table, predict_corners,
                     predict_outcome, prob_over, split_masks, three_way_split)

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


def _split_info(dates: pd.Series, i_val: int, i_test: int) -> dict:
    """Size and date range of each chronological block."""
    bounds = {"train": (0, i_val), "val": (i_val, i_test), "test": (i_test, len(dates))}
    return {sp: {"n": b - a, "desde": dates.iloc[a], "hasta": dates.iloc[b - 1]} for sp, (a, b) in bounds.items()}


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def fit_count_target(frame: pd.DataFrame, leagues: list[str], key: str, seed: int = 42) -> dict | None:
    t = TARGETS[key]
    f = frame[frame[t["col"]].notna()].reset_index(drop=True)
    if len(f) < 300:
        return None  # the data source does not provide this statistic
    X = feature_matrix(f, leagues)
    y = f[t["col"]].astype(float)
    i_val, i_test = three_way_split(len(f))
    ev = evaluate_count_models(X, y, i_val, i_test, seed=seed, lines=t["lines"], rank_lines=t["rank"])
    return {
        "key": key, "metrics": ev["metrics"], "top2": ev["top2"], "mu": ev["mu"], "alpha": ev["alpha"],
        "columns": list(X.columns), "y": y.to_numpy(), "league": f["league"].to_numpy(), "i_test": i_test,
        "splits": _split_info(f["date"], i_val, i_test),
        # models used for real predictions: refitted on ALL matches
        "models": fit_corner_ensemble(ev["top2"], X, y, seed=seed),
        "home_models": fit_corner_ensemble(ev["top2"], X, f[t["home"]].astype(float), seed=seed),
        "away_models": fit_corner_ensemble(ev["top2"], X, f[t["away"]].astype(float), seed=seed),
    }


def train(data: dict[str, pd.DataFrame], seed: int = 42) -> dict:
    """Evaluate every model on train / validation / test and refit the top-2 of each task on everything."""
    leagues = list(data)
    frame, builders = build_training_frame(pd.concat(data.values(), ignore_index=True))
    X = feature_matrix(frame, leagues)
    y = frame["y_outcome"].astype(int)
    i_val, i_test = three_way_split(len(frame))
    ev = evaluate_outcome_models(X, y, i_val, i_test, seed=seed)
    return {
        "leagues": leagues, "builders": builders, "n_matches": len(frame), "n_features": X.shape[1],
        "last_match": frame["date"].max(), "splits": _split_info(frame["date"], i_val, i_test),
        "columns": list(X.columns), "metrics": ev["metrics"], "preds": ev["preds"], "top2": ev["top2"],
        "y": y.to_numpy(), "league": frame["league"].to_numpy(), "i_test": i_test,
        "models": fit_outcome_ensemble(ev["top2"], X, y, seed=seed),
        "counts": {k: fit_count_target(frame, leagues, k, seed) for k in TARGETS},
    }


# --------------------------------------------------------------------------- #
# test-set views (optionally restricted to one league)
# --------------------------------------------------------------------------- #
def _test_mask(obj: dict, league: str | None, min_n: int) -> tuple[np.ndarray, bool]:
    test = np.arange(len(obj["y"])) >= obj["i_test"]
    if league is not None:
        lg = test & (obj["league"] == league)
        if lg.sum() >= min_n:
            return lg, True
    return test, False


def outcome_view(tr: dict, league: str | None = None) -> dict:
    mask, specific = _test_mask(tr, league, 30)
    return {"table": outcome_table(tr["preds"], tr["y"], mask), "n": int(mask.sum()), "league_specific": specific}


def count_view(ct: dict, league: str | None, conf: float) -> dict:
    """Over/under metrics per line of the top-2 ensemble on the test block."""
    mask, specific = _test_mask(ct, league, 80)
    y, mu = ct["y"][mask], ct["mu"][mask]
    bt = line_metrics(y, mu, ct["alpha"], lines=TARGETS[ct["key"]]["lines"], conf=conf)
    return {"bt": bt, "n": int(mask.sum()), "mean_real": float(y.mean()), "league_specific": specific}


# --------------------------------------------------------------------------- #
# match prediction
# --------------------------------------------------------------------------- #
def count_predict(ct: dict, x: pd.DataFrame, league: str, conf: float) -> dict:
    x = x.reindex(columns=ct["columns"])
    mu = predict_corners(ct["models"], x)
    mh, ma = max(predict_corners(ct["home_models"], x), 0.01), max(predict_corners(ct["away_models"], x), 0.01)
    share = mh / (mh + ma)
    view = count_view(ct, league, conf)
    rows, best_over, best_under = [], None, None
    for L in TARGETS[ct["key"]]["lines"]:
        po = float(prob_over(np.array([mu]), ct["alpha"], L)[0])
        r = view["bt"][view["bt"]["linea"] == L].iloc[0]
        side = "over" if po >= 0.5 else "under"
        rows.append({"Línea": L, "P(Over)": po, "P(Under)": 1 - po, "Veredicto": side.upper(),
                     "Accuracy": r["accuracy"], "Precisión": r[f"precision_{side}"],
                     "Recall": r[f"recall_{side}"], "F1": r[f"f1_{side}"]})
        if po >= conf:
            best_over = {"linea": L, "prob": po, "precision": r["senal_precision"], "n": int(r["senal_n"]),
                         "accuracy": r["accuracy"], "recall": r["recall_over"], "f1": r["f1_over"]}
        if (1 - po) >= conf and best_under is None:
            best_under = {"linea": L, "prob": 1 - po, "precision": r["precision_under"],
                          "accuracy": r["accuracy"], "recall": r["recall_under"], "f1": r["f1_under"]}
    return {"mu": mu, "mu_home": mu * share, "mu_away": mu * (1 - share), "lines": pd.DataFrame(rows),
            "best_over": best_over, "best_under": best_under, "view": view}


def predict(tr: dict, league: str, home: str, away: str, kickoff: pd.Timestamp, neutral: bool, conf: float) -> dict:
    fb = tr["builders"][league]
    if not fb.hist[home] or not fb.hist[away]:
        raise ValueError("No hay historial suficiente para uno de los equipos.")
    feats = fb.features(home, away, kickoff, neutral)
    x = feature_matrix(pd.DataFrame([{**feats, "league": league}]), tr["leagues"])
    proba = predict_outcome(tr["models"], x.reindex(columns=tr["columns"]))
    return {"proba": proba, "feats": feats, "elo_home": fb.elo[home], "elo_away": fb.elo[away],
            "counts": {k: count_predict(ct, x, league, conf) for k, ct in tr["counts"].items() if ct}}


# --------------------------------------------------------------------------- #
# markdown report (METRICS.md, shown on GitHub)
# --------------------------------------------------------------------------- #
PCT = ("accuracy", "precision", "recall", "f1", "score", "pct_over_real", "precision_over", "recall_over",
       "f1_over", "precision_under", "recall_under", "f1_under", "senal_precision")
HEADERS = {"modelo": "Modelo", "conjunto": "Conjunto", "accuracy": "Accuracy", "precision": "Precision",
           "recall": "Recall", "f1": "F1", "log_loss": "Log-loss", "score": "Score", "linea": "Línea",
           "pct_over_real": "% over real", "precision_over": "Prec. over", "recall_over": "Recall over",
           "f1_over": "F1 over", "precision_under": "Prec. under", "recall_under": "Recall under",
           "f1_under": "F1 under", "senal_n": "Señales", "senal_precision": "Prec. señal"}


def _md(df: pd.DataFrame) -> str:
    def cell(col, v):
        if isinstance(v, str):
            return v
        if pd.isna(v):
            return "n/a"
        if col in PCT:
            return f"{100 * v:.1f}%"
        if col in ("senal_n",):
            return str(int(v))
        return f"{v:.1f}" if col == "linea" else f"{v:.3f}"
    cols = list(df.columns)
    out = ["| " + " | ".join(HEADERS.get(c, c) for c in cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(cell(c, v) for c, v in zip(cols, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def metrics_markdown(tr: dict, conf: float = 0.60) -> str:
    sp = tr["splits"]
    L = ["# Métricas de los modelos", "",
         f"Generado con datos hasta el {tr['last_match']:%d/%m/%Y}. {tr['n_matches']:,} partidos de las 7 ligas, "
         f"{tr['n_features']} variables. Para regenerarlo: `python report.py`.", "",
         "## Cómo se entrena", "",
         "Los partidos se ordenan por fecha y se dividen en tres bloques, sin mezclarlos:", "",
         "| Bloque | Partidos | Desde | Hasta | Para qué se usa |", "|---|---|---|---|---|"]
    uses = {"train": "Ajustar los modelos", "val": "Elegir los 2 mejores y calibrar el over/under",
            "test": "Solo para medir. No interviene en ninguna decisión"}
    for k, name in SPLITS.items():
        L.append(f"| {name} | {sp[k]['n']:,} | {sp[k]['desde']:%d/%m/%Y} | {sp[k]['hasta']:%d/%m/%Y} | {uses[k]} |")
    L += ["", "Si un modelo rinde mucho mejor en entrenamiento que en prueba, está memorizando (sobreajuste). "
          "La cifra honesta es la de **Prueba**. Los modelos que predicen partidos futuros se vuelven a ajustar "
          "con todos los partidos.", "",
          "## Resultado del partido (1X2)", "",
          f"Mejores dos en validación: **{tr['top2'][0]}** y **{tr['top2'][1]}**. "
          f"«{ENSEMBLE}» es el promedio de ambos y es lo que usa la predicción. "
          "Precision, recall y F1 son el promedio de las tres clases (local, empate, visitante).", "",
          _md(tr["metrics"]), ""]
    for key, ct in tr["counts"].items():
        if not ct:
            continue
        t = TARGETS[key]
        v = count_view(ct, None, conf)
        L += [f"## {t['label']}", "",
              f"Mejores dos en validación: **{ct['top2'][0]}** y **{ct['top2'][1]}**. MAE y RMSE miden el error "
              f"en {t['unit']} por partido. Accuracy, precision, recall y F1 son del over/under, promediados en "
              f"las líneas {t['rank'][0]} a {t['rank'][-1]}.", "", _md(ct["metrics"]), "",
              f"Over/under por línea de «{ENSEMBLE}» en el bloque de prueba ({v['n']:,} partidos, promedio real "
              f"{v['mean_real']:.2f} {t['unit']}). «Señales» son los partidos con P(over) de al menos {conf:.0%}.", "",
              _md(v["bt"]), ""]
    return "\n".join(L)
