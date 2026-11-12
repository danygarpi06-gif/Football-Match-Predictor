"""Reusable pipeline: load data -> train once -> predict any match.

Count targets (goals, shots on target, corners) all share the same machinery:
5 regressors -> top-2 ensemble -> negative-binomial over/under probabilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LEAGUES, load_league
from .features import build_training_frame, feature_matrix
from .players import STARTER, fetch_lineup, load_lineups
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


def load_players(data: dict[str, pd.DataFrame], refresh: bool = False, verbose: bool = False) -> dict:
    """Line-ups and player stats of every finished match (empty when ESPN is unreachable)."""
    try:
        return load_lineups(data, refresh=refresh, verbose=verbose)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  Sin datos de jugadores ({exc}); se usan solo datos de equipo.")
        return {"matches": {}, "names": {}}


def train(data: dict[str, pd.DataFrame], seed: int = 42, players: dict | None = None) -> dict:
    """Evaluate every model on train / validation / test and refit the top-2 of each task on everything."""
    leagues = list(data)
    players = players or {"matches": {}, "names": {}}
    frame, builders, book = build_training_frame(pd.concat(data.values(), ignore_index=True), players["matches"])
    X = feature_matrix(frame, leagues)
    y = frame["y_outcome"].astype(int)
    i_val, i_test = three_way_split(len(frame))
    ev = evaluate_outcome_models(X, y, i_val, i_test, seed=seed)
    return {
        "leagues": leagues, "builders": builders, "n_matches": len(frame), "n_features": X.shape[1],
        "book": book, "names": {int(k): v for k, v in players["names"].items()},
        "n_lineups": int(frame["h_xi_g"].notna().sum()) if book else 0,
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


def confirmed_lineup(league: str, match_id: str) -> dict | None:
    """Starting XIs that ESPN has already published for a match (usually about one hour before kick-off)."""
    try:
        rec, names = fetch_lineup(league, str(match_id))
    except Exception:  # noqa: BLE001
        return None
    if not rec:
        return None
    return {"h": [pl[0] for pl in rec["h"]["p"] if pl[1] == STARTER],
            "a": [pl[0] for pl in rec["a"]["p"] if pl[1] == STARTER], "names": {int(k): v for k, v in names.items()}}


def team_players(tr: dict, league: str, team: str) -> dict | None:
    """Usual XI and recent squad of a team (player ids), or None when there is no player data."""
    book = tr["book"]
    tid = book.team_id.get((league, team)) if book else None
    if not tid:
        return None
    usual = book.usual_xi(tid)
    return {"tid": tid, "usual": usual, "squad": book.squad(tid)} if len(usual) == 11 else None


def without_players(tr: dict, league: str, team: str, out: list[int]) -> list[int]:
    """Usual XI with some players removed; each one is replaced by the most used squad player of his line."""
    tp, book = team_players(tr, league, team), tr["book"]
    xi = list(tp["usual"])
    for pid in out:
        if pid not in xi:
            continue
        pool = [c for c in tp["squad"] if c not in xi and c not in out]
        same = [c for c in pool if book.p[c]["grp"] == book.p[pid]["grp"]]
        outfield = [c for c in pool if book.p[c]["grp"] != "G"]
        pick = (same or (pool if book.p[pid]["grp"] == "G" else outfield) or pool or [None])[0]
        if pick is not None:
            xi[xi.index(pid)] = pick
    return xi


GROUPS = {"G": "Portero", "D": "Defensa", "M": "Medio", "F": "Delantero", "U": "Sin dato"}


def player_table(tr: dict, league: str, team: str, xi: list[int]) -> pd.DataFrame:
    """What the model knows about each starter (recency-weighted, per match started)."""
    book, tp = tr["book"], team_players(tr, league, team)
    _, share = book.start_scores(tp["tid"])
    rows = []
    for pid in xi:
        g, a, sh, sot = book._rates(pid)
        st = book.p.get(pid, {"grp": "U", "r": 0.0, "n": 0})
        rows.append({"Jugador": tr["names"].get(pid, str(pid)), "Posición": GROUPS[st["grp"]],
                     "Titular reciente": share.get(pid, 0.0), "Partidos": st["n"], "Goles": g, "Asistencias": a,
                     "Tiros": sh, "Tiros al arco": sot, "Rating": st["r"]})
    order = {v: i for i, v in enumerate(GROUPS.values())}
    return pd.DataFrame(rows).sort_values("Posición", key=lambda c: c.map(order), kind="stable").reset_index(drop=True)


def predict(tr: dict, league: str, home: str, away: str, kickoff: pd.Timestamp, neutral: bool, conf: float,
            home_xi: list[int] | None = None, away_xi: list[int] | None = None) -> dict:
    """home_xi / away_xi: 11 player ids each. When omitted, the usual XI of each team is used."""
    fb = tr["builders"][league]
    if not fb.hist[home] or not fb.hist[away]:
        raise ValueError("No hay historial suficiente para uno de los equipos.")
    xi, usual = {}, {}
    for side, team, chosen in (("h", home, home_xi), ("a", away, away_xi)):
        tp = team_players(tr, league, team)
        usual[side] = tp["usual"] if tp else None
        ids = chosen if chosen and len(chosen) == 11 and tp else usual[side]
        if ids:
            xi[side] = [(pid, "") for pid in ids]
    feats = fb.features(home, away, kickoff, neutral, xi)
    x = feature_matrix(pd.DataFrame([{**feats, "league": league}]), tr["leagues"])
    proba = predict_outcome(tr["models"], x.reindex(columns=tr["columns"]))
    res = {"proba": proba, "feats": feats, "elo_home": fb.elo[home], "elo_away": fb.elo[away],
           "counts": {k: count_predict(ct, x, league, conf) for k, ct in tr["counts"].items() if ct},
           "xi": {s: [pid for pid, _ in v] for s, v in xi.items()}, "usual": usual, "base": None}
    if any(s in xi and set(res["xi"][s]) != set(usual[s]) for s in ("h", "a")):
        res["base"] = predict(tr, league, home, away, kickoff, neutral, conf)  # same match with the usual XIs
    return res


# --------------------------------------------------------------------------- #
# markdown report (METRICS.md, shown on GitHub)
# --------------------------------------------------------------------------- #
PCT = ("accuracy", "precision", "recall", "f1", "score", "pct_over_real", "precision_over", "recall_over",
       "f1_over", "precision_under", "recall_under", "f1_under", "senal_precision")
HEADERS = {"modelo": "Modelo", "conjunto": "Conjunto", "tarea": "Tarea", "datos": "Datos", "accuracy": "Accuracy", "precision": "Precision",
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


def player_impact(tr: dict, base: dict) -> pd.DataFrame:
    """Top-2 ensemble with and without player data, on validation and test."""
    tasks = [("Resultado 1X2", tr["metrics"], base["metrics"])]
    tasks += [(TARGETS[k]["label"], tr["counts"][k]["metrics"], base["counts"][k]["metrics"])
              for k in TARGETS if tr["counts"][k] and base["counts"][k]]
    rows = []
    for label, with_players, team_only in tasks:
        for split in ("val", "test"):
            for name, m in (("Solo equipos", team_only), ("Equipos + jugadores", with_players)):
                r = m[(m["modelo"] == ENSEMBLE) & (m["conjunto"] == SPLITS[split])].iloc[0]
                rows.append({"tarea": label, "conjunto": SPLITS[split], "datos": name, "accuracy": r["accuracy"],
                             "f1": r["f1"], "log_loss": r.get("log_loss", np.nan), "MAE": r.get("MAE", np.nan),
                             "RMSE": r.get("RMSE", np.nan)})
    return pd.DataFrame(rows)


PLAYERS_TEXT = ("Además de los datos de equipo, cada partido usa su **alineación titular**. De cada jugador se "
                "llevan goles, asistencias, tiros y tiros al arco por partido de titular, con más peso en lo reciente, "
                "y un rating que sube cuando el equipo rinde mejor de lo esperado con él en cancha. El once se resume "
                "en la suma de sus jugadores y se compara con el once habitual del equipo, para medir bajas y rotaciones. "
                "Todo se calcula solo con partidos anteriores.")


def metrics_markdown(tr: dict, conf: float = 0.60, impact: pd.DataFrame | None = None) -> str:
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
          "con todos los partidos.", ""]
    if tr["book"]:
        L += ["## Datos de jugadores y alineaciones", "",
              f"{PLAYERS_TEXT} Hay alineación en {tr['n_lineups']:,} de los {tr['n_matches']:,} partidos.", ""]
        if impact is not None:
            L += ["Comparación de «" + ENSEMBLE + "» entrenando sin y con los datos de jugadores. En accuracy y F1 "
                  "más alto es mejor; en log-loss, MAE y RMSE más bajo es mejor.", "", _md(impact), ""]
    L += ["## Resultado del partido (1X2)", "",
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
