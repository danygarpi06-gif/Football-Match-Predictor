#!/usr/bin/env python
"""Football Match Predictor - CLI.

Ejemplos:
  python predict.py                                   # modo interactivo
  python predict.py --league premier_league --home Arsenal --away Chelsea
  python predict.py --league liga_mx --list-teams
  python predict.py --league champions --home "Real Madrid" --away "Bayern" --scope all --conf 0.65
"""
from __future__ import annotations

import argparse
import difflib
import sys

import numpy as np
import pandas as pd

from fmp.data import LEAGUES, current_teams, load_league, upcoming_fixtures
from fmp.features import build_training_frame, feature_matrix
from fmp.pipeline import TARGETS, count_predict, fit_count_target
from fmp.models import (CORNER_LINES, OUTCOME_LABELS, corner_ensemble_backtest, evaluate_corner_models,
                        evaluate_outcome_models, fit_corner_ensemble, fit_outcome_ensemble, outcome_metrics_table,
                        predict_corners, predict_outcome, prob_over, time_split)

LINE = "─" * 78


def hr(title: str = "") -> None:
    print(f"\n{LINE}\n{title}\n{LINE}" if title else LINE)


def pct(x: float) -> str:
    return "  n/a " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:5.1f}%"


# --------------------------------------------------------------------------- #
# interactive helpers
# --------------------------------------------------------------------------- #
def choose(prompt: str, options: list[str], allow_text: bool = False) -> str:
    for i, opt in enumerate(options, 1):
        print(f"  {i:>2}. {opt}")
    while True:
        raw = input(f"{prompt} ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        if allow_text and raw:
            return raw
        print("  Opción inválida, intenta de nuevo.")


ALIASES = {  # apodos comunes -> nombre (o fragmento) usado por ESPN
    "chivas": "Guadalajara", "rayados": "Monterrey", "la maquina": "Cruz Azul", "pumas": "Pumas UNAM",
    "tigres": "Tigres UANL", "aguilas": "América", "san luis": "Atlético de San Luis", "juarez": "FC Juárez",
    "man utd": "Manchester United", "man united": "Manchester United", "man city": "Manchester City",
    "spurs": "Tottenham", "wolves": "Wolverhampton", "barca": "Barcelona", "atleti": "Atlético Madrid",
    "real sociedad": "Real Sociedad", "inter": "Internazionale", "juve": "Juventus", "bayern": "Bayern Munich",
    "dortmund": "Borussia Dortmund", "gladbach": "Borussia Mönchengladbach", "leverkusen": "Bayer Leverkusen",
    "psg": "Paris Saint-Germain", "atletico mineiro": "Atlético Mineiro", "galo": "Atlético Mineiro",
    "flu": "Fluminense", "mengao": "Flamengo", "tricolor paulista": "São Paulo", "verdao": "Palmeiras",
}


def _norm(text: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def match_team(name: str, teams: list[str]) -> str:
    name = ALIASES.get(_norm(name), name)
    low = {_norm(t): t for t in teams}
    if _norm(name) in low:
        return low[_norm(name)]
    partial = [t for t in teams if _norm(name) in _norm(t)]
    if len(partial) == 1:
        return partial[0]
    close = difflib.get_close_matches(name, teams, n=3, cutoff=0.5)
    cands = partial or close
    if len(cands) == 1:
        return cands[0]
    if cands:
        print(f"  '{name}' es ambiguo, candidatos: {', '.join(cands)}")
        return choose("  Elige el número:", cands)
    sys.exit(f"Equipo no encontrado: '{name}'. Usa --list-teams para ver los nombres disponibles.")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Predicción 1X2 y tiros de esquina (over/under)")
    ap.add_argument("--league", choices=list(LEAGUES), help="liga a analizar")
    ap.add_argument("--home", help="equipo local")
    ap.add_argument("--away", help="equipo visitante")
    ap.add_argument("--neutral", action="store_true", help="cancha neutral (p.ej. final de Champions)")
    ap.add_argument("--scope", choices=["all", "league"], default="all",
                    help="entrenar con las 7 ligas juntas (all, default, más preciso) o solo con la liga elegida (league, más rápido)")
    ap.add_argument("--conf", type=float, default=0.60, help="confianza mínima para recomendar una línea de córners")
    ap.add_argument("--start", default="2022-07", help="mes inicial del histórico (YYYY-MM)")
    ap.add_argument("--source", choices=["auto", "espn", "fdcouk"], default="auto", help="fuente de datos")
    ap.add_argument("--refresh", action="store_true", help="ignorar caché y volver a descargar todo")
    ap.add_argument("--list-teams", action="store_true", help="solo listar equipos de la liga y salir")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hr("FOOTBALL MATCH PREDICTOR")
    league = args.league
    if not league:
        print("Selecciona la liga:")
        name = choose("Liga (número):", [LEAGUES[k]["name"] for k in LEAGUES])
        league = next(k for k, v in LEAGUES.items() if v["name"] == name)
    print(f"Liga: {LEAGUES[league]['name']}")

    # ---- data ------------------------------------------------------------- #
    hr("1) Descargando datos (ESPN, actualizado a hoy; caché local en data_cache/)")
    leagues_to_load = list(LEAGUES) if args.scope == "all" else [league]
    data = {lg: load_league(lg, source=args.source, start=args.start, refresh=args.refresh) for lg in leagues_to_load}
    df_league = data[league]
    finished = df_league[df_league["status"] == "FT"]
    print(f"Partidos terminados en {LEAGUES[league]['name']}: {len(finished)} "
          f"(último: {finished['date'].max():%Y-%m-%d})  | con córners: {int(finished[['hc', 'ac']].notna().all(axis=1).sum())}")

    teams = current_teams(df_league)
    if args.list_teams:
        hr(f"Equipos ({len(teams)})")
        print("\n".join(f"  - {t}" for t in teams))
        return

    # ---- team selection --------------------------------------------------- #
    neutral = args.neutral
    kickoff = pd.Timestamp.now("UTC").tz_convert(None)
    if args.home and args.away:
        home, away = match_team(args.home, teams), match_team(args.away, teams)
    else:
        hr("2) Selección del partido")
        fixtures = upcoming_fixtures(df_league)
        home = away = None
        if len(fixtures):
            print("Próximos partidos programados (elige número) o escribe 0 para escoger equipos manualmente:")
            labels = [f"{r.date:%a %d/%m %H:%M} UTC  {r.home} vs {r.away}" for r in fixtures.itertuples()]
            for i, lab in enumerate(labels, 1):
                print(f"  {i:>2}. {lab}")
            raw = input("Partido: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(labels):
                r = fixtures.iloc[int(raw) - 1]
                home, away, neutral, kickoff = r["home"], r["away"], bool(r["neutral"]), r["date"]
        if home is None:
            print("Equipos disponibles:")
            home = choose("Equipo LOCAL (número):", teams)
            away = choose("Equipo VISITANTE (número):", [t for t in teams if t != home])
    if home == away:
        sys.exit("El local y el visitante no pueden ser el mismo equipo.")
    sched = df_league[(df_league["status"] == "SCHED") & (df_league["home"] == home) & (df_league["away"] == away)]
    if len(sched):
        kickoff, neutral = sched.iloc[0]["date"], bool(sched.iloc[0]["neutral"]) or neutral
    print(f"\nPartido: {home} (local) vs {away} (visitante)  |  fecha usada: {kickoff:%Y-%m-%d}"
          f"{'  |  cancha neutral' if neutral else ''}")

    # ---- features --------------------------------------------------------- #
    all_matches = pd.concat(data.values(), ignore_index=True)
    frame, builders = build_training_frame(all_matches)
    X_all = feature_matrix(frame, leagues_to_load)
    y_out = frame["y_outcome"].astype(int)
    split = time_split(len(frame))
    test_rows = frame.iloc[split:]
    scope_txt = "las 7 ligas" if args.scope == "all" else LEAGUES[league]["name"]
    print(f"Dataset de entrenamiento ({scope_txt}): {len(frame)} partidos, {X_all.shape[1]} variables | "
          f"entrenamiento: {split}  prueba (más recientes): {len(frame) - split} "
          f"({test_rows['date'].min():%Y-%m-%d} → {test_rows['date'].max():%Y-%m-%d})")

    # ---- outcome models --------------------------------------------------- #
    hr("3) Resultado del partido (1X2): 5 modelos evaluados en los partidos más recientes")
    table, preds = evaluate_outcome_models(X_all, y_out, split, seed=args.seed)
    yte = y_out.iloc[split:].to_numpy()

    def print_outcome_table(tab: pd.DataFrame, y_ref: np.ndarray, title: str) -> None:
        base = pd.Series(y_ref).value_counts(normalize=True).max()
        print(f"{title} (n={len(y_ref)}, baseline clase mayoritaria = {pct(base).strip()})")
        print(f"{'Modelo':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'LogLoss':>10}{'Score':>8}")
        for r in tab.itertuples():
            print(f"{r.modelo:<22}{pct(r.accuracy):>10}{pct(r.precision):>11}{pct(r.recall):>9}{pct(r.f1):>8}{r.log_loss:>10.3f}{pct(r.score):>8}")

    print_outcome_table(table, yte, f"Métricas en prueba - {scope_txt}")
    print("(Precision/Recall = macro sobre las 3 clases; Score = promedio de accuracy, precision y recall)")
    league_mask = (test_rows["league"] == league).to_numpy()
    if args.scope == "all" and league_mask.sum() >= 30:
        print()
        print_outcome_table(outcome_metrics_table(preds, yte, league_mask), yte[league_mask],
                            f"Mismos modelos, solo partidos de {LEAGUES[league]['name']}")
    top2 = table["modelo"].head(2).tolist()
    print(f"\n>>> TOP 2 modelos 1X2: {top2[0]}  y  {top2[1]}  (se promedian sus probabilidades)")

    # ---- corner models ---------------------------------------------------- #
    has_corners = frame["y_corners"].notna()
    corner_ok = has_corners.sum() >= 300
    if corner_ok:
        hr("4) Tiros de esquina totales: 5 modelos (over/under evaluado en líneas 7.5 a 11.5)")
        fc = frame[has_corners].reset_index(drop=True)
        Xc = feature_matrix(fc, leagues_to_load)
        yc = fc["y_corners"].astype(float)
        splitc = time_split(len(fc))
        ctable, cinfo = evaluate_corner_models(Xc, yc, splitc, seed=args.seed)
        print(f"{'Modelo':<28}{'MAE':>7}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'Score':>8}")
        for r in ctable.itertuples():
            print(f"{r.modelo:<28}{r.MAE:>7.2f}{pct(r.accuracy):>10}{pct(r.precision):>11}{pct(r.recall):>9}{pct(r.score):>8}")
        ctop2 = ctable["modelo"].head(2).tolist()
        print(f"\n>>> TOP 2 modelos de córners: {ctop2[0]}  y  {ctop2[1]}  (se promedia su media esperada)")
        cmask = (fc.iloc[splitc:]["league"] == league).to_numpy()
        use_mask = cmask if (args.scope == "all" and cmask.sum() >= 80) else None
        bt, alpha = corner_ensemble_backtest(ctop2, cinfo, yc.iloc[:splitc].to_numpy(), yc.iloc[splitc:].to_numpy(),
                                             args.conf, mask=use_mask)
        yte_c = yc.iloc[splitc:].to_numpy()[use_mask] if use_mask is not None else yc.iloc[splitc:].to_numpy()
        print(f"\nBacktest del ensamble en {LEAGUES[league]['name'] if use_mask is not None else scope_txt} "
              f"(n={len(yte_c)} partidos de prueba, media real {yte_c.mean():.2f} córners):")
        print(f"{'Línea':>6}{'%Over real':>12}{'Accuracy':>10}{'Prec.Over':>11}{'Rec.Over':>10}{'Prec.Under':>12}{'Rec.Under':>11}{f'Señal≥{args.conf:.0%}':>12}{'Prec.señal':>12}")
        for r in bt.itertuples():
            print(f"{r.linea:>6}{pct(r.pct_over_real):>12}{pct(r.accuracy):>10}{pct(r.precision_over):>11}{pct(r.recall_over):>10}"
                  f"{pct(r.precision_under):>12}{pct(r.recall_under):>11}{r.senal_n:>12}{pct(r.senal_precision):>12}")
        print("('Señal' = partidos de prueba donde el modelo dio P(over) ≥ confianza; 'Prec.señal' = % que realmente fue over)")
    else:
        print("\n(La fuente actual no trae tiros de esquina suficientes para esta liga; se omite el modelo de córners)")

    # ---- final prediction ------------------------------------------------- #
    hr(f"5) PREDICCIÓN: {home} vs {away}")
    fb = builders[league]
    if fb.hist[home] == [] or fb.hist[away] == []:
        sys.exit("No hay historial suficiente para uno de los equipos.")
    feats = fb.features(home, away, kickoff, neutral)
    x = feature_matrix(pd.DataFrame([{**feats, "league": league}]), leagues_to_load).reindex(columns=X_all.columns)
    models = fit_outcome_ensemble(top2, X_all, y_out, seed=args.seed)
    proba = predict_outcome(models, x)
    print(f"Elo: {home} {fb.elo[home]:.0f}  |  {away} {fb.elo[away]:.0f}   "
          f"Forma (pts/partido últimos 5): {home} {feats['h_pts_s']:.2f} | {away} {feats['a_pts_s']:.2f}")
    print(f"\n  Gana {home:<24} {pct(proba[0])}")
    print(f"  Empate{'':<23} {pct(proba[1])}")
    print(f"  Gana {away:<24} {pct(proba[2])}")
    winner = int(proba.argmax())
    loser = {0: away, 2: home}.get(winner)
    print(f"\n>>> Resultado más probable: {OUTCOME_LABELS[winner]}"
          + (f" → GANADOR: {home if winner == 0 else away}  |  PERDEDOR: {loser}" if winner != 1 else "")
          + f"  (prob. {pct(proba[winner]).strip()})")
    p_no_draw = proba[0] / (proba[0] + proba[2])
    print(f"    Si descartamos el empate: {home} {pct(p_no_draw).strip()} vs {away} {pct(1 - p_no_draw).strip()}")

    if corner_ok:
        cmodels = fit_corner_ensemble(ctop2, Xc, yc, seed=args.seed)
        xc = x[Xc.columns]
        mu = predict_corners(cmodels, xc)
        print(f"\nTIROS DE ESQUINA - total esperado: {mu:.2f}  (promedio últimos 10: {home} {feats['h_cf_l']:.1f} a favor / "
              f"{feats['h_ca_l']:.1f} en contra; {away} {feats['a_cf_l']:.1f} / {feats['a_ca_l']:.1f})")
        print(f"{'Línea':>6}{'P(Over)':>10}{'P(Under)':>10}{'Veredicto':>12}{'Acc. hist.':>12}{'Prec. hist.':>13}")
        best_over = best_under = None
        for L in CORNER_LINES:
            po = float(prob_over(np.array([mu]), alpha, L)[0])
            row = bt[bt["linea"] == L].iloc[0]
            verdict = "OVER" if po >= 0.5 else "UNDER"
            prec_hist = row["precision_over"] if po >= 0.5 else row["precision_under"]
            print(f"{L:>6}{pct(po):>10}{pct(1 - po):>10}{verdict:>12}{pct(row['accuracy']):>12}{pct(prec_hist):>13}")
            if po >= args.conf:
                best_over = (L, po, row)
            if (1 - po) >= args.conf and best_under is None:
                best_under = (L, 1 - po, row)
        print()
        if best_over:
            L, po, row = best_over
            print(f">>> RECOMENDACIÓN OVER (línea más alta con P ≥ {args.conf:.0%}): MÁS DE {L} córners  "
                  f"— prob. {po:.0%} | histórico de esta señal: precisión {pct(row['senal_precision']).strip()} "
                  f"(n={int(row['senal_n'])}), accuracy de la línea {pct(row['accuracy']).strip()}, recall over {pct(row['recall_over']).strip()}")
        else:
            print(f">>> Ninguna línea OVER alcanza la confianza {args.conf:.0%}; la más probable es OVER "
                  f"{max(CORNER_LINES, key=lambda L: float(prob_over(np.array([mu]), alpha, L)[0]) if float(prob_over(np.array([mu]), alpha, L)[0]) >= 0.5 else -1)}")
        if best_under:
            L, pu, row = best_under
            print(f">>> RECOMENDACIÓN UNDER (línea más baja con P ≥ {args.conf:.0%}): MENOS DE {L} córners "
                  f"— prob. {pu:.0%} | precisión histórica under {pct(row['precision_under']).strip()}, recall under {pct(row['recall_under']).strip()}")
    # ---- goals and shots on target (same method as corners) --------------- #
    for key in ("goals", "sot"):
        t = TARGETS[key]
        ct = fit_count_target(frame, leagues_to_load, key, seed=args.seed)
        if ct is None:
            print(f"\n({t['label']}: la fuente actual no trae este dato para esta liga)")
            continue
        r = count_predict(ct, x, league, args.conf)
        top = ct["table"].head(2)
        print(f"\n{t['label'].upper()} - total esperado: {r['mu']:.2f}  ({home} {r['mu_home']:.2f} | {away} {r['mu_away']:.2f})")
        print("    TOP 2 modelos: " + "  y  ".join(
            f"{m.modelo} (acc {pct(m.accuracy).strip()}, prec {pct(m.precision).strip()}, rec {pct(m.recall).strip()})" for m in top.itertuples()))
        print(f"{'Línea':>6}{'P(Over)':>10}{'P(Under)':>10}{'Veredicto':>12}{'Acc. hist.':>12}{'Prec. hist.':>13}{'Rec. hist.':>12}")
        for row in r["lines"].itertuples(index=False):
            print(f"{row[0]:>6}{pct(row[1]):>10}{pct(row[2]):>10}{row[3]:>12}{pct(row[4]):>12}{pct(row[5]):>13}{pct(row[6]):>12}")
        bo, bu = r["best_over"], r["best_under"]
        if bo:
            print(f">>> OVER recomendado: MÁS DE {bo['linea']} {t['unit']} — prob. {bo['prob']:.0%} | "
                  f"precisión histórica de la señal {pct(bo['precision']).strip()} (n={bo['n']})")
        if bu:
            print(f">>> UNDER recomendado: MENOS DE {bu['linea']} {t['unit']} — prob. {bu['prob']:.0%} | "
                  f"precisión histórica under {pct(bu['precision']).strip()}")
    print()


if __name__ == "__main__":
    main()
