#!/usr/bin/env python
"""Football Match Predictor - CLI.

Ejemplos:
  python predict.py                                   # modo interactivo
  python predict.py --league premier_league --home Arsenal --away Chelsea
  python predict.py --league liga_mx --list-teams
  python predict.py --league champions --home "Real Madrid" --away "Bayern" --scope all --conf 0.65
  python predict.py --league la_liga --home "Real Madrid" --away Barcelona --home-out "Mbappé,Vinícius"
"""
from __future__ import annotations

import argparse
import difflib
import sys

import numpy as np
import pandas as pd  # noqa: F401

from fmp.data import LEAGUES, current_teams, load_league, upcoming_fixtures
from fmp.models import ENSEMBLE, OUTCOME_LABELS, SPLITS
from fmp.pipeline import (TARGETS, confirmed_lineup, load_players, outcome_view, player_table, predict, team_players,
                          train, without_players)

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


def match_players(raw: str | None, tr: dict, league: str, team: str) -> list[int]:
    """Comma-separated player names -> ids of that team's squad (accents and case are ignored)."""
    tp = team_players(tr, league, team)
    if not raw or not tp:
        return []
    out = []
    for name in (x.strip() for x in raw.split(",") if x.strip()):
        hits = [pid for pid in tp["squad"] if _norm(name) in _norm(tr["names"].get(pid, ""))]
        if len(hits) == 1:
            out.append(hits[0])
        else:
            found = ", ".join(tr["names"].get(pid, "") for pid in hits) or "nadie"
            print(f"  Aviso: '{name}' no identifica a un solo jugador de {team} (coincide con: {found}); se ignora.")
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Predicción 1X2, goles, tiros al arco y tiros de esquina (over/under)")
    ap.add_argument("--league", choices=list(LEAGUES), help="liga a analizar")
    ap.add_argument("--home", help="equipo local")
    ap.add_argument("--away", help="equipo visitante")
    ap.add_argument("--neutral", action="store_true", help="cancha neutral (p.ej. final de Champions)")
    ap.add_argument("--scope", choices=["all", "league"], default="all",
                    help="entrenar con las 7 ligas juntas (all, default, más preciso) o solo con la liga elegida (league, más rápido)")
    ap.add_argument("--conf", type=float, default=0.60, help="confianza mínima para recomendar una línea over/under")
    ap.add_argument("--start", default="2022-07", help="mes inicial del histórico (YYYY-MM)")
    ap.add_argument("--home-out", help="jugadores del local que NO serán titulares, separados por coma")
    ap.add_argument("--away-out", help="jugadores del visitante que NO serán titulares, separados por coma")
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

    players = load_players(data, refresh=args.refresh, verbose=True)

    teams = current_teams(df_league)
    if args.list_teams:
        hr(f"Equipos ({len(teams)})")
        print("\n".join(f"  - {t}" for t in teams))
        return

    # ---- team selection --------------------------------------------------- #
    neutral, match_id = args.neutral, None
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
        match_id = sched.iloc[0]["match_id"]
    print(f"\nPartido: {home} (local) vs {away} (visitante)  |  fecha usada: {kickoff:%Y-%m-%d}"
          f"{'  |  cancha neutral' if neutral else ''}")

    # ---- training + evaluation -------------------------------------------- #
    scope_txt = "las 7 ligas" if args.scope == "all" else LEAGUES[league]["name"]
    hr(f"3) Entrenamiento y evaluación ({scope_txt})")
    print("Entrenando 5 modelos por tarea (resultado, goles, tiros al arco, córners)...")
    tr = train(data, seed=args.seed, players=players)
    print(f"\n{tr['n_matches']} partidos ordenados por fecha, {tr['n_features']} variables, tres bloques:")
    print(f"Partidos con alineación titular y datos de jugadores: {tr['n_lineups']}")
    uses = {"train": "ajustar los modelos", "val": "elegir los 2 mejores y calibrar el over/under",
            "test": "solo medir (no interviene en ninguna decisión)"}
    for sp, name in SPLITS.items():
        b = tr["splits"][sp]
        print(f"  {name:<14}{b['n']:>6} partidos  {b['desde']:%Y-%m-%d} → {b['hasta']:%Y-%m-%d}   {uses[sp]}")
    print("Si un modelo rinde mucho mejor en Entrenamiento que en Prueba, está sobreajustando. La cifra honesta es Prueba.")

    def print_metrics(df: pd.DataFrame, cols: list[str]) -> None:
        print(f"{'Modelo':<29}{'Conjunto':<15}" + "".join(f"{c:>10}" for c in cols))
        for i, row in enumerate(df.to_dict("records")):
            if i and i % 3 == 0:
                print()
            cells = "".join(f"{row[c.lower() if c not in ('MAE', 'RMSE') else c]:>10.3f}" if c in ("MAE", "RMSE", "log_loss")
                            else f"{pct(row[c.lower()]):>10}" for c in cols)
            print(f"{row['modelo']:<29}{row['conjunto']:<15}{cells}")

    hr("3a) Resultado 1X2: precision, recall y F1 son macro sobre las 3 clases")
    print_metrics(tr["metrics"], ["Accuracy", "Precision", "Recall", "F1", "log_loss"])
    print(f"\n>>> TOP 2 (elegidos en Validación): {tr['top2'][0]}  y  {tr['top2'][1]}. '{ENSEMBLE}' promedia ambos y es lo que predice.")
    ov = outcome_view(tr, league)
    if ov["league_specific"] and args.scope == "all":
        r = ov["table"][ov["table"]["modelo"] == ENSEMBLE].iloc[0]
        print(f"    {ENSEMBLE} en Prueba, solo {LEAGUES[league]['name']} (n={ov['n']}): accuracy {pct(r['accuracy']).strip()}, "
              f"precision {pct(r['precision']).strip()}, recall {pct(r['recall']).strip()}, F1 {pct(r['f1']).strip()}")
    for i, (key, ct) in enumerate(tr["counts"].items()):
        t = TARGETS[key]
        if ct is None:
            print(f"\n({t['label']}: la fuente actual no trae este dato para esta liga)")
            continue
        hr(f"3{'bcd'[i]}) {t['label']}: MAE y RMSE en {t['unit']} por partido; el resto es over/under en líneas {t['rank'][0]} a {t['rank'][-1]}")
        print_metrics(ct["metrics"], ["MAE", "RMSE", "Accuracy", "Precision", "Recall", "F1"])
        print(f"\n>>> TOP 2 (elegidos en Validación): {ct['top2'][0]}  y  {ct['top2'][1]}")

    # ---- final prediction ------------------------------------------------- #
    hr(f"4) PREDICCIÓN: {home} vs {away}")
    xi, origin = {"h": None, "a": None}, "habitual (los 11 con más titularidades recientes)"
    official = confirmed_lineup(league, match_id) if match_id and tr["book"] else None
    if official:
        xi, origin = {"h": official["h"], "a": official["a"]}, "confirmada por ESPN"
        tr["names"].update(official["names"])
    for side, team, raw in (("h", home, args.home_out), ("a", away, args.away_out)):
        out = match_players(raw, tr, league, team)
        if out:
            xi[side], origin = without_players(tr, league, team, out), "habitual con las bajas indicadas"
    try:
        res = predict(tr, league, home, away, kickoff, neutral, args.conf, xi["h"], xi["a"])
    except ValueError as exc:
        sys.exit(str(exc))
    proba, feats = res["proba"], res["feats"]
    if res["xi"]:
        print(f"Alineación usada: {origin}. Para quitar jugadores: --home-out \"Nombre,Nombre\" --away-out \"Nombre\"")
        for side, team in (("h", home), ("a", away)):
            if side not in res["xi"]:
                print(f"  {team}: sin datos de jugadores")
                continue
            tab = player_table(tr, league, team, res["xi"][side])
            print(f"  {team}: " + ", ".join(tab["Jugador"]))
            top = tab.sort_values("Goles", ascending=False).head(3)
            print("      más peligrosos (goles por partido de titular): "
                  + ", ".join(f"{r.Jugador} {r.Goles:.2f}" for r in top.itertuples()))
    else:
        print("Sin datos de jugadores para este partido; la predicción usa solo datos de equipo.")
    print(f"Elo: {home} {res['elo_home']:.0f}  |  {away} {res['elo_away']:.0f}   "
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
    base = res["base"]
    if base:
        b = base["proba"]
        print(f"    Con las alineaciones habituales sería: {home} {pct(b[0]).strip()}, empate {pct(b[1]).strip()}, "
              f"{away} {pct(b[2]).strip()}. Goles esperados {base['counts']['goals']['mu']:.2f} → "
              f"{res['counts']['goals']['mu']:.2f} con esta alineación.")

    for key, r in res["counts"].items():
        t, v = TARGETS[key], r["view"]
        where = LEAGUES[league]["name"] if v["league_specific"] else scope_txt
        print(f"\n{t['label'].upper()} - total esperado: {r['mu']:.2f}  ({home} {r['mu_home']:.2f} | {away} {r['mu_away']:.2f})")
        print(f"    Métricas históricas de cada línea: bloque de Prueba en {where} (n={v['n']}, promedio real {v['mean_real']:.2f})")
        print(f"{'Línea':>6}{'P(Over)':>10}{'P(Under)':>10}{'Veredicto':>11}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
        for row in r["lines"].itertuples(index=False):
            print(f"{row[0]:>6}{pct(row[1]):>10}{pct(row[2]):>10}{row[3]:>11}{pct(row[4]):>10}{pct(row[5]):>11}{pct(row[6]):>9}{pct(row[7]):>8}")
        bo, bu = r["best_over"], r["best_under"]
        if bo:
            print(f">>> OVER recomendado (línea más alta con P ≥ {args.conf:.0%}): MÁS DE {bo['linea']} {t['unit']} — prob. {bo['prob']:.0%} | "
                  f"esta señal acertó {pct(bo['precision']).strip()} en Prueba (n={bo['n']}), recall {pct(bo['recall']).strip()}, F1 {pct(bo['f1']).strip()}")
        else:
            print(f">>> Ninguna línea OVER alcanza la confianza {args.conf:.0%}")
        if bu:
            print(f">>> UNDER recomendado (línea más baja con P ≥ {args.conf:.0%}): MENOS DE {bu['linea']} {t['unit']} — prob. {bu['prob']:.0%} | "
                  f"precisión {pct(bu['precision']).strip()}, recall {pct(bu['recall']).strip()}, F1 {pct(bu['f1']).strip()}")
    print()


if __name__ == "__main__":
    main()
