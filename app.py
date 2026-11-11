"""Football Match Predictor - interfaz web (Streamlit).

Local:   streamlit run app.py
Online:  Streamlit Community Cloud apuntando a este archivo.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from fmp.data import LEAGUES, current_teams, upcoming_fixtures
from fmp.models import OUTCOME_LABELS
from fmp.pipeline import TARGETS, load_data, predict, train

st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="wide")


@st.cache_resource(ttl=6 * 3600, show_spinner=False)
def get_state() -> dict:
    """Download the latest data and train the models (cached for 6 hours)."""
    data = load_data()
    return {"data": data, "tr": train(data), "loaded_at": pd.Timestamp.now("UTC")}


def pct_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = (out[c] * 100).round(1)
    return out


st.title("⚽ Football Match Predictor")
st.caption("Predicción del resultado (1X2), goles, tiros al arco y tiros de esquina (over/under) con datos actualizados de ESPN.")

with st.spinner("Descargando datos actualizados y entrenando los modelos (1 a 4 minutos la primera vez)…"):
    try:
        state = get_state()
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudieron cargar los datos: {exc}")
        st.stop()
data, tr = state["data"], state["tr"]

# ------------------------------------------------------------------ sidebar --
with st.sidebar:
    st.header("Partido")
    league = st.selectbox("Liga", list(LEAGUES), format_func=lambda k: LEAGUES[k]["name"])
    df_league = data[league]
    teams = current_teams(df_league)
    fixtures = upcoming_fixtures(df_league)
    modes = ["Elegir equipos"] + (["Próximos partidos"] if len(fixtures) else [])
    mode = st.radio("Cómo elegir", modes[::-1], horizontal=True)
    kickoff, neutral = pd.Timestamp.now("UTC").tz_convert(None), False
    if mode == "Próximos partidos":
        labels = [f"{r.date:%d/%m %H:%M} UTC · {r.home} vs {r.away}" for r in fixtures.itertuples()]
        idx = st.selectbox("Partido programado", range(len(labels)), format_func=lambda i: labels[i])
        row = fixtures.iloc[idx]
        home, away, kickoff, neutral = row["home"], row["away"], row["date"], bool(row["neutral"])
    else:
        home = st.selectbox("Equipo local", teams)
        away = st.selectbox("Equipo visitante", [t for t in teams if t != home])
        neutral = st.checkbox("Cancha neutral")
    conf = st.slider("Confianza mínima para recomendar una línea over/under", 0.50, 0.80, 0.60, 0.01)
    go = st.button("Predecir", type="primary", width="stretch")
    st.divider()
    st.caption(f"Último partido en los datos: {tr['last_match']:%d/%m/%Y}. "
               f"Modelos entrenados con {tr['n_matches']:,} partidos de las 7 ligas.")
    if st.button("Actualizar datos ahora", width="stretch"):
        get_state.clear()
        st.rerun()

if go:
    st.session_state["query"] = (league, home, away, kickoff, neutral)
query = st.session_state.get("query")
if not query or query[0] != league:
    st.info("Elige la liga y el partido en el panel de la izquierda y presiona **Predecir**.")
    st.stop()
league, home, away, kickoff, neutral = query

try:
    res = predict(tr, league, home, away, kickoff, neutral, conf)
except ValueError as exc:
    st.warning(str(exc))
    st.stop()
proba, feats = res["proba"], res["feats"]

# ------------------------------------------------------------------- result --
st.subheader(f"{home} vs {away}")
st.caption(f"{LEAGUES[league]['name']} · fecha usada: {kickoff:%d/%m/%Y}{' · cancha neutral' if neutral else ''}")

winner = int(proba.argmax())
c1, c2, c3 = st.columns(3)
c1.metric(f"Gana {home}", f"{proba[0]:.1%}")
c2.metric("Empate", f"{proba[1]:.1%}")
c3.metric(f"Gana {away}", f"{proba[2]:.1%}")
if winner == 1:
    st.success(f"**Resultado más probable: empate** ({proba[1]:.1%})")
else:
    win, lose = (home, away) if winner == 0 else (away, home)
    st.success(f"**Ganador probable: {win}** · perdedor: {lose} · probabilidad {proba[winner]:.1%} ({OUTCOME_LABELS[winner].lower()})")
no_draw = proba[0] / (proba[0] + proba[2])
st.caption(f"Sin contar el empate: {home} {no_draw:.1%} · {away} {1 - no_draw:.1%}. "
           f"Elo: {home} {res['elo_home']:.0f}, {away} {res['elo_away']:.0f}. "
           f"Puntos por partido en los últimos 5: {home} {feats['h_pts_s']:.2f}, {away} {feats['a_pts_s']:.2f}.")

# ------------------------------------------------- goals / shots / corners --
FORM = {"goals": ("gf", "ga"), "sot": ("stf", "sta"), "corners": ("cf", "ca")}
pcols = ["P(Over)", "P(Under)", "Accuracy histórica", "Precisión histórica", "Recall histórico"]
for key, r in res["counts"].items():
    t = TARGETS[key]
    st.subheader(t["label"])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total esperado", f"{r['mu']:.1f}")
    k2.metric("Por equipo", f"{r['mu_home']:.1f} - {r['mu_away']:.1f}", f"{home} - {away}", delta_color="off")
    bo, bu = r["best_over"], r["best_under"]
    k3.metric("Over recomendado", f"Más de {bo['linea']}" if bo else "Sin señal", f"{bo['prob']:.0%} de probabilidad" if bo else None, delta_color="off")
    k4.metric("Under recomendado", f"Menos de {bu['linea']}" if bu else "Sin señal", f"{bu['prob']:.0%} de probabilidad" if bu else None, delta_color="off")
    if bo:
        prec = "sin casos" if pd.isna(bo["precision"]) else f"{bo['precision']:.1%}"
        st.info(f"**Más de {bo['linea']} {t['unit']}** es la línea más alta con probabilidad de al menos {conf:.0%}. "
                f"En el backtest esta señal acertó {prec} de las veces en {bo['n']} partidos. "
                f"Accuracy de la línea {bo['accuracy']:.1%}, recall {bo['recall']:.1%}.")
    else:
        st.info(f"Ninguna línea over alcanza {conf:.0%} de probabilidad. Baja la confianza en el panel para ver una recomendación.")
    f_, a_ = FORM[key]
    st.caption(f"Promedio en los últimos 10 partidos: {home} {feats[f'h_{f_}_l']:.1f} a favor y {feats[f'h_{a_}_l']:.1f} en contra, "
               f"{away} {feats[f'a_{f_}_l']:.1f} a favor y {feats[f'a_{a_}_l']:.1f} en contra.")
    st.dataframe(pct_table(r["lines"], pcols), hide_index=True, width="stretch",
                 column_config={"Línea": st.column_config.NumberColumn(format="%.1f"),
                                **{c: st.column_config.NumberColumn(format="%.1f%%") for c in pcols}})

# ------------------------------------------------------------------- models --
with st.expander("Rendimiento de los 5 modelos de resultado (1X2)"):
    st.write(f"Mejores dos, usados en la predicción: **{tr['top2'][0]}** y **{tr['top2'][1]}**. "
             f"Evaluación en los {tr['n_test']:,} partidos más recientes de las 7 ligas, desde {tr['test_from']:%d/%m/%Y}.")
    mcols = ["accuracy", "precision", "recall", "f1", "score"]
    cfg = {c: st.column_config.NumberColumn(format="%.1f%%") for c in mcols}
    st.dataframe(pct_table(tr["table"], mcols), hide_index=True, width="stretch", column_config=cfg)
    st.write(f"Los mismos modelos, solo en partidos de {LEAGUES[league]['name']} ({res['n_outcome']} partidos):")
    st.dataframe(pct_table(res["outcome_table"], mcols), hide_index=True, width="stretch", column_config=cfg)
for key, r in res["counts"].items():
    t, ct, view = TARGETS[key], tr["counts"][key], r["view"]
    with st.expander(f"Rendimiento de los 5 modelos de {t['label'].lower()}"):
        st.write(f"Mejores dos, usados en la predicción: **{ct['top2'][0]}** y **{ct['top2'][1]}**. Accuracy, precision y recall "
                 f"son el promedio del over/under en las líneas {t['rank'][0]} a {t['rank'][-1]}.")
        ccols = ["accuracy", "precision", "recall", "score"]
        st.dataframe(pct_table(ct["table"], ccols), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.1f%%") for c in ccols})
        scope = LEAGUES[league]["name"] if view["league_specific"] else "las 7 ligas"
        st.write(f"Backtest de la combinación de los dos mejores en {scope} "
                 f"({view['n']} partidos, promedio real {view['mean_real']:.2f} {t['unit']}):")
        bcols = ["pct_over_real", "accuracy", "precision_over", "recall_over", "precision_under", "recall_under", "senal_precision"]
        st.dataframe(pct_table(view["bt"], bcols), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.1f%%") for c in bcols})
st.caption("Modelo estadístico con fines informativos. No garantiza resultados.")
