"""Football Match Predictor - interfaz web (Streamlit).

Local:   streamlit run app.py
Online:  Streamlit Community Cloud apuntando a este archivo.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from fmp.data import LEAGUES, current_teams, upcoming_fixtures
from fmp.models import ENSEMBLE, OUTCOME_LABELS, SPLITS
from fmp.pipeline import HEADERS, PCT, TARGETS, count_view, load_data, outcome_view, predict, train

st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="wide")


@st.cache_resource(ttl=6 * 3600, show_spinner=False)
def get_state() -> dict:
    """Download the latest data and train the models (cached for 6 hours)."""
    data = load_data()
    return {"data": data, "tr": train(data), "loaded_at": pd.Timestamp.now("UTC")}


def show(df: pd.DataFrame) -> None:
    """Render a metrics table: percentages with one decimal, errors with three, Spanish headers."""
    out, cfg = df.copy(), {}
    for c in out.columns:
        label = HEADERS.get(c, c)
        if c in PCT or c in ("P(Over)", "P(Under)", "Accuracy", "Precisión", "Recall", "F1"):
            out[c] = (out[c] * 100).round(1)
            cfg[label] = st.column_config.NumberColumn(format="%.1f%%")
        elif c in ("MAE", "RMSE", "log_loss"):
            cfg[label] = st.column_config.NumberColumn(format="%.3f")
        elif c in ("linea", "Línea"):
            cfg[label] = st.column_config.NumberColumn(format="%.1f")
    st.dataframe(out.rename(columns=HEADERS), hide_index=True, width="stretch", column_config=cfg)


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

tab_pred, tab_perf = st.tabs(["Predicción", "Rendimiento de los modelos"])

# -------------------------------------------------------------- performance --
with tab_perf:
    st.subheader("Cómo se entrenan y qué tan bien funcionan")
    st.write(f"Los {tr['n_matches']:,} partidos se ordenan por fecha y se dividen en tres bloques que nunca se mezclan. "
             "Cada tarea prueba 5 modelos y usa el promedio de los 2 mejores.")
    uses = {"train": "Ajustar los modelos", "val": "Elegir los 2 mejores y calibrar el over/under",
            "test": "Solo medir. No interviene en ninguna decisión"}
    st.dataframe(pd.DataFrame([{"Bloque": name, "Partidos": tr["splits"][k]["n"],
                                "Desde": f"{tr['splits'][k]['desde']:%d/%m/%Y}", "Hasta": f"{tr['splits'][k]['hasta']:%d/%m/%Y}",
                                "Para qué se usa": uses[k]} for k, name in SPLITS.items()]),
                 hide_index=True, width="stretch")
    st.info("Compara las filas de **Entrenamiento** y **Prueba** de cada modelo. Si la de entrenamiento es mucho mejor, "
            "el modelo está memorizando (sobreajuste). La cifra honesta es la de **Prueba**. "
            "Los modelos que predicen partidos futuros se vuelven a ajustar con todos los partidos.")
    keys = [k for k, ct in tr["counts"].items() if ct]
    perf_tabs = st.tabs(["Resultado 1X2"] + [TARGETS[k]["label"] for k in keys])
    with perf_tabs[0]:
        st.write(f"Mejores dos en validación: **{tr['top2'][0]}** y **{tr['top2'][1]}**. «{ENSEMBLE}» es el promedio de ambos "
                 "y es lo que usa la predicción. Precision, recall y F1 son el promedio de las tres clases.")
        show(tr["metrics"].drop(columns="score"))
        ov = outcome_view(tr, league)
        where = LEAGUES[league]["name"] if ov["league_specific"] else "las 7 ligas"
        st.write(f"Bloque de prueba, solo partidos de {where} ({ov['n']} partidos):")
        show(ov["table"].drop(columns="score"))
    for tab, key in zip(perf_tabs[1:], keys):
        t, ct = TARGETS[key], tr["counts"][key]
        with tab:
            st.write(f"Mejores dos en validación: **{ct['top2'][0]}** y **{ct['top2'][1]}**. MAE y RMSE miden el error en "
                     f"{t['unit']} por partido. Accuracy, precision, recall y F1 son del over/under, promediados en las "
                     f"líneas {t['rank'][0]} a {t['rank'][-1]}.")
            show(ct["metrics"].drop(columns="score"))
            v = count_view(ct, league, conf)
            where = LEAGUES[league]["name"] if v["league_specific"] else "las 7 ligas"
            st.write(f"Over/under por línea de «{ENSEMBLE}» en el bloque de prueba de {where} ({v['n']} partidos, "
                     f"promedio real {v['mean_real']:.2f} {t['unit']}). «Señales» son los partidos con P(over) de al menos {conf:.0%}.")
            show(v["bt"])

# --------------------------------------------------------------- prediction --
FORM = {"goals": ("gf", "ga"), "sot": ("stf", "sta"), "corners": ("cf", "ca")}


def render_prediction() -> None:
    if go:
        st.session_state["query"] = (league, home, away, kickoff, neutral)
    query = st.session_state.get("query")
    if not query or query[0] != league:
        st.info("Elige la liga y el partido en el panel de la izquierda y presiona **Predecir**.")
        return
    _, q_home, q_away, q_kickoff, q_neutral = query
    try:
        res = predict(tr, league, q_home, q_away, q_kickoff, q_neutral, conf)
    except ValueError as exc:
        st.warning(str(exc))
        return
    proba, feats = res["proba"], res["feats"]

    st.subheader(f"{q_home} vs {q_away}")
    st.caption(f"{LEAGUES[league]['name']} · fecha usada: {q_kickoff:%d/%m/%Y}{' · cancha neutral' if q_neutral else ''}")
    winner = int(proba.argmax())
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Gana {q_home}", f"{proba[0]:.1%}")
    c2.metric("Empate", f"{proba[1]:.1%}")
    c3.metric(f"Gana {q_away}", f"{proba[2]:.1%}")
    if winner == 1:
        st.success(f"**Resultado más probable: empate** ({proba[1]:.1%})")
    else:
        win, lose = (q_home, q_away) if winner == 0 else (q_away, q_home)
        st.success(f"**Ganador probable: {win}** · perdedor: {lose} · probabilidad {proba[winner]:.1%} ({OUTCOME_LABELS[winner].lower()})")
    no_draw = proba[0] / (proba[0] + proba[2])
    st.caption(f"Sin contar el empate: {q_home} {no_draw:.1%} · {q_away} {1 - no_draw:.1%}. "
               f"Elo: {q_home} {res['elo_home']:.0f}, {q_away} {res['elo_away']:.0f}. "
               f"Puntos por partido en los últimos 5: {q_home} {feats['h_pts_s']:.2f}, {q_away} {feats['a_pts_s']:.2f}.")

    for key, r in res["counts"].items():
        t = TARGETS[key]
        st.subheader(t["label"])
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total esperado", f"{r['mu']:.1f}")
        k2.metric("Por equipo", f"{r['mu_home']:.1f} - {r['mu_away']:.1f}", f"{q_home} - {q_away}", delta_color="off")
        bo, bu = r["best_over"], r["best_under"]
        k3.metric("Over recomendado", f"Más de {bo['linea']}" if bo else "Sin señal",
                  f"{bo['prob']:.0%} de probabilidad" if bo else None, delta_color="off")
        k4.metric("Under recomendado", f"Menos de {bu['linea']}" if bu else "Sin señal",
                  f"{bu['prob']:.0%} de probabilidad" if bu else None, delta_color="off")
        if bo:
            prec = "sin casos" if pd.isna(bo["precision"]) else f"{bo['precision']:.1%}"
            st.info(f"**Más de {bo['linea']} {t['unit']}** es la línea más alta con probabilidad de al menos {conf:.0%}. "
                    f"En el bloque de prueba esta señal acertó {prec} de las veces en {bo['n']} partidos. "
                    f"Accuracy de la línea {bo['accuracy']:.1%}, recall {bo['recall']:.1%}, F1 {bo['f1']:.1%}.")
        else:
            st.info(f"Ninguna línea over alcanza {conf:.0%} de probabilidad. Baja la confianza en el panel para ver una recomendación.")
        f_, a_ = FORM[key]
        st.caption(f"Promedio en los últimos 10 partidos: {q_home} {feats[f'h_{f_}_l']:.1f} a favor y {feats[f'h_{a_}_l']:.1f} en contra, "
                   f"{q_away} {feats[f'a_{f_}_l']:.1f} a favor y {feats[f'a_{a_}_l']:.1f} en contra. "
                   "Accuracy, precisión, recall y F1 de cada línea vienen del bloque de prueba.")
        show(r["lines"])
    st.caption("Las métricas completas de entrenamiento, validación y prueba están en la pestaña «Rendimiento de los modelos».")


with tab_pred:
    render_prediction()
st.caption("Modelo estadístico con fines informativos. No garantiza resultados.")
