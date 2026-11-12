# Football Match Predictor

Predice el **resultado de un partido (1X2: gana local / empate / gana visitante)** y el
**total de tiros de esquina (over/under)** para 7 competiciones:

| clave CLI        | competición                 |
|------------------|-----------------------------|
| `premier_league` | Premier League (Inglaterra) |
| `la_liga`        | La Liga (España)            |
| `serie_a`        | Serie A (Italia)            |
| `bundesliga`     | Bundesliga (Alemania)       |
| `liga_mx`        | Liga MX (México)            |
| `brasileirao`    | Brasileirão Série A         |
| `champions`      | UEFA Champions League       |

## Fuente de datos

* **Principal: API pública de ESPN** (`site.api.espn.com`). Sin API key. Cada vez que se corre el
  código se descargan los meses abiertos (el mes actual y el siguiente) y el resto se lee de la caché
  local `data_cache/`. Trae, para cada partido terminado: goles, tiros, tiros a puerta, **tiros de
  esquina**, posesión y faltas. Histórico desde julio 2022 (configurable con `--start`).
* **Respaldo: football-data.co.uk** (CSV). Se usa automáticamente si ESPN falla. Tiene resultados y
  córners para Premier, La Liga, Serie A y Bundesliga, y solo resultados para Liga MX y Brasileirão
  (Champions no está disponible ahí).

## Instalación

```bash
cd "Football match predictor"
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Uso

```bash
# interactivo: elige liga, luego un partido programado (o equipos manualmente)
./.venv/bin/python predict.py

# directo
./.venv/bin/python predict.py --league la_liga --home "Real Madrid" --away Barcelona
./.venv/bin/python predict.py --league liga_mx --list-teams          # ver nombres de equipos
./.venv/bin/python predict.py --league champions --home Arsenal --away "Bayern" --neutral

# opciones útiles
#   --scope league entrena solo con la liga elegida (más rápido; por defecto se usan las 7 ligas
#                  juntas, ~9.600 partidos, que dio mejor accuracy en las pruebas)
#   --conf 0.65    confianza mínima para recomendar una línea de córners (default 0.60)
#   --refresh      ignora la caché y vuelve a descargar todo
#   --source espn|fdcouk|auto
```

Los nombres de equipo aceptan coincidencias parciales (`--home "Man City"`, `--away Chivas`).

## Qué hace el código

1. **Descarga / actualiza** los partidos de la liga (terminados y programados).
2. **Construye variables sin fuga de información**: los partidos se reproducen en orden cronológico y
   cada uno usa solo lo ocurrido antes: Elo (con ventaja de localía), forma en últimos 5 y 10
   partidos (puntos, goles, tiros, tiros a puerta, córners a favor/en contra, posesión), forma como
   local/visitante, descanso, momentum, head-to-head y córners esperados por el cruce
   ataque-defensa.
3. **Tres bloques por fecha, sin mezclar**: entrenamiento (70 % más antiguo, se ajustan los modelos),
   validación (15 % siguiente, se eligen los 2 mejores y se calibra el over/under) y prueba (15 % más
   reciente, solo se mide). Para cada modelo se reportan las métricas en los tres bloques, así se ve
   si está sobreajustando. Los modelos que predicen se reajustan después con todos los partidos.
4. **5 modelos por tarea**. Resultado 1X2: Regresión Logística, Random Forest, Extra Trees, Gradient
   Boosting y Red Neuronal, con accuracy, precision, recall, F1 y log-loss. Goles, tiros al arco y
   córners: Ridge, Random Forest, Extra Trees, Gradient Boosting Poisson y Red Neuronal, con MAE y
   RMSE del total, más accuracy, precision, recall y F1 del over/under (binomial negativa). En cada
   tarea se promedian los **2 mejores**.
5. **Predicción del partido**: probabilidades 1X2 + ganador/perdedor; total esperado de córners,
   P(over)/P(under) por línea con su accuracy/precisión histórica, y la **recomendación OVER con la
   línea más alta** cuya probabilidad supera la confianza (`--conf`), mostrando la precisión que esa
   señal tuvo en el backtest (para no "ganar" córners a costa de precisión).

## Métricas

Las métricas completas de entrenamiento, validación y prueba están en [METRICS.md](METRICS.md).
También se ven en la terminal y en la pestaña «Rendimiento de los modelos» de la página web.
Para regenerar el archivo con los datos del día:

```bash
./.venv/bin/python report.py
```

## Goles y tiros al arco

Además de los córners, el programa predice el **total de goles** y el **total de tiros al arco** del
partido, con el reparto esperado por equipo. Usa el mismo método: 5 modelos de regresión, se eligen
los 2 mejores y se calcula el over/under por línea (goles 0.5 a 4.5, tiros al arco 5.5 a 11.5) con su
accuracy, precisión y recall históricos.

## Interfaz web

```bash
./.venv/bin/streamlit run app.py
```

Para publicarla: subir el repositorio a GitHub y desplegar `app.py` en Streamlit Community Cloud.

## Notas

* Los nombres de equipos son los de ESPN (p. ej. "Club León", "Atlético Madrid").
* Si un equipo tiene menos de ~3 partidos de historial (ascensos, inicio de temporada) las variables se
  imputan y la predicción es menos confiable.
* xgboost / lightgbm no se usan porque en macOS requieren `libomp`; los 5 modelos son de scikit-learn.
