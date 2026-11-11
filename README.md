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
3. **Resultado 1X2 – 5 modelos**: Regresión Logística, Random Forest, Extra Trees, Gradient
   Boosting y Red Neuronal (MLP). Se entrenan con el 80 % más antiguo y se evalúan en el 20 % más
   reciente (accuracy, precision macro, recall macro, F1, log-loss). Se eligen los **2 mejores** por
   el promedio de accuracy + precision + recall y se promedian sus probabilidades para la predicción.
4. **Córners – 5 modelos** de regresión del total (Ridge, Random Forest, Extra Trees, Gradient
   Boosting Poisson, MLP). La media predicha se convierte en probabilidad de over/under por línea
   (6.5 … 13.5) con una binomial negativa cuya dispersión se calibra out-of-fold. Cada modelo se
   evalúa como clasificador over/under en las líneas 7.5–11.5 y se eligen los **2 mejores**.
5. **Predicción del partido**: probabilidades 1X2 + ganador/perdedor; total esperado de córners,
   P(over)/P(under) por línea con su accuracy/precisión histórica, y la **recomendación OVER con la
   línea más alta** cuya probabilidad supera la confianza (`--conf`), mostrando la precisión que esa
   señal tuvo en el backtest (para no "ganar" córners a costa de precisión).

## Notas

* Los nombres de equipos son los de ESPN (p. ej. "Club León", "Atlético Madrid").
* Si un equipo tiene menos de ~3 partidos de historial (ascensos, inicio de temporada) las variables se
  imputan y la predicción es menos confiable.
* xgboost / lightgbm no se usan porque en macOS requieren `libomp`; los 5 modelos son de scikit-learn.
