# Métricas de los modelos

Generado con datos hasta el 09/11/2026. 9,661 partidos de las 7 ligas, 117 variables. Para regenerarlo: `python report.py`.

## Cómo se entrena

Los partidos se ordenan por fecha y se dividen en tres bloques, sin mezclarlos:

| Bloque | Partidos | Desde | Hasta | Para qué se usa |
|---|---|---|---|---|
| Entrenamiento | 6,763 | 20/07/2022 | 31/08/2025 | Ajustar los modelos |
| Validación | 1,449 | 31/08/2025 | 01/03/2026 | Elegir los 2 mejores y calibrar el over/under |
| Prueba | 1,449 | 01/03/2026 | 09/11/2026 | Solo para medir. No interviene en ninguna decisión |

Si un modelo rinde mucho mejor en entrenamiento que en prueba, está memorizando (sobreajuste). La cifra honesta es la de **Prueba**. Los modelos que predicen partidos futuros se vuelven a ajustar con todos los partidos.

## Datos de jugadores y alineaciones

Además de los datos de equipo, cada partido usa su **alineación titular**. De cada jugador se llevan goles, asistencias, tiros y tiros al arco por partido de titular, con más peso en lo reciente, y un rating que sube cuando el equipo rinde mejor de lo esperado con él en cancha. El once se resume en la suma de sus jugadores y se compara con el once habitual del equipo, para medir bajas y rotaciones. Todo se calcula solo con partidos anteriores. Hay alineación en 9,651 de los 9,661 partidos.

Comparación de «Combinación top 2» entrenando sin y con los datos de jugadores. En accuracy y F1 más alto es mejor; en log-loss, MAE y RMSE más bajo es mejor.

| Tarea | Conjunto | Datos | Accuracy | F1 | Log-loss | MAE | RMSE |
|---|---|---|---|---|---|---|---|
| Resultado 1X2 | Validación | Solo equipos | 52.7% | 40.6% | 0.988 | n/a | n/a |
| Resultado 1X2 | Validación | Equipos + jugadores | 52.9% | 40.4% | 0.988 | n/a | n/a |
| Resultado 1X2 | Prueba | Solo equipos | 51.3% | 39.1% | 1.000 | n/a | n/a |
| Resultado 1X2 | Prueba | Equipos + jugadores | 50.6% | 38.3% | 0.999 | n/a | n/a |
| Goles | Validación | Solo equipos | 68.0% | 51.3% | n/a | 1.291 | 1.623 |
| Goles | Validación | Equipos + jugadores | 68.6% | 52.0% | n/a | 1.287 | 1.617 |
| Goles | Prueba | Solo equipos | 67.7% | 51.2% | n/a | 1.337 | 1.683 |
| Goles | Prueba | Equipos + jugadores | 68.3% | 52.8% | n/a | 1.334 | 1.677 |
| Tiros al arco | Validación | Solo equipos | 67.2% | 48.2% | n/a | 2.403 | 3.051 |
| Tiros al arco | Validación | Equipos + jugadores | 67.3% | 48.4% | n/a | 2.395 | 3.049 |
| Tiros al arco | Prueba | Solo equipos | 65.7% | 52.1% | n/a | 2.619 | 3.269 |
| Tiros al arco | Prueba | Equipos + jugadores | 65.8% | 53.0% | n/a | 2.603 | 3.246 |
| Tiros de esquina | Validación | Solo equipos | 64.7% | 45.4% | n/a | 2.687 | 3.408 |
| Tiros de esquina | Validación | Equipos + jugadores | 65.1% | 42.2% | n/a | 2.676 | 3.377 |
| Tiros de esquina | Prueba | Solo equipos | 63.4% | 52.3% | n/a | 2.890 | 3.585 |
| Tiros de esquina | Prueba | Equipos + jugadores | 64.3% | 49.1% | n/a | 2.847 | 3.533 |

## Resultado del partido (1X2)

Mejores dos en validación: **Random Forest** y **Gradient Boosting**. «Combinación top 2» es el promedio de ambos y es lo que usa la predicción. Precision, recall y F1 son el promedio de las tres clases (local, empate, visitante).

| Modelo | Conjunto | Accuracy | Precision | Recall | F1 | Log-loss | Score |
|---|---|---|---|---|---|---|---|
| Random Forest | Entrenamiento | 91.4% | 93.5% | 89.9% | 91.3% | 0.591 | 91.6% |
| Random Forest | Validación | 54.2% | 45.6% | 46.7% | 40.7% | 0.985 | 48.8% |
| Random Forest | Prueba | 50.7% | 36.2% | 43.1% | 37.5% | 0.996 | 43.3% |
| Gradient Boosting | Entrenamiento | 61.2% | 68.2% | 54.7% | 53.0% | 0.866 | 61.4% |
| Gradient Boosting | Validación | 52.2% | 47.3% | 45.8% | 41.7% | 0.997 | 48.4% |
| Gradient Boosting | Prueba | 50.2% | 43.2% | 43.8% | 40.0% | 1.009 | 45.7% |
| Regresión Logística | Entrenamiento | 54.1% | 49.6% | 47.1% | 43.6% | 0.965 | 50.3% |
| Regresión Logística | Validación | 52.3% | 45.3% | 46.6% | 42.7% | 1.000 | 48.1% |
| Regresión Logística | Prueba | 50.4% | 46.0% | 44.8% | 41.7% | 1.005 | 47.1% |
| Extra Trees | Entrenamiento | 86.6% | 90.3% | 83.9% | 86.0% | 0.651 | 86.9% |
| Extra Trees | Validación | 54.4% | 42.1% | 46.4% | 40.2% | 0.980 | 47.6% |
| Extra Trees | Prueba | 50.9% | 33.2% | 42.9% | 37.2% | 0.994 | 42.3% |
| Red Neuronal (MLP) | Entrenamiento | 58.4% | 56.7% | 52.9% | 51.8% | 0.902 | 56.0% |
| Red Neuronal (MLP) | Validación | 49.8% | 44.0% | 45.1% | 43.6% | 1.010 | 46.3% |
| Red Neuronal (MLP) | Prueba | 49.3% | 44.5% | 44.7% | 44.0% | 1.026 | 46.2% |
| Combinación top 2 | Entrenamiento | 76.9% | 82.9% | 72.7% | 74.6% | 0.714 | 77.5% |
| Combinación top 2 | Validación | 52.9% | 44.8% | 45.8% | 40.4% | 0.988 | 47.8% |
| Combinación top 2 | Prueba | 50.6% | 41.5% | 43.4% | 38.3% | 0.999 | 45.2% |

## Goles

Mejores dos en validación: **Extra Trees** y **Random Forest**. MAE y RMSE miden el error en goles por partido. Accuracy, precision, recall y F1 son del over/under, promediados en las líneas 1.5 a 3.5.

| Modelo | Conjunto | MAE | RMSE | Accuracy | Precision | Recall | F1 | Score |
|---|---|---|---|---|---|---|---|---|
| Extra Trees | Entrenamiento | 0.984 | 1.245 | 79.4% | 86.7% | 69.6% | 69.2% | 78.6% |
| Extra Trees | Validación | 1.285 | 1.616 | 68.3% | 66.2% | 56.7% | 51.5% | 63.7% |
| Extra Trees | Prueba | 1.333 | 1.679 | 68.2% | 76.6% | 57.4% | 52.3% | 67.4% |
| Random Forest | Entrenamiento | 1.067 | 1.355 | 77.1% | 83.0% | 65.9% | 63.3% | 75.3% |
| Random Forest | Validación | 1.291 | 1.623 | 68.4% | 62.4% | 57.2% | 52.0% | 62.7% |
| Random Forest | Prueba | 1.339 | 1.679 | 68.2% | 71.1% | 59.6% | 53.5% | 66.3% |
| Gradient Boosting (Poisson) | Entrenamiento | 1.238 | 1.549 | 70.9% | 74.4% | 59.0% | 55.6% | 68.1% |
| Gradient Boosting (Poisson) | Validación | 1.289 | 1.622 | 67.9% | 63.5% | 54.5% | 50.5% | 62.0% |
| Gradient Boosting (Poisson) | Prueba | 1.342 | 1.689 | 68.0% | 62.8% | 56.8% | 52.2% | 62.6% |
| Ridge (lineal) | Entrenamiento | 1.292 | 1.622 | 68.3% | 63.9% | 56.2% | 52.0% | 62.8% |
| Ridge (lineal) | Validación | 1.289 | 1.622 | 68.1% | 60.4% | 56.7% | 52.8% | 61.7% |
| Ridge (lineal) | Prueba | 1.332 | 1.676 | 68.2% | 67.3% | 58.5% | 55.2% | 64.7% |
| Red Neuronal (MLP) | Entrenamiento | 1.254 | 1.574 | 69.7% | 65.5% | 60.0% | 58.5% | 65.1% |
| Red Neuronal (MLP) | Validación | 1.334 | 1.673 | 66.9% | 58.7% | 58.7% | 56.2% | 61.4% |
| Red Neuronal (MLP) | Prueba | 1.385 | 1.728 | 66.9% | 60.4% | 65.5% | 61.4% | 64.3% |
| Combinación top 2 | Entrenamiento | 1.024 | 1.298 | 78.4% | 85.5% | 67.7% | 66.1% | 77.2% |
| Combinación top 2 | Validación | 1.287 | 1.617 | 68.6% | 66.8% | 57.1% | 52.0% | 64.2% |
| Combinación top 2 | Prueba | 1.334 | 1.677 | 68.3% | 72.6% | 58.4% | 52.8% | 66.4% |

Over/under por línea de «Combinación top 2» en el bloque de prueba (1,449 partidos, promedio real 2.91 goles). «Señales» son los partidos con P(over) de al menos 60%.

| Línea | % over real | Accuracy | Prec. over | Recall over | F1 over | Prec. under | Recall under | F1 under | Señales | Prec. señal |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 94.3% | 94.3% | 94.3% | 100.0% | 97.1% | 0.0% | 0.0% | 0.0% | 1449 | 94.3% |
| 1.5 | 78.7% | 78.7% | 78.7% | 100.0% | 88.1% | 0.0% | 0.0% | 0.0% | 1449 | 78.7% |
| 2.5 | 55.4% | 57.0% | 59.1% | 72.6% | 65.2% | 52.5% | 37.6% | 43.8% | 312 | 66.3% |
| 3.5 | 31.5% | 69.1% | 80.0% | 2.6% | 5.1% | 69.0% | 99.7% | 81.5% | 0 | n/a |
| 4.5 | 18.4% | 81.6% | 0.0% | 0.0% | 0.0% | 81.6% | 100.0% | 89.9% | 0 | n/a |

## Tiros al arco

Mejores dos en validación: **Gradient Boosting (Poisson)** y **Random Forest**. MAE y RMSE miden el error en tiros al arco por partido. Accuracy, precision, recall y F1 son del over/under, promediados en las líneas 6.5 a 10.5.

| Modelo | Conjunto | MAE | RMSE | Accuracy | Precision | Recall | F1 | Score |
|---|---|---|---|---|---|---|---|---|
| Gradient Boosting (Poisson) | Entrenamiento | 2.507 | 3.246 | 69.0% | 73.0% | 53.0% | 49.9% | 65.0% |
| Gradient Boosting (Poisson) | Validación | 2.394 | 3.052 | 67.4% | 69.3% | 53.0% | 49.1% | 63.2% |
| Gradient Boosting (Poisson) | Prueba | 2.613 | 3.262 | 65.9% | 72.8% | 57.4% | 53.0% | 65.4% |
| Random Forest | Entrenamiento | 2.153 | 2.826 | 76.2% | 83.2% | 61.7% | 61.5% | 73.7% |
| Random Forest | Validación | 2.401 | 3.055 | 67.4% | 69.5% | 52.6% | 48.7% | 63.2% |
| Random Forest | Prueba | 2.597 | 3.239 | 65.9% | 69.1% | 57.4% | 53.4% | 64.1% |
| Extra Trees | Entrenamiento | 1.983 | 2.589 | 79.2% | 86.5% | 65.6% | 66.9% | 77.1% |
| Extra Trees | Validación | 2.390 | 3.043 | 67.6% | 65.5% | 52.9% | 49.0% | 62.0% |
| Extra Trees | Prueba | 2.605 | 3.256 | 65.9% | 71.8% | 56.1% | 52.2% | 64.6% |
| Ridge (lineal) | Entrenamiento | 2.653 | 3.403 | 67.0% | 62.5% | 52.0% | 51.4% | 60.5% |
| Ridge (lineal) | Validación | 2.424 | 3.064 | 66.7% | 64.0% | 52.6% | 51.8% | 61.1% |
| Ridge (lineal) | Prueba | 2.594 | 3.223 | 66.0% | 67.2% | 59.3% | 57.4% | 64.2% |
| Red Neuronal (MLP) | Entrenamiento | 2.510 | 3.226 | 68.6% | 66.9% | 52.5% | 55.3% | 62.7% |
| Red Neuronal (MLP) | Validación | 2.542 | 3.203 | 64.8% | 58.5% | 51.7% | 52.8% | 58.3% |
| Red Neuronal (MLP) | Prueba | 2.705 | 3.372 | 64.5% | 63.7% | 60.4% | 60.2% | 62.9% |
| Combinación top 2 | Entrenamiento | 2.327 | 3.030 | 72.7% | 79.5% | 57.2% | 55.3% | 69.8% |
| Combinación top 2 | Validación | 2.395 | 3.049 | 67.3% | 68.5% | 52.6% | 48.4% | 62.8% |
| Combinación top 2 | Prueba | 2.603 | 3.246 | 65.8% | 69.2% | 57.2% | 53.0% | 64.1% |

Over/under por línea de «Combinación top 2» en el bloque de prueba (1,449 partidos, promedio real 9.37 tiros al arco). «Señales» son los partidos con P(over) de al menos 60%.

| Línea | % over real | Accuracy | Prec. over | Recall over | F1 over | Prec. under | Recall under | F1 under | Señales | Prec. señal |
|---|---|---|---|---|---|---|---|---|---|---|
| 5.5 | 88.5% | 88.5% | 88.5% | 100.0% | 93.9% | 0.0% | 0.0% | 0.0% | 1449 | 88.5% |
| 6.5 | 80.0% | 80.0% | 80.0% | 100.0% | 88.9% | 0.0% | 0.0% | 0.0% | 1443 | 80.1% |
| 7.5 | 68.8% | 68.8% | 69.2% | 98.4% | 81.3% | 50.0% | 3.5% | 6.6% | 1135 | 73.0% |
| 8.5 | 57.8% | 58.1% | 62.3% | 69.8% | 65.8% | 50.4% | 42.1% | 45.9% | 273 | 73.3% |
| 9.5 | 46.1% | 57.3% | 64.7% | 16.5% | 26.3% | 56.4% | 92.3% | 70.0% | 16 | 75.0% |
| 10.5 | 35.3% | 64.9% | 70.0% | 1.4% | 2.7% | 64.9% | 99.7% | 78.6% | 0 | n/a |
| 11.5 | 26.2% | 73.8% | 0.0% | 0.0% | 0.0% | 73.8% | 100.0% | 84.9% | 0 | n/a |

## Tiros de esquina

Mejores dos en validación: **Ridge (lineal)** y **Random Forest**. MAE y RMSE miden el error en córners por partido. Accuracy, precision, recall y F1 son del over/under, promediados en las líneas 7.5 a 11.5.

| Modelo | Conjunto | MAE | RMSE | Accuracy | Precision | Recall | F1 | Score |
|---|---|---|---|---|---|---|---|---|
| Ridge (lineal) | Entrenamiento | 2.962 | 3.780 | 64.7% | 59.9% | 48.3% | 46.6% | 57.6% |
| Ridge (lineal) | Validación | 2.711 | 3.411 | 64.7% | 59.5% | 48.5% | 43.5% | 57.5% |
| Ridge (lineal) | Prueba | 2.867 | 3.544 | 63.6% | 56.4% | 57.1% | 51.4% | 59.0% |
| Random Forest | Entrenamiento | 2.387 | 3.107 | 74.8% | 85.7% | 59.3% | 58.2% | 73.3% |
| Random Forest | Validación | 2.665 | 3.376 | 65.0% | 48.5% | 48.4% | 41.5% | 54.0% |
| Random Forest | Prueba | 2.849 | 3.550 | 63.8% | 69.8% | 52.7% | 46.5% | 62.1% |
| Extra Trees | Entrenamiento | 2.201 | 2.853 | 78.3% | 88.2% | 64.2% | 65.0% | 76.9% |
| Extra Trees | Validación | 2.659 | 3.367 | 65.0% | 46.7% | 48.8% | 41.4% | 53.5% |
| Extra Trees | Prueba | 2.839 | 3.549 | 64.1% | 50.7% | 52.7% | 45.8% | 55.8% |
| Red Neuronal (MLP) | Entrenamiento | 2.792 | 3.575 | 66.6% | 63.8% | 50.6% | 52.1% | 60.3% |
| Red Neuronal (MLP) | Validación | 2.791 | 3.534 | 62.5% | 51.3% | 46.6% | 46.0% | 53.5% |
| Red Neuronal (MLP) | Prueba | 3.002 | 3.716 | 61.4% | 57.2% | 54.0% | 53.0% | 57.5% |
| Gradient Boosting (Poisson) | Entrenamiento | 2.789 | 3.581 | 66.7% | 76.4% | 50.2% | 45.1% | 64.4% |
| Gradient Boosting (Poisson) | Validación | 2.666 | 3.378 | 65.0% | 43.5% | 48.4% | 41.2% | 52.3% |
| Gradient Boosting (Poisson) | Prueba | 2.848 | 3.561 | 63.7% | 48.2% | 52.1% | 45.2% | 54.7% |
| Combinación top 2 | Entrenamiento | 2.664 | 3.425 | 70.0% | 76.3% | 53.8% | 51.7% | 66.7% |
| Combinación top 2 | Validación | 2.676 | 3.377 | 65.1% | 48.3% | 48.7% | 42.2% | 54.0% |
| Combinación top 2 | Prueba | 2.847 | 3.533 | 64.3% | 60.1% | 55.2% | 49.1% | 59.9% |

Over/under por línea de «Combinación top 2» en el bloque de prueba (1,449 partidos, promedio real 10.07 córners). «Señales» son los partidos con P(over) de al menos 60%.

| Línea | % over real | Accuracy | Prec. over | Recall over | F1 over | Prec. under | Recall under | F1 under | Señales | Prec. señal |
|---|---|---|---|---|---|---|---|---|---|---|
| 6.5 | 85.0% | 85.0% | 85.0% | 100.0% | 91.9% | 0.0% | 0.0% | 0.0% | 1449 | 85.0% |
| 7.5 | 74.9% | 74.9% | 74.9% | 99.9% | 85.6% | 0.0% | 0.0% | 0.0% | 1431 | 75.2% |
| 8.5 | 63.7% | 63.9% | 64.2% | 97.7% | 77.5% | 53.3% | 4.6% | 8.4% | 1028 | 67.4% |
| 9.5 | 53.1% | 56.8% | 58.2% | 66.1% | 61.9% | 54.7% | 46.3% | 50.2% | 190 | 58.9% |
| 10.5 | 41.8% | 58.8% | 52.9% | 12.1% | 19.7% | 59.4% | 92.3% | 72.3% | 6 | 83.3% |
| 11.5 | 32.8% | 67.2% | 50.0% | 0.4% | 0.8% | 67.3% | 99.8% | 80.4% | 0 | n/a |
| 12.5 | 25.3% | 74.7% | 0.0% | 0.0% | 0.0% | 74.7% | 100.0% | 85.5% | 0 | n/a |
| 13.5 | 17.9% | 82.1% | 0.0% | 0.0% | 0.0% | 82.1% | 100.0% | 90.1% | 0 | n/a |
