# Métricas de los modelos

Generado con datos hasta el 09/11/2026. 9,661 partidos de las 7 ligas, 89 variables. Para regenerarlo: `python report.py`.

## Cómo se entrena

Los partidos se ordenan por fecha y se dividen en tres bloques, sin mezclarlos:

| Bloque | Partidos | Desde | Hasta | Para qué se usa |
|---|---|---|---|---|
| Entrenamiento | 6,763 | 20/07/2022 | 31/08/2025 | Ajustar los modelos |
| Validación | 1,449 | 31/08/2025 | 01/03/2026 | Elegir los 2 mejores y calibrar el over/under |
| Prueba | 1,449 | 01/03/2026 | 09/11/2026 | Solo para medir. No interviene en ninguna decisión |

Si un modelo rinde mucho mejor en entrenamiento que en prueba, está memorizando (sobreajuste). La cifra honesta es la de **Prueba**. Los modelos que predicen partidos futuros se vuelven a ajustar con todos los partidos.

## Resultado del partido (1X2)

Mejores dos en validación: **Regresión Logística** y **Random Forest**. «Combinación top 2» es el promedio de ambos y es lo que usa la predicción. Precision, recall y F1 son el promedio de las tres clases (local, empate, visitante).

| Modelo | Conjunto | Accuracy | Precision | Recall | F1 | Log-loss | Score |
|---|---|---|---|---|---|---|---|
| Regresión Logística | Entrenamiento | 53.1% | 49.2% | 46.0% | 42.3% | 0.974 | 49.4% |
| Regresión Logística | Validación | 52.3% | 47.1% | 46.0% | 42.0% | 1.000 | 48.5% |
| Regresión Logística | Prueba | 50.1% | 44.7% | 44.0% | 39.7% | 1.008 | 46.3% |
| Random Forest | Entrenamiento | 90.2% | 92.8% | 88.5% | 90.1% | 0.606 | 90.5% |
| Random Forest | Validación | 52.9% | 45.1% | 45.8% | 40.2% | 0.989 | 47.9% |
| Random Forest | Prueba | 50.1% | 39.3% | 42.7% | 37.1% | 1.004 | 44.0% |
| Red Neuronal (MLP) | Entrenamiento | 55.1% | 53.3% | 48.3% | 45.4% | 0.953 | 52.2% |
| Red Neuronal (MLP) | Validación | 51.8% | 45.6% | 45.1% | 41.5% | 0.996 | 47.5% |
| Red Neuronal (MLP) | Prueba | 49.6% | 38.6% | 41.9% | 37.8% | 1.017 | 43.4% |
| Gradient Boosting | Entrenamiento | 60.3% | 67.4% | 53.6% | 51.5% | 0.865 | 60.4% |
| Gradient Boosting | Validación | 51.5% | 41.7% | 45.0% | 39.6% | 1.005 | 46.0% |
| Gradient Boosting | Prueba | 49.4% | 39.8% | 42.8% | 37.7% | 1.017 | 44.0% |
| Extra Trees | Entrenamiento | 84.5% | 88.8% | 81.4% | 83.6% | 0.667 | 84.9% |
| Extra Trees | Validación | 53.6% | 38.4% | 45.6% | 39.7% | 0.983 | 45.9% |
| Extra Trees | Prueba | 50.9% | 66.4% | 42.9% | 37.3% | 0.998 | 53.4% |
| Combinación top 2 | Entrenamiento | 70.6% | 78.7% | 65.1% | 65.9% | 0.765 | 71.5% |
| Combinación top 2 | Validación | 52.4% | 45.8% | 45.5% | 40.3% | 0.989 | 47.9% |
| Combinación top 2 | Prueba | 51.1% | 51.8% | 44.2% | 38.7% | 1.000 | 49.0% |

## Goles

Mejores dos en validación: **Extra Trees** y **Ridge (lineal)**. MAE y RMSE miden el error en goles por partido. Accuracy, precision, recall y F1 son del over/under, promediados en las líneas 1.5 a 3.5.

| Modelo | Conjunto | MAE | RMSE | Accuracy | Precision | Recall | F1 | Score |
|---|---|---|---|---|---|---|---|---|
| Extra Trees | Entrenamiento | 1.020 | 1.286 | 78.0% | 85.6% | 67.7% | 66.6% | 77.1% |
| Extra Trees | Validación | 1.296 | 1.626 | 67.5% | 69.3% | 56.2% | 50.9% | 64.3% |
| Extra Trees | Prueba | 1.342 | 1.688 | 68.1% | 68.3% | 57.0% | 51.8% | 64.5% |
| Ridge (lineal) | Entrenamiento | 1.300 | 1.630 | 67.9% | 63.3% | 55.5% | 50.8% | 62.2% |
| Ridge (lineal) | Validación | 1.294 | 1.629 | 67.9% | 67.7% | 55.8% | 51.6% | 63.8% |
| Ridge (lineal) | Prueba | 1.339 | 1.688 | 67.7% | 66.6% | 54.7% | 51.9% | 63.0% |
| Random Forest | Entrenamiento | 1.088 | 1.377 | 76.6% | 83.2% | 64.7% | 61.7% | 74.8% |
| Random Forest | Validación | 1.300 | 1.632 | 67.4% | 63.2% | 57.8% | 51.8% | 62.8% |
| Random Forest | Prueba | 1.348 | 1.687 | 68.0% | 65.5% | 60.7% | 53.7% | 64.7% |
| Gradient Boosting (Poisson) | Entrenamiento | 1.207 | 1.512 | 72.1% | 77.1% | 60.0% | 57.6% | 69.7% |
| Gradient Boosting (Poisson) | Validación | 1.302 | 1.634 | 67.6% | 61.5% | 55.8% | 51.7% | 61.7% |
| Gradient Boosting (Poisson) | Prueba | 1.351 | 1.690 | 68.1% | 61.9% | 58.3% | 53.4% | 62.7% |
| Red Neuronal (MLP) | Entrenamiento | 1.273 | 1.601 | 68.8% | 63.5% | 56.3% | 54.7% | 62.9% |
| Red Neuronal (MLP) | Validación | 1.317 | 1.662 | 67.1% | 60.5% | 57.0% | 54.0% | 61.5% |
| Red Neuronal (MLP) | Prueba | 1.376 | 1.712 | 67.6% | 61.7% | 64.7% | 60.2% | 64.7% |
| Combinación top 2 | Entrenamiento | 1.157 | 1.454 | 73.4% | 79.4% | 60.9% | 57.1% | 71.2% |
| Combinación top 2 | Validación | 1.291 | 1.623 | 68.0% | 72.7% | 56.3% | 51.3% | 65.6% |
| Combinación top 2 | Prueba | 1.337 | 1.683 | 67.7% | 73.2% | 55.5% | 51.2% | 65.5% |

Over/under por línea de «Combinación top 2» en el bloque de prueba (1,449 partidos, promedio real 2.91 goles). «Señales» son los partidos con P(over) de al menos 60%.

| Línea | % over real | Accuracy | Prec. over | Recall over | F1 over | Prec. under | Recall under | F1 under | Señales | Prec. señal |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 94.3% | 94.3% | 94.3% | 100.0% | 97.1% | 0.0% | 0.0% | 0.0% | 1449 | 94.3% |
| 1.5 | 78.7% | 78.7% | 78.7% | 100.0% | 88.1% | 0.0% | 0.0% | 0.0% | 1449 | 78.7% |
| 2.5 | 55.4% | 55.6% | 59.1% | 64.4% | 61.6% | 50.2% | 44.6% | 47.2% | 287 | 64.8% |
| 3.5 | 31.5% | 68.9% | 81.8% | 2.0% | 3.8% | 68.8% | 99.8% | 81.5% | 0 | n/a |
| 4.5 | 18.4% | 81.6% | 0.0% | 0.0% | 0.0% | 81.6% | 100.0% | 89.9% | 0 | n/a |

## Tiros al arco

Mejores dos en validación: **Extra Trees** y **Gradient Boosting (Poisson)**. MAE y RMSE miden el error en tiros al arco por partido. Accuracy, precision, recall y F1 son del over/under, promediados en las líneas 6.5 a 10.5.

| Modelo | Conjunto | MAE | RMSE | Accuracy | Precision | Recall | F1 | Score |
|---|---|---|---|---|---|---|---|---|
| Extra Trees | Entrenamiento | 2.052 | 2.671 | 77.9% | 85.4% | 63.8% | 64.4% | 75.7% |
| Extra Trees | Validación | 2.409 | 3.057 | 67.1% | 63.7% | 52.8% | 48.6% | 61.2% |
| Extra Trees | Prueba | 2.616 | 3.268 | 65.6% | 72.9% | 56.0% | 52.0% | 64.8% |
| Gradient Boosting (Poisson) | Entrenamiento | 2.509 | 3.240 | 68.9% | 71.3% | 53.5% | 50.2% | 64.6% |
| Gradient Boosting (Poisson) | Validación | 2.412 | 3.062 | 66.8% | 63.2% | 53.0% | 48.4% | 61.0% |
| Gradient Boosting (Poisson) | Prueba | 2.628 | 3.272 | 65.8% | 69.1% | 58.3% | 53.8% | 64.4% |
| Ridge (lineal) | Entrenamiento | 2.662 | 3.411 | 66.8% | 61.9% | 51.5% | 50.6% | 60.0% |
| Ridge (lineal) | Validación | 2.427 | 3.064 | 66.5% | 63.4% | 52.2% | 51.2% | 60.7% |
| Ridge (lineal) | Prueba | 2.606 | 3.235 | 66.2% | 68.0% | 59.0% | 57.3% | 64.4% |
| Random Forest | Entrenamiento | 2.195 | 2.871 | 75.2% | 81.8% | 60.5% | 59.9% | 72.5% |
| Random Forest | Validación | 2.417 | 3.066 | 67.0% | 62.0% | 52.8% | 48.7% | 60.6% |
| Random Forest | Prueba | 2.604 | 3.246 | 66.1% | 68.3% | 58.6% | 54.4% | 64.3% |
| Red Neuronal (MLP) | Entrenamiento | 2.639 | 3.371 | 66.8% | 61.2% | 52.8% | 53.5% | 60.3% |
| Red Neuronal (MLP) | Validación | 2.559 | 3.214 | 64.7% | 57.5% | 52.8% | 52.8% | 58.3% |
| Red Neuronal (MLP) | Prueba | 2.746 | 3.407 | 63.9% | 61.2% | 63.4% | 60.8% | 62.8% |
| Combinación top 2 | Entrenamiento | 2.277 | 2.949 | 73.6% | 79.9% | 58.4% | 56.9% | 70.6% |
| Combinación top 2 | Validación | 2.407 | 3.054 | 67.2% | 63.3% | 53.1% | 48.7% | 61.2% |
| Combinación top 2 | Prueba | 2.619 | 3.265 | 65.6% | 67.9% | 56.8% | 52.4% | 63.4% |

Over/under por línea de «Combinación top 2» en el bloque de prueba (1,449 partidos, promedio real 9.37 tiros al arco). «Señales» son los partidos con P(over) de al menos 60%.

| Línea | % over real | Accuracy | Prec. over | Recall over | F1 over | Prec. under | Recall under | F1 under | Señales | Prec. señal |
|---|---|---|---|---|---|---|---|---|---|---|
| 5.5 | 88.5% | 88.5% | 88.5% | 100.0% | 93.9% | 0.0% | 0.0% | 0.0% | 1449 | 88.5% |
| 6.5 | 80.0% | 80.0% | 80.0% | 100.0% | 88.9% | 0.0% | 0.0% | 0.0% | 1446 | 80.0% |
| 7.5 | 68.8% | 68.7% | 69.0% | 99.1% | 81.4% | 47.1% | 1.8% | 3.4% | 1152 | 72.0% |
| 8.5 | 57.8% | 58.2% | 62.6% | 68.7% | 65.5% | 50.5% | 43.7% | 46.8% | 279 | 70.6% |
| 9.5 | 46.1% | 56.4% | 61.1% | 14.8% | 23.9% | 55.8% | 91.9% | 69.4% | 12 | 83.3% |
| 10.5 | 35.3% | 64.9% | 66.7% | 1.2% | 2.3% | 64.9% | 99.7% | 78.6% | 0 | n/a |
| 11.5 | 26.2% | 73.8% | 0.0% | 0.0% | 0.0% | 73.8% | 100.0% | 84.9% | 0 | n/a |

## Tiros de esquina

Mejores dos en validación: **Gradient Boosting (Poisson)** y **Red Neuronal (MLP)**. MAE y RMSE miden el error en córners por partido. Accuracy, precision, recall y F1 son del over/under, promediados en las líneas 7.5 a 11.5.

| Modelo | Conjunto | MAE | RMSE | Accuracy | Precision | Recall | F1 | Score |
|---|---|---|---|---|---|---|---|---|
| Gradient Boosting (Poisson) | Entrenamiento | 2.780 | 3.567 | 66.9% | 70.1% | 51.0% | 46.3% | 62.7% |
| Gradient Boosting (Poisson) | Validación | 2.662 | 3.371 | 65.2% | 50.3% | 48.6% | 41.6% | 54.7% |
| Gradient Boosting (Poisson) | Prueba | 2.854 | 3.563 | 64.0% | 50.1% | 52.5% | 46.1% | 55.5% |
| Red Neuronal (MLP) | Entrenamiento | 2.804 | 3.607 | 66.4% | 63.6% | 48.7% | 49.7% | 59.6% |
| Red Neuronal (MLP) | Validación | 2.786 | 3.549 | 63.0% | 53.1% | 46.8% | 45.4% | 54.3% |
| Red Neuronal (MLP) | Prueba | 2.986 | 3.718 | 61.3% | 56.1% | 55.6% | 52.9% | 57.7% |
| Random Forest | Entrenamiento | 2.430 | 3.153 | 74.1% | 84.9% | 58.6% | 57.2% | 72.5% |
| Random Forest | Validación | 2.665 | 3.375 | 64.9% | 48.8% | 48.8% | 41.7% | 54.2% |
| Random Forest | Prueba | 2.860 | 3.560 | 63.8% | 49.5% | 53.5% | 46.9% | 55.6% |
| Extra Trees | Entrenamiento | 2.266 | 2.933 | 77.1% | 87.3% | 62.6% | 62.7% | 75.7% |
| Extra Trees | Validación | 2.662 | 3.368 | 65.1% | 48.2% | 49.0% | 41.7% | 54.1% |
| Extra Trees | Prueba | 2.836 | 3.551 | 64.1% | 49.7% | 52.6% | 45.6% | 55.5% |
| Ridge (lineal) | Entrenamiento | 2.968 | 3.785 | 64.5% | 59.4% | 48.0% | 46.0% | 57.3% |
| Ridge (lineal) | Validación | 2.702 | 3.401 | 64.8% | 46.4% | 47.7% | 42.9% | 53.0% |
| Ridge (lineal) | Prueba | 2.864 | 3.542 | 63.8% | 56.3% | 56.6% | 51.1% | 58.9% |
| Combinación top 2 | Entrenamiento | 2.756 | 3.540 | 67.2% | 67.0% | 50.1% | 47.7% | 61.4% |
| Combinación top 2 | Validación | 2.680 | 3.409 | 64.7% | 57.4% | 47.9% | 43.0% | 56.7% |
| Combinación top 2 | Prueba | 2.880 | 3.589 | 63.2% | 60.2% | 54.5% | 49.4% | 59.3% |

Over/under por línea de «Combinación top 2» en el bloque de prueba (1,449 partidos, promedio real 10.07 córners). «Señales» son los partidos con P(over) de al menos 60%.

| Línea | % over real | Accuracy | Prec. over | Recall over | F1 over | Prec. under | Recall under | F1 under | Señales | Prec. señal |
|---|---|---|---|---|---|---|---|---|---|---|
| 6.5 | 85.0% | 85.0% | 85.0% | 100.0% | 91.9% | 0.0% | 0.0% | 0.0% | 1449 | 85.0% |
| 7.5 | 74.9% | 74.7% | 74.9% | 99.6% | 85.5% | 0.0% | 0.0% | 0.0% | 1408 | 74.9% |
| 8.5 | 63.7% | 61.8% | 63.5% | 94.4% | 75.9% | 32.5% | 4.8% | 8.3% | 969 | 65.9% |
| 9.5 | 53.1% | 53.3% | 55.5% | 61.4% | 58.3% | 50.3% | 44.3% | 47.1% | 207 | 58.5% |
| 10.5 | 41.8% | 58.7% | 51.7% | 14.9% | 23.1% | 59.6% | 90.0% | 71.7% | 20 | 65.0% |
| 11.5 | 32.8% | 67.4% | 55.6% | 2.1% | 4.1% | 67.5% | 99.2% | 80.3% | 2 | 0.0% |
| 12.5 | 25.3% | 74.5% | 0.0% | 0.0% | 0.0% | 74.6% | 99.8% | 85.4% | 0 | n/a |
| 13.5 | 17.9% | 82.1% | 0.0% | 0.0% | 0.0% | 82.1% | 100.0% | 90.1% | 0 | n/a |
