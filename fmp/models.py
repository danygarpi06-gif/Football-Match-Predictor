"""Model zoo, evaluation and selection.

Chronological 3-way split, so that reported numbers are honest:
  * train (oldest 70%)      -> models are fitted here
  * validation (next 15%)   -> used to pick the top-2 models and calibrate over/under probabilities
  * test (most recent 15%)  -> never used for any decision; only for the final report
The models that predict future matches are then refitted on all the data.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error, precision_score,
                             recall_score)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

OUTCOME_LABELS = {0: "Gana local", 1: "Empate", 2: "Gana visitante"}
CORNER_LINES = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
RANK_LINES = [7.5, 8.5, 9.5, 10.5, 11.5]  # central lines used to rank corner models


def outcome_models(seed: int = 42) -> dict:
    imp = SimpleImputer(strategy="median")
    return {
        "Regresión Logística": make_pipeline(clone(imp), StandardScaler(),
                                             LogisticRegression(C=0.2, max_iter=3000)),
        "Random Forest": make_pipeline(clone(imp), RandomForestClassifier(
            n_estimators=600, min_samples_leaf=6, max_features="sqrt", n_jobs=-1, random_state=seed)),
        "Extra Trees": make_pipeline(clone(imp), ExtraTreesClassifier(
            n_estimators=600, min_samples_leaf=6, max_features="sqrt", n_jobs=-1, random_state=seed)),
        "Gradient Boosting": make_pipeline(HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=500, max_leaf_nodes=12, min_samples_leaf=30, l2_regularization=2.0,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=seed)),
        "Red Neuronal (MLP)": make_pipeline(clone(imp), StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True, max_iter=600, random_state=seed)),
    }


def corner_models(seed: int = 42) -> dict:
    imp = SimpleImputer(strategy="median")
    return {
        "Ridge (lineal)": make_pipeline(clone(imp), StandardScaler(), Ridge(alpha=5.0)),
        "Random Forest": make_pipeline(clone(imp), RandomForestRegressor(
            n_estimators=200, min_samples_leaf=10, max_features=0.33, max_samples=0.5, n_jobs=-1, random_state=seed)),
        "Extra Trees": make_pipeline(clone(imp), ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=10, max_features=0.33, n_jobs=-1, random_state=seed)),
        "Gradient Boosting (Poisson)": make_pipeline(HistGradientBoostingRegressor(
            loss="poisson", learning_rate=0.03, max_iter=500, max_leaf_nodes=12, min_samples_leaf=30,
            l2_regularization=2.0, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=seed)),
        "Red Neuronal (MLP)": make_pipeline(clone(imp), StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True, max_iter=800, random_state=seed)),
    }


SPLITS = {"train": "Entrenamiento", "val": "Validación", "test": "Prueba"}
ENSEMBLE = "Combinación top 2"


def three_way_split(n: int, frac: float = 0.15, min_part: int = 120) -> tuple[int, int]:
    """Indices where validation and test start (rows must be sorted by date)."""
    part = max(int(n * frac), min(min_part, n // 5))
    return n - 2 * part, n - part


def split_masks(n: int, i_val: int, i_test: int) -> dict[str, np.ndarray]:
    idx = np.arange(n)
    return {"train": idx < i_val, "val": (idx >= i_val) & (idx < i_test), "test": idx >= i_test}


# --------------------------------------------------------------------------- #
# 1X2 outcome
# --------------------------------------------------------------------------- #
def class_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, average="macro", zero_division=0)
    rec = recall_score(y, pred, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1_score(y, pred, average="macro", zero_division=0),
            "log_loss": log_loss(y, proba, labels=[0, 1, 2]), "score": (acc + prec + rec) / 3}


def outcome_table(preds: dict[str, np.ndarray], y: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """One row per model with the metrics on the rows selected by `mask`."""
    rows = [{"modelo": name, **class_metrics(y[mask], proba[mask])} for name, proba in preds.items()]
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def _long_table(per_model: dict[str, dict[str, dict]], ranking: list[str]) -> pd.DataFrame:
    rows = [{"modelo": name, "conjunto": SPLITS[sp], **per_model[name][sp]} for name in ranking for sp in SPLITS]
    return pd.DataFrame(rows)


def evaluate_outcome_models(X: pd.DataFrame, y: pd.Series, i_val: int, i_test: int, seed: int = 42) -> dict:
    """Fit the 5 classifiers on train, rank them on validation, report train / validation / test."""
    yv, masks = y.to_numpy(), split_masks(len(y), i_val, i_test)
    preds, per_model = {}, {}
    for name, model in outcome_models(seed).items():
        model.fit(X.iloc[:i_val], y.iloc[:i_val])
        preds[name] = model.predict_proba(X)
        per_model[name] = {sp: class_metrics(yv[m], preds[name][m]) for sp, m in masks.items()}
    ranking = sorted(per_model, key=lambda n: per_model[n]["val"]["score"], reverse=True)
    top2 = ranking[:2]
    preds[ENSEMBLE] = np.mean([preds[n] for n in top2], axis=0)
    per_model[ENSEMBLE] = {sp: class_metrics(yv[m], preds[ENSEMBLE][m]) for sp, m in masks.items()}
    return {"metrics": _long_table(per_model, ranking + [ENSEMBLE]), "preds": preds, "top2": top2}


def fit_outcome_ensemble(names: list[str], X: pd.DataFrame, y: pd.Series, seed: int = 42) -> list:
    zoo = outcome_models(seed)
    models = [zoo[n] for n in names]
    for m in models:
        m.fit(X, y)
    return models


def predict_outcome(models: list, x: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict_proba(x)[0] for m in models], axis=0)


# --------------------------------------------------------------------------- #
# count targets: corners, goals, shots on target
# --------------------------------------------------------------------------- #
def _dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """Negative-binomial dispersion alpha (var = mu + alpha*mu^2) by method of moments."""
    resid_var = float(np.mean((y - mu) ** 2))
    alpha = (resid_var - float(np.mean(mu))) / float(np.mean(mu ** 2))
    return float(max(alpha, 0.0))


def prob_over(mu: np.ndarray, alpha: float, line: float) -> np.ndarray:
    """P(total > line) under NegBin(mu, alpha) (Poisson when alpha == 0)."""
    mu = np.clip(np.asarray(mu, dtype=float), 0.5, 30.0)
    k = int(np.floor(line))
    if alpha <= 1e-6:
        return 1.0 - stats.poisson.cdf(k, mu)
    n = 1.0 / alpha
    p = n / (n + mu)
    return 1.0 - stats.nbinom.cdf(k, n, p)


def line_metrics(y: np.ndarray, mu: np.ndarray, alpha: float, lines=CORNER_LINES, conf: float = 0.5) -> pd.DataFrame:
    """Over/under classification metrics for every line."""
    rows = []
    for L in lines:
        p = prob_over(mu, alpha, L)
        actual = y > L
        pred = p >= 0.5
        sig = p >= conf  # confident "over" calls
        rows.append({"linea": L, "pct_over_real": actual.mean(), "accuracy": accuracy_score(actual, pred),
                     "precision_over": precision_score(actual, pred, zero_division=0),
                     "recall_over": recall_score(actual, pred, zero_division=0),
                     "f1_over": f1_score(actual, pred, zero_division=0),
                     "precision_under": precision_score(~actual, ~pred, zero_division=0),
                     "recall_under": recall_score(~actual, ~pred, zero_division=0),
                     "f1_under": f1_score(~actual, ~pred, zero_division=0),
                     "senal_n": int(sig.sum()),
                     "senal_precision": float(actual[sig].mean()) if sig.any() else np.nan})
    return pd.DataFrame(rows)


def count_metrics(y: np.ndarray, mu: np.ndarray, alpha: float, lines, rank_lines) -> dict:
    """Regression error (MAE, RMSE) + over/under metrics averaged over the central lines."""
    lm = line_metrics(y, mu, alpha, lines=lines)
    c = lm[lm["linea"].isin(rank_lines)]
    acc, prec, rec = c["accuracy"].mean(), c["precision_over"].mean(), c["recall_over"].mean()
    return {"MAE": mean_absolute_error(y, mu), "RMSE": float(np.sqrt(np.mean((y - mu) ** 2))),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": c["f1_over"].mean(),
            "score": (acc + prec + rec) / 3}


def evaluate_count_models(X: pd.DataFrame, y: pd.Series, i_val: int, i_test: int, seed: int = 42,
                          lines=CORNER_LINES, rank_lines=RANK_LINES) -> dict:
    """Fit the 5 regressors on train, calibrate + rank on validation, report train / validation / test."""
    yv, masks = y.to_numpy(), split_masks(len(y), i_val, i_test)
    mus, alphas, per_model = {}, {}, {}

    def add(name: str, mu: np.ndarray) -> None:
        mus[name] = mu
        alphas[name] = _dispersion(yv[masks["val"]], mu[masks["val"]])   # out-of-sample calibration
        per_model[name] = {sp: count_metrics(yv[m], mu[m], alphas[name], lines, rank_lines) for sp, m in masks.items()}

    for name, model in corner_models(seed).items():
        model.fit(X.iloc[:i_val], yv[:i_val])
        add(name, model.predict(X))
    ranking = sorted(per_model, key=lambda n: per_model[n]["val"]["score"], reverse=True)
    top2 = ranking[:2]
    add(ENSEMBLE, np.mean([mus[n] for n in top2], axis=0))
    return {"metrics": _long_table(per_model, ranking + [ENSEMBLE]), "top2": top2,
            "mu": mus[ENSEMBLE], "alpha": alphas[ENSEMBLE]}


def fit_corner_ensemble(names: list[str], X: pd.DataFrame, y: pd.Series, seed: int = 42) -> list:
    zoo = corner_models(seed)
    models = [zoo[n] for n in names]
    for m in models:
        m.fit(X, y)
    return models


def predict_corners(models: list, x: pd.DataFrame) -> float:
    return float(np.mean([m.predict(x)[0] for m in models]))
