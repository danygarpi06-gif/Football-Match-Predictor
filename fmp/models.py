"""Model zoo, evaluation and selection (top-2 by accuracy / precision / recall)."""
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
from sklearn.model_selection import KFold, cross_val_predict
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


def time_split(n: int, test_frac: float = 0.2, min_test: int = 150) -> int:
    """Index where the chronological hold-out starts."""
    return max(int(n * (1 - test_frac)), 0) if n - int(n * (1 - test_frac)) >= min_test else max(n - min_test, n // 2)


# --------------------------------------------------------------------------- #
# 1X2 outcome
# --------------------------------------------------------------------------- #
def outcome_metrics_table(preds: dict[str, np.ndarray], yte: np.ndarray, mask: np.ndarray | None = None) -> pd.DataFrame:
    """Accuracy / macro precision / macro recall per model on the hold-out (optionally a subset)."""
    rows = []
    for name, proba in preds.items():
        if mask is not None:
            proba, y = proba[mask], yte[mask]
        else:
            y = yte
        pred = proba.argmax(axis=1)
        acc = accuracy_score(y, pred)
        prec = precision_score(y, pred, average="macro", zero_division=0)
        rec = recall_score(y, pred, average="macro", zero_division=0)
        rows.append({"modelo": name, "accuracy": acc, "precision": prec, "recall": rec,
                     "f1": f1_score(y, pred, average="macro", zero_division=0),
                     "log_loss": log_loss(y, proba, labels=[0, 1, 2]), "score": (acc + prec + rec) / 3})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def evaluate_outcome_models(X: pd.DataFrame, y: pd.Series, split: int, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Train the 5 classifiers on the oldest `split` rows, evaluate on the most recent ones."""
    Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:].to_numpy()
    preds = {}
    for name, model in outcome_models(seed).items():
        model.fit(Xtr, ytr)
        preds[name] = model.predict_proba(Xte)
    return outcome_metrics_table(preds, yte), preds


def fit_outcome_ensemble(names: list[str], X: pd.DataFrame, y: pd.Series, seed: int = 42) -> list:
    zoo = outcome_models(seed)
    models = [zoo[n] for n in names]
    for m in models:
        m.fit(X, y)
    return models


def predict_outcome(models: list, x: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict_proba(x)[0] for m in models], axis=0)


# --------------------------------------------------------------------------- #
# corners
# --------------------------------------------------------------------------- #
def _dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """Negative-binomial dispersion alpha (var = mu + alpha*mu^2) by method of moments."""
    resid_var = float(np.mean((y - mu) ** 2))
    alpha = (resid_var - float(np.mean(mu))) / float(np.mean(mu ** 2))
    return float(max(alpha, 0.0))


def prob_over(mu: np.ndarray, alpha: float, line: float) -> np.ndarray:
    """P(total corners > line) under NegBin(mu, alpha) (Poisson when alpha == 0)."""
    mu = np.clip(np.asarray(mu, dtype=float), 0.5, 30.0)
    k = int(np.floor(line))
    if alpha <= 1e-6:
        return 1.0 - stats.poisson.cdf(k, mu)
    n = 1.0 / alpha
    p = n / (n + mu)
    return 1.0 - stats.nbinom.cdf(k, n, p)


def line_metrics(y: np.ndarray, mu: np.ndarray, alpha: float, lines=CORNER_LINES, conf: float = 0.5) -> pd.DataFrame:
    rows = []
    for L in lines:
        p = prob_over(mu, alpha, L)
        actual = y > L
        pred = p >= 0.5
        sig = p >= conf  # confident "over" calls
        rows.append({"linea": L, "pct_over_real": actual.mean(), "accuracy": accuracy_score(actual, pred),
                     "precision_over": precision_score(actual, pred, zero_division=0),
                     "recall_over": recall_score(actual, pred, zero_division=0),
                     "precision_under": precision_score(~actual, ~pred, zero_division=0),
                     "recall_under": recall_score(~actual, ~pred, zero_division=0),
                     "senal_n": int(sig.sum()),
                     "senal_precision": float(actual[sig].mean()) if sig.any() else np.nan})
    return pd.DataFrame(rows)


def evaluate_corner_models(X: pd.DataFrame, y: pd.Series, split: int, seed: int = 42,
                           lines=CORNER_LINES, rank_lines=RANK_LINES) -> tuple[pd.DataFrame, dict]:
    """Works for any count target (corners, goals, shots on target)."""
    Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split].to_numpy(), y.iloc[split:].to_numpy()
    rows, info = [], {}
    kf = KFold(n_splits=3, shuffle=False)
    for name, model in corner_models(seed).items():
        oof = cross_val_predict(clone(model), Xtr, ytr, cv=kf)          # out-of-fold -> honest dispersion
        alpha = _dispersion(ytr, oof)
        model.fit(Xtr, ytr)
        mu = model.predict(Xte)
        lm = line_metrics(yte, mu, alpha, lines=lines)
        central = lm[lm["linea"].isin(rank_lines)]
        score = float(((central["accuracy"] + central["precision_over"] + central["recall_over"]) / 3).mean())
        rows.append({"modelo": name, "MAE": mean_absolute_error(yte, mu), "accuracy": central["accuracy"].mean(),
                     "precision": central["precision_over"].mean(), "recall": central["recall_over"].mean(),
                     "score": score})
        info[name] = {"model": model, "alpha": alpha, "mu_test": mu, "oof": oof}
    table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return table, info


def corner_ensemble_backtest(names: list[str], info: dict, ytr: np.ndarray, yte: np.ndarray, conf: float,
                             mask: np.ndarray | None = None, lines=CORNER_LINES) -> tuple[pd.DataFrame, float]:
    mu = np.mean([info[n]["mu_test"] for n in names], axis=0)
    oof = np.mean([info[n]["oof"] for n in names], axis=0)
    alpha = _dispersion(ytr, oof)
    if mask is not None:
        mu, yte = mu[mask], yte[mask]
    return line_metrics(yte, mu, alpha, lines=lines, conf=conf), alpha


def fit_corner_ensemble(names: list[str], X: pd.DataFrame, y: pd.Series, seed: int = 42) -> list:
    zoo = corner_models(seed)
    models = [zoo[n] for n in names]
    for m in models:
        m.fit(X, y)
    return models


def predict_corners(models: list, x: pd.DataFrame) -> float:
    return float(np.mean([m.predict(x)[0] for m in models]))
