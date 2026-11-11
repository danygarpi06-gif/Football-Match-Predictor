"""Leak-free feature engineering.

Matches are replayed in chronological order. For each match the features use
ONLY information available before kick-off (previous matches of both teams,
Elo ratings, head-to-head). After the features are computed the match result
is used to update the team histories / Elo.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

STAT_KEYS = ("pts", "gf", "ga", "sf", "sa", "stf", "sta", "cf", "ca", "poss")
META_COLS = ("league", "date", "home", "away", "y_outcome", "y_corners")


def _nanmean(values) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else np.nan


class FeatureBuilder:
    def __init__(self, n_short: int = 5, n_long: int = 10, k: float = 20.0, home_adv: float = 60.0):
        self.n_short, self.n_long, self.k, self.home_adv = n_short, n_long, k, home_adv
        self.hist: dict[str, list[dict]] = defaultdict(list)
        self.elo: dict[str, float] = defaultdict(lambda: 1500.0)

    # ---- per-team block --------------------------------------------------- #
    def _team_block(self, team: str, date: pd.Timestamp, venue: str, p: str) -> dict:
        h = self.hist[team]
        out = {f"{p}_played": len(h), f"{p}_elo": self.elo[team]}
        for n, tag in ((self.n_short, "s"), (self.n_long, "l")):
            seg = h[-n:]
            for key in STAT_KEYS:
                out[f"{p}_{key}_{tag}"] = _nanmean(x[key] for x in seg)
        vseg = [x for x in h if x["venue"] == venue][-self.n_short:]
        for key in ("pts", "gf", "ga", "cf", "ca"):
            out[f"{p}_{key}_v"] = _nanmean(x[key] for x in vseg)
        # exponentially weighted recent points (momentum)
        if h:
            w = np.array([0.8 ** i for i in range(len(h[-8:]))])[::-1]
            out[f"{p}_pts_ewm"] = float(np.dot(w, [x["pts"] for x in h[-8:]]) / w.sum())
            out[f"{p}_rest"] = float(min((date - h[-1]["date"]).days, 30))
        else:
            out[f"{p}_pts_ewm"], out[f"{p}_rest"] = np.nan, np.nan
        return out

    # ---- match features --------------------------------------------------- #
    def features(self, home: str, away: str, date: pd.Timestamp, neutral: bool = False) -> dict:
        f = {}
        f.update(self._team_block(home, date, "H", "h"))
        f.update(self._team_block(away, date, "A", "a"))
        ha = 0.0 if neutral else self.home_adv
        f["elo_diff"] = f["h_elo"] - f["a_elo"] + ha
        f["elo_exp_home"] = 1.0 / (1.0 + 10 ** (-f["elo_diff"] / 400.0))
        for key in ("pts_s", "pts_l", "gf_l", "ga_l", "sf_l", "sa_l", "stf_l", "sta_l", "cf_l", "ca_l", "poss_l", "pts_ewm", "rest"):
            f[f"d_{key}"] = f[f"h_{key}"] - f[f"a_{key}"]
        f["attack_h"] = f["h_gf_l"] - f["a_ga_l"]          # home attack vs away defence
        f["attack_a"] = f["a_gf_l"] - f["h_ga_l"]
        f["exp_corners_l"] = _nanmean([f["h_cf_l"], f["a_ca_l"]]) + _nanmean([f["a_cf_l"], f["h_ca_l"]])
        f["exp_corners_s"] = _nanmean([f["h_cf_s"], f["a_ca_s"]]) + _nanmean([f["a_cf_s"], f["h_ca_s"]])
        f["exp_corners_v"] = _nanmean([f["h_cf_v"], f["a_ca_v"]]) + _nanmean([f["a_cf_v"], f["h_ca_v"]])
        prev = [x for x in self.hist[home] if x["opp"] == away][-3:]
        f["h2h_n"] = len(prev)
        f["h2h_gd"] = _nanmean(x["gf"] - x["ga"] for x in prev)
        f["h2h_corners"] = _nanmean(x["cf"] + x["ca"] for x in prev)
        f["neutral"] = int(neutral)
        return f

    # ---- update state with a finished match ------------------------------- #
    def update(self, m) -> None:
        hg, ag = float(m.hg), float(m.ag)
        ha = 0.0 if m.neutral else self.home_adv
        exp_h = 1.0 / (1.0 + 10 ** (-(self.elo[m.home] - self.elo[m.away] + ha) / 400.0))
        res = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        gd = abs(hg - ag)
        mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11.0 + gd) / 8.0
        delta = self.k * mult * (res - exp_h)
        self.elo[m.home] += delta
        self.elo[m.away] -= delta
        self.hist[m.home].append(dict(date=m.date, venue="H", opp=m.away, pts=3.0 if hg > ag else 1.0 if hg == ag else 0.0,
                                      gf=hg, ga=ag, sf=m.hs, sa=m.as_, stf=m.hst, sta=m.ast, cf=m.hc, ca=m.ac, poss=m.hposs))
        self.hist[m.away].append(dict(date=m.date, venue="A", opp=m.home, pts=3.0 if ag > hg else 1.0 if hg == ag else 0.0,
                                      gf=ag, ga=hg, sf=m.as_, sa=m.hs, stf=m.ast, sta=m.hst, cf=m.ac, ca=m.hc, poss=m.aposs))


def build_training_frame(matches: pd.DataFrame, min_played: int = 3) -> tuple[pd.DataFrame, dict[str, FeatureBuilder]]:
    """Replay finished matches league by league. Returns (frame, builders per league)."""
    builders: dict[str, FeatureBuilder] = {}
    rows: list[dict] = []
    finished = matches[matches["status"] == "FT"].sort_values("date")
    for league, df in finished.groupby("league", sort=False):
        fb = FeatureBuilder()
        for m in df.itertuples(index=False):
            f = fb.features(m.home, m.away, m.date, bool(m.neutral))
            f.update(league=league, date=m.date, home=m.home, away=m.away,
                     y_outcome=0 if m.hg > m.ag else 1 if m.hg == m.ag else 2,
                     y_corners=m.hc + m.ac, y_hc=m.hc, y_ac=m.ac,          # NaN propagates when a stat is missing
                     y_goals=m.hg + m.ag, y_hg=m.hg, y_ag=m.ag,
                     y_sot=m.hst + m.ast, y_hst=m.hst, y_ast=m.ast)
            rows.append(f)
            fb.update(m)
        builders[league] = fb
    frame = pd.DataFrame(rows)
    frame = frame[(frame["h_played"] >= min_played) & (frame["a_played"] >= min_played)]
    return frame.sort_values("date").reset_index(drop=True), builders


def feature_matrix(frame: pd.DataFrame, leagues: list[str]) -> pd.DataFrame:
    """Numeric design matrix (+ league one-hot when several leagues are pooled)."""
    X = frame.drop(columns=[c for c in frame.columns if c in META_COLS or c.startswith("y_")]).astype(float)
    if len(X) > 1:  # training matrix: drop variables the data source does not provide (e.g. possession)
        X = X.loc[:, X.notna().any()]
    if len(leagues) > 1:
        for lg in leagues:
            X[f"lg_{lg}"] = (frame["league"] == lg).astype(float).values
    return X
