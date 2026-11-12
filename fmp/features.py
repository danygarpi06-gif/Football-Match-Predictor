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

from .players import BENCH, STARTER, pos_group

STAT_KEYS = ("pts", "gf", "ga", "sf", "sa", "stf", "sta", "cf", "ca", "poss")
META_COLS = ("league", "date", "home", "away", "match_id", "y_outcome", "y_corners")
XI_RATES = ("g", "a", "sh", "sot")                     # per-player goals, assists, shots, shots on target
XI_KEYS = XI_RATES + ("rating", "exp", "cont", "starts", "g_gap", "sot_gap", "rating_gap")
XI_DIFFS = ("g", "sot", "rating", "cont", "g_gap", "rating_gap")


def _nanmean(values) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else np.nan


class PlayerBook:
    """Leak-free player state shared by every league (ESPN player and team ids are global).

    Each player keeps recency-weighted goals / assists / shots / shots on target per start and a rating
    that rises when the team beats its Elo expectation with him on the pitch. A starting XI is then
    described by the sum of its players, and compared with the team's usual XI.
    """
    DECAY, SHRINK, N_RECENT, SUB_W = 0.97, 5.0, 8, 0.3

    def __init__(self):
        self.p: dict[int, dict] = {}
        self.prior: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(5))  # position group -> [starts, g, a, sh, sot]
        self.team: dict[str, list[dict]] = defaultdict(list)                  # team id -> matches as {player id: role}
        self.team_id: dict[tuple[str, str], str] = {}                         # (league, team name) -> team id

    def _rates(self, pid: int, pos: str = "") -> np.ndarray:
        st = self.p.get(pid)
        grp = st["grp"] if st and st["grp"] != "U" else pos_group(pos)  # today's position only for unknown players
        pr = self.prior[grp]
        prior = pr[1:] / pr[0] if pr[0] > 0 else np.zeros(4)
        if st is None:
            return prior
        return (st["x"] + self.SHRINK * prior) / (st["w"] + self.SHRINK)

    def start_scores(self, tid: str) -> tuple[dict[int, float], dict[int, float]]:
        """(recency-weighted start score, share of the last matches started) per player."""
        hist = self.team[tid][-self.N_RECENT:]
        score, share = defaultdict(float), defaultdict(float)
        for age, match in enumerate(reversed(hist)):
            for pid, role in match.items():
                if role == STARTER:
                    score[pid] += 0.85 ** age
                    share[pid] += 1.0 / len(hist)
        return score, share

    def usual_xi(self, tid: str) -> list[int]:
        """The 11 players who started most often in the team's last matches (one goalkeeper)."""
        score, _ = self.start_scores(tid)
        xi, keeper = [], False
        for pid in sorted(score, key=score.get, reverse=True):
            is_gk = self.p.get(pid, {}).get("grp") == "G"
            if is_gk and keeper:
                continue
            keeper = keeper or is_gk
            xi.append(pid)
            if len(xi) == 11:
                break
        return xi

    def squad(self, tid: str, n: int = 12) -> list[int]:
        """Players in the last match-day squads, most used first."""
        score, _ = self.start_scores(tid)
        seen: dict[int, float] = {}
        for age, match in enumerate(reversed(self.team[tid][-n:])):
            for pid, role in match.items():
                seen[pid] = seen.get(pid, 0.0) + (0.0 if role == BENCH else 0.01) + 0.001 * 0.85 ** age
        return sorted(seen, key=lambda pid: score.get(pid, 0.0) + seen[pid], reverse=True)

    def xi_block(self, tid: str | None, xi: list[tuple[int, str]] | None, p: str) -> dict:
        out = {f"{p}_xi_{k}": np.nan for k in XI_KEYS}
        if not tid or not xi or len(xi) != 11:
            return out
        rates = np.sum([self._rates(pid, pos) for pid, pos in xi], axis=0)
        rating = float(np.mean([self.p[pid]["r"] if pid in self.p else 0.0 for pid, _ in xi]))
        out.update({f"{p}_xi_{k}": float(v) for k, v in zip(XI_RATES, rates)})
        out[f"{p}_xi_rating"] = rating
        out[f"{p}_xi_exp"] = float(np.mean([np.log1p(self.p[pid]["n"]) if pid in self.p else 0.0 for pid, _ in xi]))
        if len(self.team[tid]) >= 3:
            usual, (_, share) = self.usual_xi(tid), self.start_scores(tid)
            ids = [pid for pid, _ in xi]
            out[f"{p}_xi_cont"] = len(set(ids) & set(usual)) / 11.0
            out[f"{p}_xi_starts"] = float(np.mean([share.get(pid, 0.0) for pid in ids]))
            if len(usual) == 11:
                u_rates = np.sum([self._rates(pid) for pid in usual], axis=0)
                out[f"{p}_xi_g_gap"] = float(rates[0] - u_rates[0])
                out[f"{p}_xi_sot_gap"] = float(rates[3] - u_rates[3])
                out[f"{p}_xi_rating_gap"] = rating - float(np.mean([self.p[pid]["r"] for pid in usual]))
        return out

    def update(self, league: str, home: str, away: str, rec: dict, surprise: float) -> None:
        """surprise = (home result - Elo expectation) * goal-margin multiplier."""
        for key, name, sign in (("h", home, 1.0), ("a", away, -1.0)):
            side = rec[key]
            tid = side["t"]
            self.team_id[(league, name)] = tid
            self.team[tid].append({pl[0]: pl[1] for pl in side["p"]})
            del self.team[tid][:-20]
            for pid, role, pos, *stats in side["p"]:
                st = self.p.setdefault(pid, {"w": 0.0, "x": np.zeros(4), "r": 0.0, "n": 0, "grp": "U"})
                if role == BENCH:
                    continue
                w = 1.0 if role == STARTER else self.SUB_W
                if role == STARTER:
                    st["grp"] = pos_group(pos)
                x = np.asarray(stats, dtype=float)
                st["w"] = st["w"] * self.DECAY + w
                st["x"] = st["x"] * self.DECAY + x
                st["r"] = st["r"] * self.DECAY + w * sign * surprise
                st["n"] += 1
                self.prior[st["grp"]] += np.concatenate(([w], x))


def starters(side: dict) -> list[tuple[int, str]]:
    return [(pl[0], pl[2]) for pl in side["p"] if pl[1] == STARTER]


class FeatureBuilder:
    def __init__(self, n_short: int = 5, n_long: int = 10, k: float = 20.0, home_adv: float = 60.0,
                 book: PlayerBook | None = None, league: str = ""):
        self.n_short, self.n_long, self.k, self.home_adv = n_short, n_long, k, home_adv
        self.book, self.league = book, league
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
    def features(self, home: str, away: str, date: pd.Timestamp, neutral: bool = False,
                 xi: dict[str, list[tuple[int, str]]] | None = None) -> dict:
        """xi = {"h": [(player id, position), ...], "a": [...]}: the two starting line-ups, when known."""
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
        if self.book is not None:
            for side, team in (("h", home), ("a", away)):
                f.update(self.book.xi_block(self.book.team_id.get((self.league, team)), (xi or {}).get(side), side))
            for key in XI_DIFFS:
                f[f"d_xi_{key}"] = f[f"h_xi_{key}"] - f[f"a_xi_{key}"]
        return f

    # ---- update state with a finished match ------------------------------- #
    def update(self, m, rec: dict | None = None) -> None:
        hg, ag = float(m.hg), float(m.ag)
        ha = 0.0 if m.neutral else self.home_adv
        exp_h = 1.0 / (1.0 + 10 ** (-(self.elo[m.home] - self.elo[m.away] + ha) / 400.0))
        res = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        gd = abs(hg - ag)
        mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11.0 + gd) / 8.0
        delta = self.k * mult * (res - exp_h)
        self.elo[m.home] += delta
        self.elo[m.away] -= delta
        if self.book is not None and rec:
            self.book.update(self.league, m.home, m.away, rec, mult * (res - exp_h))
        self.hist[m.home].append(dict(date=m.date, venue="H", opp=m.away, pts=3.0 if hg > ag else 1.0 if hg == ag else 0.0,
                                      gf=hg, ga=ag, sf=m.hs, sa=m.as_, stf=m.hst, sta=m.ast, cf=m.hc, ca=m.ac, poss=m.hposs))
        self.hist[m.away].append(dict(date=m.date, venue="A", opp=m.home, pts=3.0 if ag > hg else 1.0 if hg == ag else 0.0,
                                      gf=ag, ga=hg, sf=m.as_, sa=m.hs, stf=m.ast, sta=m.hst, cf=m.ac, ca=m.hc, poss=m.aposs))


def build_training_frame(matches: pd.DataFrame, lineups: dict | None = None,
                         min_played: int = 3) -> tuple[pd.DataFrame, dict[str, FeatureBuilder], PlayerBook | None]:
    """Replay every finished match in chronological order. Returns (frame, builders per league, player book).

    Team form and Elo are kept per league; the player book is shared, so a player's league matches
    also inform his Champions League line-ups.
    """
    lineups = lineups or {}
    book = PlayerBook() if lineups else None
    builders = {lg: FeatureBuilder(book=book, league=lg) for lg in matches["league"].unique()}
    rows: list[dict] = []
    finished = matches[matches["status"] == "FT"].sort_values("date", kind="stable")
    for m in finished.itertuples(index=False):
        fb, rec = builders[m.league], lineups.get(str(m.match_id))
        f = fb.features(m.home, m.away, m.date, bool(m.neutral),
                        {"h": starters(rec["h"]), "a": starters(rec["a"])} if rec else None)
        f.update(league=m.league, date=m.date, home=m.home, away=m.away,
                 y_outcome=0 if m.hg > m.ag else 1 if m.hg == m.ag else 2,
                 y_corners=m.hc + m.ac, y_hc=m.hc, y_ac=m.ac,          # NaN propagates when a stat is missing
                 y_goals=m.hg + m.ag, y_hg=m.hg, y_ag=m.ag,
                 y_sot=m.hst + m.ast, y_hst=m.hst, y_ast=m.ast)
        rows.append(f)
        fb.update(m, rec)
    frame = pd.DataFrame(rows)
    frame = frame[(frame["h_played"] >= min_played) & (frame["a_played"] >= min_played)]
    return frame.sort_values("date", kind="stable").reset_index(drop=True), builders, book


def feature_matrix(frame: pd.DataFrame, leagues: list[str]) -> pd.DataFrame:
    """Numeric design matrix (+ league one-hot when several leagues are pooled)."""
    X = frame.drop(columns=[c for c in frame.columns if c in META_COLS or c.startswith("y_")]).astype(float)
    if len(X) > 1:  # training matrix: drop variables the data source does not provide (e.g. possession)
        X = X.loc[:, X.notna().any()]
    if len(leagues) > 1:
        for lg in leagues:
            X[f"lg_{lg}"] = (frame["league"] == lg).astype(float).values
    return X
