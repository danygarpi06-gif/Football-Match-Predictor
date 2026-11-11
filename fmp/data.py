"""Data layer.

Primary source : ESPN public scoreboard API (no API key, updated live, includes
                 shots, shots on target, corners and possession for every league).
Fallback source: football-data.co.uk CSV files (results + stats for the big-4
                 European leagues, results only for Liga MX / Brasileirao).

Everything is cached under ``data_cache/`` so that only the current month(s)
are re-downloaded on each run.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data_cache"
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
FDCOUK_SEASON_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
FDCOUK_NEW_URL = "https://www.football-data.co.uk/new/{code}.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (football-match-predictor; educational use)"}

LEAGUES: dict[str, dict] = {
    "premier_league": {"name": "Premier League (Inglaterra)", "espn": "eng.1", "fdcouk": "E0"},
    "la_liga": {"name": "La Liga (España)", "espn": "esp.1", "fdcouk": "SP1"},
    "serie_a": {"name": "Serie A (Italia)", "espn": "ita.1", "fdcouk": "I1"},
    "bundesliga": {"name": "Bundesliga (Alemania)", "espn": "ger.1", "fdcouk": "D1"},
    "liga_mx": {"name": "Liga MX (México)", "espn": "mex.1", "fdcouk_new": "MEX"},
    "brasileirao": {"name": "Brasileirão Série A (Brasil)", "espn": "bra.1", "fdcouk_new": "BRA"},
    "champions": {"name": "UEFA Champions League", "espn": "uefa.champions"},
}

COLUMNS = [
    "league", "match_id", "date", "status", "neutral", "home", "away",
    "hg", "ag", "hs", "as_", "hst", "ast", "hc", "ac", "hposs", "aposs", "hf", "af",
]
_STAT_MAP = {
    "totalShots": "s", "shotsOnTarget": "st", "wonCorners": "c",
    "possessionPct": "poss", "foulsCommitted": "f",
}


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
_session = requests.Session()
_session.headers.update(HEADERS)


def _get(url: str, params: dict | None = None, retries: int = 4) -> requests.Response:
    delay = 1.0
    last_exc: Exception | None = None
    for _ in range(retries):
        try:
            r = _session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r
            last_exc = RuntimeError(f"HTTP {r.status_code} for {r.url}")
        except requests.RequestException as exc:  # network hiccup
            last_exc = exc
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"Download failed: {last_exc}")


# --------------------------------------------------------------------------- #
# ESPN
# --------------------------------------------------------------------------- #
def _month_range(start: str, end: dt.date) -> list[str]:
    y, m = (int(x) for x in start.split("-"))
    out = []
    while (y, m) <= (end.year, end.month):
        out.append(f"{y}{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def _month_closed(ym: str, today: dt.date) -> bool:
    """A month is 'closed' when it ended more than 3 days ago (stats are final)."""
    y, m = int(ym[:4]), int(ym[4:])
    first_next = dt.date(y + (m == 12), 1 if m == 12 else m + 1, 1)
    return (today - first_next).days > 3


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _parse_event(league: str, ev: dict) -> dict | None:
    try:
        comp = ev["competitions"][0]
        status = comp["status"]["type"]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        home, away = teams["home"], teams["away"]
    except (KeyError, IndexError):
        return None
    completed = bool(status.get("completed"))
    name = status.get("name", "")
    if completed:
        st = "FT"
    elif name in ("STATUS_SCHEDULED",):
        st = "SCHED"
    else:
        st = "OTHER"  # live, postponed, cancelled...
    rec = {
        "league": league,
        "match_id": str(ev.get("id")),
        "date": ev.get("date"),
        "status": st,
        "neutral": bool(comp.get("neutralSite", False)),
        "home": home["team"].get("displayName"),
        "away": away["team"].get("displayName"),
        "hg": _to_float(home.get("score")) if completed else np.nan,
        "ag": _to_float(away.get("score")) if completed else np.nan,
    }
    for side, prefix in ((home, "h"), (away, "a")):
        stats = {s.get("name"): s.get("displayValue") for s in side.get("statistics", [])}
        for espn_name, short in _STAT_MAP.items():
            col = f"{prefix}{short}"
            if col == "as":
                col = "as_"
            rec[col] = _to_float(stats.get(espn_name)) if completed else np.nan
    return rec


def fetch_espn_month(league: str, ym: str, today: dt.date, refresh: bool = False) -> list[dict]:
    slug = LEAGUES[league]["espn"]
    path = CACHE / "espn" / slug / f"{ym}.json"
    if path.exists() and not refresh:
        fresh = (time.time() - path.stat().st_mtime) < 3 * 3600  # 3h for open months
        if _month_closed(ym, today) or fresh:
            with path.open() as fh:
                return json.load(fh)
    resp = _get(ESPN_URL.format(slug=slug), params={"dates": ym, "limit": 1000})
    events = resp.json().get("events", [])
    records = [r for r in (_parse_event(league, e) for e in events) if r]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(records, fh)
    return records


def load_espn(league: str, start: str = "2022-07", refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    today = dt.date.today()
    end = (today.replace(day=1) + dt.timedelta(days=40)).replace(day=1)  # next month (upcoming fixtures)
    months = _month_range(start, end)
    records: list[dict] = []
    downloaded = 0
    for i, ym in enumerate(months, 1):
        slug = LEAGUES[league]["espn"]
        cached = (CACHE / "espn" / slug / f"{ym}.json").exists() and not refresh
        if not cached:
            downloaded += 1
        if verbose:
            print(f"\r  [{LEAGUES[league]['name']}] descargando/cargando meses {i}/{len(months)} ({ym})", end="", flush=True)
        records.extend(fetch_espn_month(league, ym, today, refresh))
    if verbose:
        print(f"\r  [{LEAGUES[league]['name']}] {len(records)} eventos ({downloaded} meses descargados, resto en caché)      ")
    df = pd.DataFrame(records, columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df = df.drop_duplicates("match_id").sort_values("date").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# football-data.co.uk fallback
# --------------------------------------------------------------------------- #
def _read_csv_cached(url: str, cache_name: str, permanent: bool, refresh: bool) -> pd.DataFrame:
    path = CACHE / "fdcouk" / cache_name
    use_cache = path.exists() and not refresh and (permanent or (time.time() - path.stat().st_mtime) < 3 * 3600)
    if use_cache:
        raw = path.read_bytes()
    else:
        raw = _get(url).content
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    try:
        return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")


def load_fdcouk(league: str, start_year: int = 2022, refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    meta = LEAGUES[league]
    today = dt.date.today()
    frames = []
    if "fdcouk" in meta:
        current_start = today.year if today.month >= 7 else today.year - 1
        for y in range(start_year, current_start + 1):
            season = f"{y % 100:02d}{(y + 1) % 100:02d}"
            df = _read_csv_cached(FDCOUK_SEASON_URL.format(season=season, code=meta["fdcouk"]),
                                  f"{meta['fdcouk']}_{season}.csv", permanent=(y < current_start), refresh=refresh)
            df = df.rename(columns={"HomeTeam": "home", "AwayTeam": "away", "FTHG": "hg", "FTAG": "ag",
                                    "HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast", "HC": "hc", "AC": "ac",
                                    "HF": "hf", "AF": "af"})
            frames.append(df)
    elif "fdcouk_new" in meta:
        df = _read_csv_cached(FDCOUK_NEW_URL.format(code=meta["fdcouk_new"]), f"{meta['fdcouk_new']}.csv",
                              permanent=False, refresh=refresh)
        df = df.rename(columns={"Home": "home", "Away": "away", "HG": "hg", "AG": "ag"})
        frames.append(df)
    else:
        raise RuntimeError(f"football-data.co.uk no cubre {meta['name']}")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["home", "away", "hg", "ag"])
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    out = pd.DataFrame({c: df[c] if c in df.columns else np.nan for c in COLUMNS})
    out["league"] = league
    out["status"] = "FT"
    out["neutral"] = False
    out["match_id"] = [f"fd-{i}" for i in range(len(out))]
    out = out[out["date"] >= pd.Timestamp(f"{start_year}-07-01")]
    if verbose:
        print(f"  [{meta['name']}] football-data.co.uk: {len(out)} partidos")
    return out.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def load_league(league: str, source: str = "auto", start: str = "2022-07", refresh: bool = False,
                verbose: bool = True) -> pd.DataFrame:
    """Return all matches (finished + scheduled) for a league with a unified schema."""
    if league not in LEAGUES:
        raise KeyError(f"Liga desconocida: {league}. Opciones: {', '.join(LEAGUES)}")
    if source in ("auto", "espn"):
        try:
            df = load_espn(league, start=start, refresh=refresh, verbose=verbose)
            if (df["status"] == "FT").sum() >= 100:
                return df
            if source == "espn":
                raise RuntimeError("ESPN devolvió muy pocos partidos")
            print("  ESPN devolvió muy pocos partidos, usando football-data.co.uk ...")
        except Exception as exc:  # noqa: BLE001
            if source == "espn":
                raise
            print(f"  ESPN no disponible ({exc}); usando football-data.co.uk ...")
    return load_fdcouk(league, start_year=int(start[:4]), refresh=refresh, verbose=verbose)


def current_teams(df: pd.DataFrame, days: int = 200) -> list[str]:
    """Teams that played or are scheduled in roughly the current season."""
    recent = df[df["date"] >= pd.Timestamp.today() - pd.Timedelta(days=days)]
    return sorted(set(recent["home"]).union(recent["away"]))


def upcoming_fixtures(df: pd.DataFrame, days: int = 14) -> pd.DataFrame:
    now = pd.Timestamp.now("UTC").tz_convert(None)
    mask = (df["status"] == "SCHED") & (df["date"] >= now - pd.Timedelta(hours=3)) & (df["date"] <= now + pd.Timedelta(days=days))
    return df.loc[mask, ["date", "home", "away", "neutral"]].sort_values("date").reset_index(drop=True)
