"""Player data layer.

Source: ESPN public match-summary endpoint (no API key). For every match it gives
both squads: who started, who came on, position, formation and per-player stats
(goals, assists, shots, shots on target).

Compact cache: ``data_cache/lineups/{slug}/{YYYYMM}.json.gz``. Finished matches never
change, so only matches that are not cached yet are downloaded on each run.

ESPN answers 403 when it receives too many requests in a short time, so downloads are
paced (a few per second) and stop as soon as ESPN asks to slow down.
"""
from __future__ import annotations

import gzip
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from .data import CACHE, LEAGUES, _session

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/summary"
LINEUP_CACHE = CACHE / "lineups"
STARTER, SUB_IN, BENCH = 2, 1, 0
REQUESTS_PER_SECOND = 4.0
COOL_DOWN = 150  # seconds to wait when ESPN asks to slow down


class RateLimited(RuntimeError):
    """ESPN refused the request because too many were sent."""


class _Throttle:
    def __init__(self, per_second: float):
        self.gap, self.lock, self.next = 1.0 / per_second, threading.Lock(), 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            slot = max(self.next, now)
            self.next = slot + self.gap
        time.sleep(max(0.0, slot - now))


_throttle = _Throttle(REQUESTS_PER_SECOND)


def pos_group(pos: str) -> str:
    """ESPN position abbreviation (G, CD-L, LB, DM, AM-R, F, ...) -> G / D / M / F / U (unknown)."""
    p = (pos or "").upper().split("-")[0]
    if p in ("G", "GK"):
        return "G"
    if p in ("D", "CD", "CB", "LB", "RB", "SW", "LWB", "RWB", "WB"):
        return "D"
    if p in ("M", "DM", "CM", "LM", "RM", "AM"):
        return "M"
    if p in ("F", "CF", "LF", "RF", "ST", "S", "LW", "RW", "W", "SS"):
        return "F"
    return "U"


def _num(stats: dict, key: str) -> int:
    try:
        return int(stats.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def parse_lineups(summary: dict) -> tuple[dict | None, dict]:
    """ESPN summary JSON -> ({"h": side, "a": side}, {player_id: name}); None when no XI is published.

    side = {"t": team_id, "f": formation, "p": [[player_id, role, position, goals, assists, shots, sot], ...]}
    """
    rec, names = {}, {}
    for side in summary.get("rosters") or []:
        key = {"home": "h", "away": "a"}.get(side.get("homeAway"))
        players = []
        for p in side.get("roster") or []:
            ath = p.get("athlete") or {}
            if not ath.get("id"):
                continue
            pid = int(ath["id"])
            names[str(pid)] = ath.get("displayName") or ath.get("fullName") or str(pid)
            role = STARTER if p.get("starter") else SUB_IN if p.get("subbedIn") else BENCH
            s = {x.get("name"): x.get("value") for x in p.get("stats") or []}
            pos = (p.get("position") or {}).get("abbreviation", "") if role == STARTER else ""
            players.append([pid, role, pos, _num(s, "totalGoals"), _num(s, "goalAssists"),
                            _num(s, "totalShots"), _num(s, "shotsOnTarget")])
        if key and sum(1 for x in players if x[1] == STARTER) == 11:
            rec[key] = {"t": str((side.get("team") or {}).get("id", "")), "f": side.get("formation") or "", "p": players}
    return (rec if len(rec) == 2 else None), names


def fetch_lineup(league: str, match_id: str) -> tuple[dict | None, dict]:
    _throttle.wait()
    resp = _session.get(SUMMARY_URL.format(slug=LEAGUES[league]["espn"]), params={"event": match_id}, timeout=30)
    if resp.status_code in (403, 429):
        raise RateLimited(f"ESPN pidió bajar el ritmo (HTTP {resp.status_code})")
    resp.raise_for_status()
    return parse_lineups(resp.json())


def _read(path) -> dict:
    if not path.exists():
        return {"names": {}, "m": {}}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _write(path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:  # mtime=0: stable bytes for git
        gz.write(json.dumps(blob, separators=(",", ":"), ensure_ascii=False, sort_keys=True).encode("utf-8"))


def _fetch_many(league: str, ids: list[str], workers: int, stop: threading.Event) -> dict:
    """{match_id: (record, names)} for the matches that could be downloaded. Sets ``stop`` when rate limited."""
    def one(mid: str):
        if stop.is_set():
            return mid, None
        try:
            return mid, fetch_lineup(league, mid)
        except RateLimited:
            stop.set()
        except (requests.RequestException, ValueError):
            pass  # network problem: this match is retried on the next run
        return mid, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return {mid: res for mid, res in pool.map(one, ids) if res is not None}


def load_lineups(data: dict[str, pd.DataFrame], refresh: bool = False, verbose: bool = True, workers: int = 3,
                 patient: bool = False) -> dict:
    """Line-ups of every finished match. Returns {"matches": {match_id: record}, "names": {player_id: name}}.

    When ESPN asks to slow down, ``patient=True`` waits and continues (first full download);
    otherwise the download stops and the matches already cached are used.
    """
    matches, names = {}, {}
    now = pd.Timestamp.now("UTC").tz_convert(None)
    stop, pauses = threading.Event(), 0
    for league, df in data.items():
        done = df[(df["status"] == "FT") & ~df["match_id"].astype(str).str.startswith("fd-")]
        if done.empty:
            continue
        slug, fetched, total = LEAGUES[league]["espn"], 0, 0
        for ym, grp in done.groupby(done["date"].dt.strftime("%Y%m"), sort=True):
            path = LINEUP_CACHE / slug / f"{ym}.json.gz"
            blob = {"names": {}, "m": {}} if refresh else _read(path)
            ids = list(grp["match_id"].astype(str))
            age = dict(zip(ids, (now - grp["date"]).dt.days))
            todo = [i for i in ids if i not in blob["m"]]
            while todo and not stop.is_set():
                for mid, (rec, nm) in _fetch_many(league, todo, workers, stop).items():
                    fetched += 1
                    if rec is not None:
                        blob["m"][mid] = rec
                        blob["names"].update(nm)
                    elif age[mid] > 3:
                        blob["m"][mid] = None  # ESPN has no line-up for this old match: do not ask again
                _write(path, blob)
                todo = [i for i in todo if i not in blob["m"]]
                if stop.is_set() and patient and pauses < 12:
                    pauses += 1
                    if verbose:
                        print(f"\n  ESPN pidió bajar el ritmo; pausa de {COOL_DOWN} s ({pauses}) y continúo...", flush=True)
                    time.sleep(COOL_DOWN)
                    stop.clear()
                else:
                    break
            got = {k: v for k, v in blob["m"].items() if v and k in set(ids)}
            matches.update(got)
            names.update(blob["names"])
            total += len(got)
            if verbose:
                print(f"\r  [{LEAGUES[league]['name']}] alineaciones {ym}: {total} partidos ({fetched} descargados)", end="", flush=True)
        if verbose:
            print(f"\r  [{LEAGUES[league]['name']}] alineaciones de {total}/{len(done)} partidos ({fetched} descargados, resto en caché)      ")
    if stop.is_set() and verbose:
        print("  ESPN limitó las descargas: se usan las alineaciones ya guardadas y el resto se baja en la próxima ejecución.")
    return {"matches": matches, "names": names}
