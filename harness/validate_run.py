"""Run inside a clean interpreter: load main.py with Kaggle's loader and play real matches.

Self-contained on purpose (executed with `python -I`, so the harness package is not importable).
Usage: validate_run.py <path-to-main.py> <games>. Prints one JSON line.
"""
import json
import os
import sys

from kaggle_environments import make
from kaggle_environments.agent import get_last_callable
from kaggle_environments.envs.cabt.cg.sim import Battle

STDERR_TAIL_CHARS = 500


def stderr_tail(env, seat):
    for step in reversed(env.logs or []):
        entry = step[seat] if seat < len(step) else None
        if entry and entry.get("stderr"):
            return entry["stderr"].strip()[-STDERR_TAIL_CHARS:]
    return ""


def main(main_path: str, games: int) -> dict:
    checks = []
    with open(main_path) as f:
        source = f.read()
    fn = get_last_callable(source, path=main_path)  # same loader as the competition runner
    checks.append({"name": "load", "ok": fn.__name__ == "agent", "detail": f"last callable is {fn.__name__!r}"})

    with open(os.path.join(os.path.dirname(main_path), "deck.csv")) as f:
        submitted_deck = [int(x) for x in f.read().split()]
    statuses, used_deck, failure = [], True, ""
    for i in range(games):
        seat = i % 2
        env = make("cabt", configuration={"bo": 3}, debug=False)
        env.run([main_path, "random"] if seat == 0 else ["random", main_path])
        statuses.append(env.state[seat].status)
        used_deck = used_deck and list(Battle.decks[seat]) == submitted_deck
        failure = failure or (stderr_tail(env, seat) if statuses[-1] != "DONE" else "")
    clean = all(s == "DONE" for s in statuses)
    detail = f"{games} matches, statuses {sorted(set(statuses))}" + (f" | {failure}" if failure else "")
    checks.append({"name": "play", "ok": clean, "detail": detail})
    checks.append({"name": "deck used", "ok": used_deck, "detail": "engine received deck.csv"})
    return {"checks": checks}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], int(sys.argv[2]))))
