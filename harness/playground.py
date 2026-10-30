"""Playground: one command that shows how an agent performs.

Plays the agent against a pool of opponents (the engine's builtins plus every other agent in
`agents/`) and reports win rate, game win rate, decision latency and faults in one table. A run
can be saved to a directory: `report.json` with every match, and HTML replays of the first
matches (the engine's own visualizer) to open in a browser.
"""
import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

from harness.agents import BUILTINS, load_agent
from harness.arena import (
    DEFAULT_MATCH_TIMEOUT,
    MATCH_BUDGET_SECS,
    ArenaReport,
    deck_id,
    run_arena,
)

AGENTS_DIR = Path("agents")
RUNS_DIR = Path("runs")
REPLAYS_SUBDIR = "replays"
REPORT_FILE = "report.json"
REPLAY_LIST_LIMIT = 10
SPEC_SUFFIXES = (".tar.gz", ".tgz", ".py")
COLUMNS = (
    "opponent", "W-L-D", "win rate", "95% CI", "games W-L", "seat0/seat1", "ms/call", "max match", "verdict",
)


class PlaygroundError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlaygroundReport:
    agent: str
    deck: str
    bo: int
    matches: int  # per opponent
    seed: int
    procs: int
    elapsed: float
    reports: tuple[ArenaReport, ...]
    out_dir: Path | None
    save_error: str = ""  # why report.json was not written; the results above still stand

    @property
    def has_faults(self) -> bool:
        return any(r.faults for r in self.reports)


def _label(spec: str) -> str:
    name = Path(spec).name
    for suffix in SPEC_SUFFIXES:
        name = name.removesuffix(suffix)
    return name


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)


def default_opponents(agent: str, agents_dir: Path = AGENTS_DIR) -> list[str]:
    """The builtins plus every other agent in `agents_dir`; the agent under test is left out."""
    own = Path(agent).resolve()
    siblings = sorted(
        str(path)
        for path in agents_dir.glob("*.py")
        if not path.name.startswith("_") and path.resolve() != own
    )
    return [name for name in BUILTINS if name != agent] + siblings


def default_out_dir(agent: str) -> Path:
    # Local time: the folder name is for the person reading it.
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return RUNS_DIR / f"{stamp}-{_slug(_label(agent))}"


def _replay_paths(out_dir: Path | None, number: int, opponent: str, count: int) -> list[Path]:
    if out_dir is None:
        return []
    stem = f"{number:02d}-{_slug(_label(opponent))}"
    return [out_dir / REPLAYS_SUBDIR / f"{stem}-{i:03d}.html" for i in range(count)]


def _prepare_out_dir(out_dir: Path) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PlaygroundError(f"cannot create {out_dir}: {exc}") from exc
    if not os.access(out_dir, os.W_OK | os.X_OK):
        raise PlaygroundError(f"{out_dir} is not writable")


def run_playground(
    agent: str,
    opponents: list[str],
    matches: int,
    *,
    deck: tuple[int, ...] | None = None,
    procs: int = 1,
    bo: int = 3,
    seed: int = 0,
    match_timeout: float = DEFAULT_MATCH_TIMEOUT,
    out_dir: Path | None = None,
    replays: int = 0,
) -> PlaygroundReport:
    """Play `matches` against each opponent. Nothing is written unless `out_dir` is given;
    `replays` is how many matches per opponent also get an HTML replay (the first ones)."""
    # Fail fast, before anything is played or written: a typo in the last opponent must not
    # throw away the results of the ones before it.
    loaded = load_agent(agent)
    for opponent in opponents:
        load_agent(opponent)
    if out_dir is not None:
        _prepare_out_dir(out_dir)

    started = time.monotonic()
    reports = tuple(
        run_arena(
            agent, opponent, matches,
            deck_a=deck, procs=procs, bo=bo, seed=seed, match_timeout=match_timeout,
            replay_paths=_replay_paths(out_dir, number, opponent, min(replays, matches)),
        )
        for number, opponent in enumerate(opponents)
    )
    report = PlaygroundReport(
        agent=agent, deck=deck_id(deck or loaded.deck), bo=bo, matches=matches, seed=seed,
        procs=procs, elapsed=time.monotonic() - started, reports=reports, out_dir=out_dir,
    )
    if out_dir is not None:
        try:
            (out_dir / REPORT_FILE).write_text(json.dumps(report_to_dict(report), indent=2) + "\n")
        except OSError as exc:  # e.g. disk full: the table is still worth printing
            report = replace(report, save_error=f"{type(exc).__name__}: {exc}")
    return report


def _relative(replay: str, out_dir: Path | None) -> str:
    """Replays are stored relative to the run directory, so the folder can be moved or zipped."""
    if not replay or out_dir is None:
        return replay
    return Path(replay).relative_to(out_dir).as_posix()


def _arena_to_dict(r: ArenaReport, out_dir: Path | None) -> dict:
    results = [{**asdict(m), "replay": _relative(m.replay, out_dir)} for m in r.results]
    return {
        **asdict(r),
        "verdict": r.verdict,
        "game_rate": r.game_rate,
        "ms_per_call_a": r.ms_per_call_a,
        "results": results,
    }


def report_to_dict(report: PlaygroundReport) -> dict:
    return {
        "agent": report.agent,
        "deck": report.deck,
        "bo": report.bo,
        "matches_per_opponent": report.matches,
        "seed": report.seed,
        "procs": report.procs,
        "elapsed_secs": round(report.elapsed, 3),
        "opponents": [_arena_to_dict(r, report.out_dir) for r in report.reports],
    }


def _labels(reports: tuple[ArenaReport, ...]) -> list[str]:
    """Row names. Opponents that share a file name (x/bot.py, y/bot.py) show their full spec."""
    short = [_label(r.spec_b) for r in reports]
    return [r.spec_b if short.count(name) > 1 else name for r, name in zip(reports, short)]


def _seat_cell(r: ArenaReport, seat: int) -> str:
    decided = any(m.seat_a == seat and m.outcome in ("win", "loss") for m in r.results)
    rate = r.rate_seat0 if seat == 0 else r.rate_seat1
    return f"{rate:.3f}" if decided else "--"  # 0.000 would read as "always lost"


def _row(r: ArenaReport, label: str) -> list[str]:
    record = f"{r.wins}-{r.losses}-{r.draws}" + (f" +{r.crashes} crashed" if r.crashes else "")
    return [
        label,
        record,
        f"{r.rate:.3f}",
        f"{r.low:.3f}-{r.high:.3f}",
        f"{r.games_won}-{r.games_lost}",
        f"{_seat_cell(r, 0)}/{_seat_cell(r, 1)}",
        f"{r.ms_per_call_a:.3f}",
        f"{r.max_secs_a:.1f}s",
        r.verdict,
    ]


def _table(rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows]


def _fault_summary(r: ArenaReport) -> str:
    counts = Counter(fault for m in r.results for fault in set(m.faults))
    return ", ".join(f"{fault} in {n} of {r.matches} matches" for fault, n in sorted(counts.items()))


def _problems(report: PlaygroundReport, labels: list[str]) -> list[str]:
    lines: list[str] = []
    for r, label in zip(report.reports, labels):
        if r.faults:
            lines += ["", f"FAULTS vs {label}: {_fault_summary(r)}"]
            lines += [f"    {line}" for line in r.fault_detail.splitlines()]
        if r.max_secs_a > MATCH_BUDGET_SECS / 2:
            lines += [
                "",
                (
                    f"SLOW vs {label}: A's slowest match took {r.max_secs_a:.0f}s, "
                    f"over half the {MATCH_BUDGET_SECS}s match budget"
                ),
            ]
    errors = [m.replay_error for r in report.reports for m in r.results if m.replay_error]
    if errors:
        plural = "" if len(errors) == 1 else "s"
        lines += ["", f"{len(errors)} replay{plural} not saved; first error: {errors[0]}"]
    return lines


def _footer(report: PlaygroundReport) -> list[str]:
    total = report.matches * len(report.reports)
    speed = total / report.elapsed if report.elapsed > 0 else 0.0
    where = f"{report.procs} processes" if report.procs > 0 else "this process"
    lines = ["", f"{total} matches in {report.elapsed:.1f}s on {where} ({speed:.1f} matches/s)"]
    if report.out_dir is None:
        return lines
    if report.save_error:
        lines.append(f"{REPORT_FILE} not saved: {report.save_error}")
    else:
        lines.append(f"saved {report.out_dir / REPORT_FILE}")
    saved = [m.replay for r in report.reports for m in r.results if m.replay]
    if saved:
        lines.append("replays (open one in a browser):")
        lines += [f"  {path}" for path in saved[:REPLAY_LIST_LIMIT]]
        if len(saved) > REPLAY_LIST_LIMIT:
            lines.append(f"  ... and {len(saved) - REPLAY_LIST_LIMIT} more in {report.out_dir / REPLAYS_SUBDIR}")
    return lines


def format_playground(report: PlaygroundReport) -> str:
    count = len(report.reports)
    opponents = "1 opponent" if count == 1 else f"each of {count} opponents"
    labels = _labels(report.reports)
    lines = [
        (
            f"A={report.agent} [deck {report.deck}]   best-of-{report.bo}   "
            f"{report.matches} matches vs {opponents}   seed {report.seed}"
        ),
        "",
        *_table([list(COLUMNS), *(_row(r, label) for r, label in zip(report.reports, labels))]),
        *_problems(report, labels),
        *_footer(report),
    ]
    return "\n".join(lines)
