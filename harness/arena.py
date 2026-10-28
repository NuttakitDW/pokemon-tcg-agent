"""Play two agents against each other over many matches and report win rates.

The cabt engine keeps its game state in process-global memory, so matches run in worker
*processes*; threads corrupt the engine and kill the interpreter with SIGSEGV. Workers are
supervised: one that dies (native crash, OOM) or overruns the wall-clock limit is replaced and
its match is reported as a CRASH instead of hanging the whole run.
"""
import hashlib
import multiprocessing as mp
import os
import random
import time
import traceback
from collections import deque
from dataclasses import dataclass
from multiprocessing.connection import wait

from harness.agents import load_agent, timed, with_deck
from harness.stats import wilson

FAULT_STATUSES = frozenset({"INVALID", "ERROR", "TIMEOUT"})
MATCH_BUDGET_SECS = 600  # per agent per match, enforced by the competition
DEFAULT_MATCH_TIMEOUT = 2 * MATCH_BUDGET_SECS + 60  # wall clock for a whole match, both agents
STDERR_TAIL_CHARS = 600


@dataclass(frozen=True)
class MatchTask:
    spec_a: str
    spec_b: str
    deck_a: tuple[int, ...] | None
    deck_b: tuple[int, ...] | None
    seat_a: int  # which engine seat (0 or 1) agent A plays
    bo: int
    seed: int


@dataclass(frozen=True)
class MatchResult:
    seat_a: int
    outcome: str  # "win" | "loss" | "draw" (A's view) or "crash" (worker died / overran)
    games_a: int
    games_b: int
    draws: int
    steps: int
    secs_a: float
    secs_b: float
    faults: tuple[str, ...]  # e.g. ("A:INVALID",) or ("CRASH",)
    detail: str = ""  # agent stderr tail, or why the worker was lost


@dataclass(frozen=True)
class ArenaReport:
    spec_a: str
    spec_b: str
    deck_a: str
    deck_b: str
    matches: int
    wins: int
    losses: int
    draws: int
    crashes: int
    rate: float  # wins / (wins + losses); draws and crashes are excluded
    low: float
    high: float
    rate_seat0: float
    rate_seat1: float
    mean_steps: float
    max_secs_a: float
    max_secs_b: float
    faults: tuple[str, ...]
    fault_detail: str

    @property
    def verdict(self) -> str:
        if self.low > 0.5:
            return "A better (95%)"
        if self.high < 0.5:
            return "B better (95%)"
        return "no significant difference"


def deck_id(deck: tuple[int, ...] | None) -> str:
    """Short fingerprint, so a report shows which deck was really played."""
    if deck is None:
        return "agent-chosen"
    return hashlib.sha1(",".join(map(str, deck)).encode()).hexdigest()[:8]


def _fault_outcome(faults: tuple[str, ...]) -> str | None:
    a_faulted = any(f.startswith("A:") for f in faults)
    b_faulted = any(f.startswith("B:") for f in faults)
    if a_faulted and b_faulted:
        return "draw"
    if a_faulted:
        return "loss"
    if b_faulted:
        return "win"
    return None


def _stderr_tail(env, seat: int) -> str:
    for step in reversed(env.logs or []):
        entry = step[seat] if seat < len(step) else None
        if entry and entry.get("stderr"):
            return entry["stderr"].strip()[-STDERR_TAIL_CHARS:]
    return ""


def play_match(task: MatchTask) -> MatchResult:
    """One best-of-`bo` match in the current process."""
    from kaggle_environments import make

    random.seed(task.seed)  # seeds agent-side randomness only; the engine's shuffles are not seedable
    loaded = (load_agent(task.spec_a), load_agent(task.spec_b))
    decks = (task.deck_a or loaded[0].deck, task.deck_b or loaded[1].deck)
    fns, clocks = [], []
    for agent, deck in zip(loaded, decks):
        fn = with_deck(agent.fn, deck) if deck is not None else agent.fn
        timed_fn, clock = timed(fn)
        fns.append(timed_fn)
        clocks.append(clock)

    seat_a = task.seat_a
    seats = [fns[0], fns[1]] if seat_a == 0 else [fns[1], fns[0]]
    env = make("cabt", configuration={"bo": task.bo}, debug=False)
    env.run(seats)

    labels = ("A", "B") if seat_a == 0 else ("B", "A")
    faulted = [seat for seat, state in enumerate(env.state) if state.status in FAULT_STATUSES]
    faults = tuple(f"{labels[seat]}:{env.state[seat].status}" for seat in faulted)
    detail = "\n".join(
        f"{labels[seat]}: {tail}" for seat in faulted if (tail := _stderr_tail(env, seat))
    )
    wins_by_seat = list(getattr(env, "result", None) or [0, 0, 0])

    outcome = _fault_outcome(faults)
    if outcome is None:
        outcome = {1: "win", -1: "loss"}.get(env.state[seat_a].reward, "draw")
    return MatchResult(
        seat_a=seat_a,
        outcome=outcome,
        games_a=wins_by_seat[seat_a],
        games_b=wins_by_seat[1 - seat_a],
        draws=wins_by_seat[2],
        steps=len(env.steps),
        secs_a=clocks[0]["secs"],
        secs_b=clocks[1]["secs"],
        faults=faults,
        detail=detail,
    )


def _crash_result(task: MatchTask, why: str) -> MatchResult:
    return MatchResult(task.seat_a, "crash", 0, 0, 0, 0, 0.0, 0.0, ("CRASH",), why)


def _worker_main(conn) -> None:
    while True:
        task = conn.recv()
        try:
            conn.send(("ok", play_match(task)))
        except Exception:
            conn.send(("error", traceback.format_exc()[-STDERR_TAIL_CHARS:]))


@dataclass
class _Slot:
    process: mp.Process
    conn: object
    index: int | None = None
    task: MatchTask | None = None
    deadline: float = 0.0


def _start_slot(ctx) -> _Slot:
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_worker_main, args=(child,), daemon=True)
    process.start()
    child.close()
    return _Slot(process, parent)


def _retire(slot: _Slot) -> None:
    if slot.process.is_alive():
        slot.process.kill()
    slot.process.join(timeout=5)
    slot.conn.close()


def run_supervised(tasks: list[MatchTask], procs: int, timeout: float) -> list[MatchResult]:
    """Run tasks on `procs` worker processes; a lost worker costs one CRASH, not the run."""
    ctx = mp.get_context("spawn")
    pending = deque(enumerate(tasks))
    results: dict[int, MatchResult] = {}
    slots = [_start_slot(ctx) for _ in range(min(procs, len(tasks)))]
    try:
        while len(results) < len(tasks):
            for slot in slots:
                if slot.index is None and pending:
                    slot.index, slot.task = pending.popleft()
                    slot.deadline = time.monotonic() + timeout
                    slot.conn.send(slot.task)
            busy = [s for s in slots if s.index is not None]
            wait([s.conn for s in busy] + [s.process.sentinel for s in busy], timeout=1.0)
            for position, slot in enumerate(slots):
                if slot.index is None:
                    continue
                lost = None
                if slot.conn.poll():
                    try:
                        kind, payload = slot.conn.recv()
                    except (EOFError, OSError):
                        lost = f"worker died (exit code {slot.process.exitcode})"
                    else:
                        results[slot.index] = (
                            payload if kind == "ok" else _crash_result(slot.task, payload)
                        )
                        slot.index, slot.task = None, None
                elif not slot.process.is_alive():
                    lost = f"worker died (exit code {slot.process.exitcode})"
                elif time.monotonic() > slot.deadline:
                    lost = f"match exceeded {timeout:.0f}s wall clock"
                if lost:
                    results[slot.index] = _crash_result(slot.task, lost)
                    _retire(slot)
                    slots[position] = _start_slot(ctx)
    finally:
        for slot in slots:
            _retire(slot)
    return [results[i] for i in range(len(tasks))]


def summarize(
    spec_a: str, spec_b: str, results: list[MatchResult], deck_a: str = "", deck_b: str = ""
) -> ArenaReport:
    def rate(subset: list[MatchResult]) -> float:
        won = sum(r.outcome == "win" for r in subset)
        decided = sum(r.outcome in ("win", "loss") for r in subset)
        return won / decided if decided else 0.0

    def count(outcome: str) -> int:
        return sum(r.outcome == outcome for r in results)

    wins, losses = count("win"), count("loss")
    p, low, high = wilson(wins, wins + losses)
    played = [r for r in results if r.outcome != "crash"]
    return ArenaReport(
        spec_a=spec_a,
        spec_b=spec_b,
        deck_a=deck_a,
        deck_b=deck_b,
        matches=len(results),
        wins=wins,
        losses=losses,
        draws=count("draw"),
        crashes=count("crash"),
        rate=p,
        low=low,
        high=high,
        rate_seat0=rate([r for r in results if r.seat_a == 0]),
        rate_seat1=rate([r for r in results if r.seat_a == 1]),
        mean_steps=sum(r.steps for r in played) / max(len(played), 1),
        max_secs_a=max((r.secs_a for r in results), default=0.0),
        max_secs_b=max((r.secs_b for r in results), default=0.0),
        faults=tuple(sorted({f for r in results for f in r.faults})),
        fault_detail=next((r.detail for r in results if r.detail), ""),
    )


def run_arena(
    spec_a: str,
    spec_b: str,
    matches: int,
    *,
    deck_a: tuple[int, ...] | None = None,
    deck_b: tuple[int, ...] | None = None,
    procs: int = 1,
    bo: int = 3,
    seed: int = 0,
    match_timeout: float = DEFAULT_MATCH_TIMEOUT,
) -> ArenaReport:
    """Seats alternate every match so first-seat effects cancel out.

    `procs=0` plays in this process (handy under a debugger; a crash then takes the run with it).
    """
    loaded_a, loaded_b = load_agent(spec_a), load_agent(spec_b)  # fail fast on a bad spec
    tasks = [
        MatchTask(spec_a, spec_b, deck_a, deck_b, i % 2, bo, seed * 1_000_003 + i)
        for i in range(matches)
    ]
    results = (
        [play_match(t) for t in tasks]
        if procs <= 0
        else run_supervised(tasks, procs, match_timeout)
    )
    return summarize(
        spec_a, spec_b, results,
        deck_id(deck_a or loaded_a.deck), deck_id(deck_b or loaded_b.deck),
    )


def format_report(r: ArenaReport) -> str:
    slow = max(r.max_secs_a, r.max_secs_b) > MATCH_BUDGET_SECS / 2
    lines = [
        f"A={r.spec_a} [deck {r.deck_a}]  vs  B={r.spec_b} [deck {r.deck_b}]   ({r.matches} matches)",
        f"  A wins {r.wins}  losses {r.losses}  draws {r.draws}  crashes {r.crashes}",
        f"  win rate  {r.rate:.3f}  95% CI [{r.low:.3f}, {r.high:.3f}]  -> {r.verdict}",
        f"  as seat 0 {r.rate_seat0:.3f}   as seat 1 {r.rate_seat1:.3f}",
        f"  mean steps {r.mean_steps:.0f}   slowest match: A {r.max_secs_a:.1f}s  B {r.max_secs_b:.1f}s"
        + (" (over half the 600s match budget)" if slow else ""),
    ]
    if r.faults:
        lines.append(f"  FAULTS: {', '.join(r.faults)}")
    if r.fault_detail:
        lines.append("  first fault detail:\n" + "\n".join(f"    {l}" for l in r.fault_detail.splitlines()))
    return "\n".join(lines)


def default_procs() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))
