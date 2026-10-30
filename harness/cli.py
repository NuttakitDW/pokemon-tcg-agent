"""Command line: `python -m harness {playground,arena,package,validate}`."""
import argparse
import sys
from pathlib import Path

from harness.agents import AgentLoadError
from harness.arena import DEFAULT_MATCH_TIMEOUT, default_procs, deck_id, format_report, run_arena
from harness.decks import DeckError, load_deck
from harness.package import PackageError, build_submission, validate_submission
from harness.playground import (
    PlaygroundError,
    default_opponents,
    default_out_dir,
    format_playground,
    run_playground,
)


def _optional_deck(path: str | None):
    return load_deck(path) if path else None


def _cmd_arena(args: argparse.Namespace) -> int:
    deck_a, deck_b = _optional_deck(args.deck_a), _optional_deck(args.deck_b)
    reports = [
        run_arena(
            args.a, opponent, args.n,
            deck_a=deck_a, deck_b=deck_b, procs=args.procs, bo=args.bo, seed=args.seed,
            match_timeout=args.match_timeout,
        )
        for opponent in args.b
    ]
    print("\n\n".join(format_report(r) for r in reports))
    return 1 if any(r.faults for r in reports) else 0


def _cmd_playground(args: argparse.Namespace) -> int:
    report = run_playground(
        args.agent, args.vs or default_opponents(args.agent), args.n,
        deck=_optional_deck(args.deck), procs=args.procs, bo=args.bo, seed=args.seed,
        match_timeout=args.match_timeout,
        out_dir=Path(args.out) if args.out else default_out_dir(args.agent),
        replays=args.replays,
    )
    print(format_playground(report))
    if report.save_error:
        return 2  # the table printed, but the run was not saved
    return 1 if report.has_faults else 0


def _cmd_package(args: argparse.Namespace) -> int:
    out = build_submission(
        Path(args.agent), Path(args.deck), Path(args.out), include=tuple(Path(p) for p in args.include)
    )
    print(f"built {out} ({out.stat().st_size / 1024:.1f} KiB), deck {deck_id(load_deck(args.deck))}")
    if not args.no_validate:
        return _report_validation(out, args.games)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    return _report_validation(Path(args.tarball), args.games)


def _report_validation(tarball: Path, games: int) -> int:
    report = validate_submission(tarball, games=games)
    for check in report.checks:
        print(f"  [{'ok' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    return 0 if report.ok else 1


def _int_at_least(minimum: int):
    """argparse type: an integer of at least `minimum`, rejected before any match is played."""

    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"{value} is below the minimum of {minimum}")
        return value

    return parse


def _add_match_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-n", type=_int_at_least(1), default=100, help="matches per opponent")
    parser.add_argument("--bo", type=_int_at_least(1), default=3, help="games per match (the playground is best-of-3)")
    parser.add_argument("--procs", type=_int_at_least(0), default=default_procs(), help="worker processes; 0 = in this process")
    parser.add_argument("--seed", type=int, default=0, help="seeds the agents' Python random only; the engine is not seedable")
    parser.add_argument("--match-timeout", type=float, default=DEFAULT_MATCH_TIMEOUT, help="wall-clock seconds before a match counts as a CRASH")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    playground = sub.add_parser(
        "playground", help="agent vs a pool of opponents: performance table, JSON report, HTML replays"
    )
    playground.add_argument("agent", help="builtin (random, first), a .py agent, or a submission.tar.gz")
    playground.add_argument("--vs", nargs="+", help="opponents (default: the builtins and every other agents/*.py)")
    playground.add_argument("--deck", help="deck.csv for the agent (default: DECK in the agent, else its own)")
    playground.add_argument("--out", help="run directory (default: runs/<timestamp>-<agent>)")
    playground.add_argument("--replays", type=_int_at_least(0), default=1, help="HTML replays to save per opponent; 0 = none")
    _add_match_options(playground)
    playground.set_defaults(run=_cmd_playground)

    arena = sub.add_parser("arena", help="agent A vs one or more opponents")
    arena.add_argument("--a", required=True, help="builtin (random, first), a .py agent, or a submission.tar.gz")
    arena.add_argument("--b", nargs="+", default=["random"], help="one or more opponents")
    arena.add_argument("--deck-a", help="deck.csv for A (default: DECK in the agent, else its own)")
    arena.add_argument("--deck-b", help="deck.csv for every opponent")
    _add_match_options(arena)
    arena.set_defaults(run=_cmd_arena)

    package = sub.add_parser("package", help="build and validate submission.tar.gz")
    package.add_argument("agent", help="path to a .py defining agent(obs)")
    package.add_argument("--deck", required=True, help="deck.csv to submit")
    package.add_argument("--out", default="dist/submission.tar.gz")
    package.add_argument("--include", nargs="*", default=[], help="extra files (weights, helpers)")
    package.add_argument("--games", type=int, default=4, help="validation matches")
    package.add_argument("--no-validate", action="store_true")
    package.set_defaults(run=_cmd_package)

    validate = sub.add_parser("validate", help="check an existing submission.tar.gz")
    validate.add_argument("tarball")
    validate.add_argument("--games", type=int, default=4)
    validate.set_defaults(run=_cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except (AgentLoadError, DeckError, PackageError, PlaygroundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
