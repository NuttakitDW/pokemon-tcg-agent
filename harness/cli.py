"""Command line: `python -m harness {arena,package,validate}`."""
import argparse
import sys
from pathlib import Path

from harness.agents import AgentLoadError
from harness.arena import DEFAULT_MATCH_TIMEOUT, default_procs, deck_id, format_report, run_arena
from harness.decks import DeckError, load_deck
from harness.package import PackageError, build_submission, validate_submission


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    arena = sub.add_parser("arena", help="agent A vs one or more opponents")
    arena.add_argument("--a", required=True, help="builtin (random, first), a .py agent, or a submission.tar.gz")
    arena.add_argument("--b", nargs="+", default=["random"], help="one or more opponents")
    arena.add_argument("--deck-a", help="deck.csv for A (default: DECK in the agent, else its own)")
    arena.add_argument("--deck-b", help="deck.csv for every opponent")
    arena.add_argument("-n", type=int, default=100, help="matches per opponent")
    arena.add_argument("--bo", type=int, default=3, help="games per match (the playground is best-of-3)")
    arena.add_argument("--procs", type=int, default=default_procs(), help="worker processes; 0 = in this process")
    arena.add_argument("--seed", type=int, default=0, help="seeds the agents' Python random only; the engine is not seedable")
    arena.add_argument("--match-timeout", type=float, default=DEFAULT_MATCH_TIMEOUT, help="wall-clock seconds before a match counts as a CRASH")
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
    except (AgentLoadError, DeckError, PackageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
