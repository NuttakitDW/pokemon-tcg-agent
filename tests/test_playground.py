import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from harness.agents import AgentLoadError
from harness.arena import MatchResult, summarize
from harness.playground import (
    PlaygroundError,
    PlaygroundReport,
    default_opponents,
    default_out_dir,
    format_playground,
    report_to_dict,
    run_playground,
)


def _write_agents(directory, *names):
    directory.mkdir(exist_ok=True)
    for name in names:
        (directory / name).write_text("def agent(obs):\n    return [0]\n")
    return directory


def test_default_opponents_are_builtins_plus_the_other_agents(tmp_path):
    agents = _write_agents(tmp_path / "agents", "alpha.py", "beta.py", "_helper.py")
    pool = default_opponents(str(agents / "alpha.py"), agents)
    assert pool == ["random", "first", str(agents / "beta.py")]


def test_default_opponents_exclude_a_builtin_under_test(tmp_path):
    assert default_opponents("random", tmp_path / "missing") == ["first"]


def test_default_opponents_without_an_agents_dir_is_just_the_builtins(tmp_path):
    assert default_opponents("agents/mine.py", tmp_path / "missing") == ["random", "first"]


def test_run_playground_saves_report_and_one_replay_per_opponent(tmp_path):
    out = tmp_path / "run"
    report = run_playground("random", ["first", "random"], 4, procs=0, out_dir=out, replays=1)
    assert [r.spec_b for r in report.reports] == ["first", "random"]
    assert all(r.matches == 4 for r in report.reports)

    data = json.loads((out / "report.json").read_text())
    assert data["agent"] == "random" and data["bo"] == 3 and data["matches_per_opponent"] == 4
    first = data["opponents"][0]
    assert first["spec_b"] == "first" and len(first["results"]) == 4
    assert 0.0 <= first["game_rate"] <= 1.0 and "verdict" in first

    replays = sorted(p.name for p in (out / "replays").glob("*.html"))
    assert len(replays) == 2 and replays[0].startswith("00-first-") and replays[1].startswith("01-random-")
    # JSON points at replays relative to the run directory, so the folder can be moved or zipped.
    assert first["results"][0]["replay"] == f"replays/{replays[0]}"
    assert first["results"][1]["replay"] == ""


def test_run_playground_without_out_dir_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = run_playground("random", ["first"], 2, procs=0, replays=3)
    assert report.out_dir is None
    assert list(tmp_path.iterdir()) == []


def test_run_playground_fails_fast_on_a_bad_agent_before_playing(tmp_path):
    with pytest.raises(AgentLoadError):
        run_playground(str(tmp_path / "missing.py"), ["first"], 2, procs=0, out_dir=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_run_playground_fails_fast_when_a_later_opponent_cannot_load(tmp_path):
    # A typo in the last --vs entry must not throw away the opponents that already played.
    out = tmp_path / "run"
    with pytest.raises(AgentLoadError):
        run_playground("random", ["first", str(tmp_path / "missing.py")], 2, procs=0, out_dir=out, replays=1)
    assert not out.exists()  # nothing was played or written


def test_run_playground_reports_an_uncreatable_out_dir(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    with pytest.raises(PlaygroundError, match="cannot create"):
        run_playground("random", ["first"], 2, procs=0, out_dir=blocker / "run")


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_run_playground_refuses_an_unwritable_out_dir_before_playing(tmp_path):
    out = tmp_path / "read-only"
    out.mkdir()
    out.chmod(0o555)
    try:
        with pytest.raises(PlaygroundError, match="not writable"):
            run_playground("random", ["first"], 2, procs=0, out_dir=out)
    finally:
        out.chmod(0o755)


def test_a_failed_report_write_keeps_the_results(tmp_path, monkeypatch):
    def full_disk(self, *args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_text", full_disk)
    report = run_playground("random", ["first"], 2, procs=0, out_dir=tmp_path / "run")
    assert "No space left" in report.save_error and report.reports[0].matches == 2
    text = format_playground(report)
    assert "not saved" in text and "No space left" in text


def test_a_relative_out_dir_still_gives_run_relative_replay_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_playground("random", ["first"], 2, procs=0, out_dir=Path("runs/x"), replays=1)
    data = json.loads((tmp_path / "runs/x/report.json").read_text())
    assert data["opponents"][0]["results"][0]["replay"] == "replays/00-first-000.html"
    assert (tmp_path / "runs/x/replays/00-first-000.html").exists()


def _report(results, out_dir=None):
    arena = summarize("bot.py", "random", results)
    return PlaygroundReport(
        agent="bot.py", deck="abc12345", bo=3, matches=len(results), seed=0, procs=2,
        elapsed=2.0, reports=(arena,), out_dir=out_dir,
    )


def test_format_playground_has_one_row_per_opponent(tmp_path):
    report = run_playground("random", ["first"], 4, procs=0)
    text = format_playground(report)
    for needle in ("first", "win rate", "95% CI", "ms/call", "games W-L", "4 matches vs 1 opponent"):
        assert needle in text
    assert "FAULTS" not in text and not report.has_faults


def test_format_playground_header_counts_several_opponents():
    report = run_playground("random", ["first", "random"], 2, procs=0)
    assert "2 matches vs each of 2 opponents" in format_playground(report)


def test_format_playground_flags_faults_and_budget_overruns():
    fault = MatchResult(0, "loss", 0, 2, 0, 5, 400.0, 1.0, ("A:ERROR",), "A: kaboom", calls_a=10)
    report = _report([fault])
    text = format_playground(report)
    assert report.has_faults
    assert "FAULTS" in text and "A:ERROR in 1 of 1 matches" in text and "kaboom" in text
    assert "budget" in text  # 400s is over half of the 600s match budget


def test_format_playground_counts_the_matches_behind_each_fault():
    erred = MatchResult(0, "loss", 0, 2, 0, 5, 1.0, 1.0, ("A:ERROR",), "A: kaboom")
    crashed = MatchResult(1, "crash", 0, 0, 0, 0, 0.0, 0.0, ("CRASH",), "worker died")
    text = format_playground(_report([erred, erred, crashed, replace(erred, faults=())]))
    assert "A:ERROR in 2 of 4 matches" in text and "CRASH in 1 of 4 matches" in text


def test_format_playground_shows_crashes_and_undefined_seat_rates():
    crash = MatchResult(0, "crash", 0, 0, 0, 0, 0.0, 0.0, ("CRASH",), "worker died")
    text = format_playground(_report([crash, crash]))
    assert "0-0-0 +2 crashed" in text  # a crash is not a quiet 0-0-0
    assert "--/--" in text  # no decided match, so no seat rate to show


def test_format_playground_tells_same_named_opponents_apart():
    results = [MatchResult(0, "win", 2, 0, 0, 5, 0.1, 0.1, ())]
    two = replace(
        _report(results),
        reports=(summarize("bot.py", "x/bot.py", results), summarize("bot.py", "y/bot.py", results)),
    )
    text = format_playground(two)
    assert "x/bot.py" in text and "y/bot.py" in text


def test_format_playground_lists_replay_paths_and_caps_the_list():
    results = [
        MatchResult(0, "win", 2, 0, 0, 5, 0.1, 0.1, (), replay=f"run/replays/00-random-{i:03d}.html")
        for i in range(12)
    ]
    text = format_playground(_report(results, out_dir=Path("run")))
    assert "run/replays/00-random-000.html" in text and "run/replays/00-random-009.html" in text
    assert "00-random-010.html" not in text and "and 2 more" in text


def test_format_playground_surfaces_replay_failures():
    broken = MatchResult(0, "win", 2, 0, 0, 5, 0.1, 0.1, (), replay_error="OSError: disk full")
    text = format_playground(_report([broken]))
    assert "replay" in text and "disk full" in text


def test_report_to_dict_is_json_serialisable_without_an_out_dir():
    ok = MatchResult(0, "win", 2, 0, 0, 5, 0.1, 0.1, (), calls_a=40, max_call_a=0.01)
    data = report_to_dict(_report([ok]))
    json.dumps(data)
    assert data["opponents"][0]["results"][0]["calls_a"] == 40


def test_default_out_dir_is_timestamped_and_named_after_the_agent():
    path = default_out_dir("agents/greedy.py")
    assert path.parent.name == "runs" and path.name.endswith("-greedy")
    assert default_out_dir("dist/submission.tar.gz").name.endswith("-submission")
