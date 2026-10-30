from pathlib import Path

import pytest

from harness.arena import MatchTask, deck_id, format_report, play_match, run_arena, summarize
from harness.decks import default_deck


def test_match_between_builtins_completes_cleanly():
    result = play_match(MatchTask("random", "random", None, None, 0, 3, 1))
    assert result.faults == ()
    assert result.outcome in {"win", "loss"}
    assert result.games_a + result.games_b + result.draws >= 2  # best-of-3 needs 2+ games


def test_best_of_one_plays_exactly_one_game():
    result = play_match(MatchTask("random", "random", None, None, 0, 1, 2))
    assert result.games_a + result.games_b + result.draws == 1


def test_seat_swap_keeps_results_in_agent_a_perspective():
    # games_a must follow agent A to whichever seat it sits in.
    for seat in (0, 1):
        r = play_match(MatchTask("random", "random", None, None, seat, 3, 10 + seat))
        assert (r.outcome == "win") == (r.games_a > r.games_b)


def test_explicit_deck_reaches_the_engine():
    # An unknown card ID is rejected by the engine; only A's override can cause that fault.
    bad = tuple([999_999] * 60)
    result = play_match(MatchTask("random", "random", bad, None, 0, 3, 3))
    assert result.faults == ("A:INVALID",) and result.outcome == "loss"
    result = play_match(MatchTask("random", "random", None, bad, 1, 3, 3))
    assert result.faults == ("B:INVALID",) and result.outcome == "win"


def test_wrong_size_deck_is_a_fault_not_a_crash():
    result = play_match(MatchTask("random", "random", tuple([3] * 59), None, 0, 3, 4))
    assert result.faults == ("A:INVALID",)


def test_run_arena_alternates_seats_and_summarizes():
    report = run_arena("random", "first", 6, deck_a=default_deck(), procs=0)
    assert report.matches == 6 and report.wins + report.losses + report.draws == 6
    assert 0.0 <= report.low <= report.rate <= report.high <= 1.0
    assert "win rate" in format_report(report)


def test_run_arena_with_worker_processes():
    report = run_arena("random", "random", 4, procs=2)
    assert report.matches == 4 and report.faults == ()


def test_summarize_reports_faults_and_excludes_draws_from_rate():
    from harness.arena import MatchResult

    def result(outcome, seat=0, faults=()):
        return MatchResult(seat, outcome, 0, 0, 0, 1, 0.0, 0.0, faults)

    report = summarize(
        "a", "b",
        [result("win"), result("draw", 1), result("loss", 1, ("A:ERROR",)), result("crash", 0, ("CRASH",))],
    )
    assert (report.wins, report.losses, report.draws, report.crashes) == (1, 1, 1, 1)
    assert report.rate == 0.5 and report.faults == ("A:ERROR", "CRASH")
    assert "FAULTS" in format_report(report)


def _agent_file(tmp_path, body):
    path = tmp_path / "bot.py"
    path.write_text(body)
    return str(path)


def test_dead_worker_is_reported_as_a_crash_not_a_hang(tmp_path):
    bot = _agent_file(
        tmp_path,
        "import os\nn = [0]\ndef agent(obs):\n    n[0] += 1\n    if n[0] == 5:\n        os._exit(139)\n    return [0]\n",
    )
    report = run_arena(bot, "random", 2, deck_a=default_deck(), procs=2, match_timeout=60)
    assert report.crashes >= 1 and "CRASH" in report.faults
    assert "exit code" in report.fault_detail


def test_hung_agent_is_killed_at_the_wall_clock_limit(tmp_path):
    bot = _agent_file(
        tmp_path, "import time\ndef agent(obs):\n    time.sleep(3600)\n",
    )
    report = run_arena(bot, "random", 1, deck_a=default_deck(), procs=1, match_timeout=3)
    assert report.crashes == 1 and "exceeded 3s" in report.fault_detail


def test_agent_exception_surfaces_its_traceback(tmp_path):
    bot = _agent_file(
        tmp_path,
        "def agent(obs):\n    if obs['select'] is not None:\n        raise RuntimeError('kaboom')\n"
        "    from kaggle_environments.envs.cabt.cabt import deck\n    return deck\n",
    )
    report = run_arena(bot, "random", 2, procs=0)
    assert "A:ERROR" in report.faults and "kaboom" in report.fault_detail


def test_unloadable_spec_fails_fast(tmp_path):
    from harness.agents import AgentLoadError

    with pytest.raises(AgentLoadError):
        run_arena(str(tmp_path / "missing.py"), "random", 1, procs=2)


def test_arena_plays_a_packaged_tarball(tmp_path):
    from harness.package import build_submission

    out = build_submission(
        Path("agents/greedy.py"), Path("decks/default.csv"), tmp_path / "s.tar.gz"
    )
    report = run_arena(str(out), "random", 4, procs=0)
    assert report.faults == () and report.deck_a == deck_id(default_deck())


def test_match_reports_decision_count_and_slowest_call():
    result = play_match(MatchTask("random", "random", None, None, 0, 3, 1))
    assert result.calls_a > 1  # deck selection plus at least one decision
    assert 0.0 <= result.max_call_a <= result.secs_a


def test_replay_is_written_only_when_asked(tmp_path):
    plain = play_match(MatchTask("random", "random", None, None, 0, 1, 5))
    assert plain.replay == "" and plain.replay_error == ""

    target = tmp_path / "nested" / "match.html"
    result = play_match(MatchTask("random", "random", None, None, 0, 1, 5, str(target)))
    assert result.replay == str(target) and result.replay_error == ""
    html = target.read_text(encoding="utf-8")
    assert "window.kaggle" in html and '"name": "cabt"' in html  # the engine's own viewer page


def test_unwritable_replay_is_reported_without_losing_the_match(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    result = play_match(MatchTask("random", "random", None, None, 0, 1, 6, str(blocker / "m.html")))
    assert result.replay == "" and result.replay_error
    assert result.faults == () and result.outcome != "crash"


def test_failed_replay_write_leaves_no_partial_file_behind(tmp_path, monkeypatch):
    real_write_text = Path.write_text

    def half_then_full_disk(self, data, *args, **kwargs):
        real_write_text(self, data[: len(data) // 2], *args, **kwargs)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_text", half_then_full_disk)
    result = play_match(MatchTask("random", "random", None, None, 0, 1, 7, str(tmp_path / "m.html")))
    assert result.replay == "" and "No space left" in result.replay_error
    assert result.faults == () and list(tmp_path.iterdir()) == []


def test_workers_leave_no_extracted_tarballs_behind(tmp_path, monkeypatch):
    from harness.arena import run_supervised
    from harness.package import build_submission

    tarball = build_submission(Path("agents/greedy.py"), Path("decks/default.csv"), tmp_path / "s.tar.gz")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))  # inherited by the spawned workers
    tasks = [MatchTask(str(tarball), "random", None, None, i % 2, 1, i) for i in range(6)]
    results = run_supervised(tasks, procs=2, timeout=120)
    assert all(r.outcome != "crash" for r in results)
    assert list(scratch.glob("agent-*")) == []


def test_run_arena_saves_replays_for_the_first_matches_only(tmp_path):
    paths = [tmp_path / "m0.html", tmp_path / "m1.html"]
    report = run_arena("random", "first", 3, procs=0, replay_paths=paths)
    assert [r.replay for r in report.results] == [str(paths[0]), str(paths[1]), ""]
    assert all(p.stat().st_size > 0 for p in paths)


def test_replays_are_written_by_worker_processes(tmp_path):
    path = tmp_path / "m0.html"
    report = run_arena("random", "first", 2, procs=2, replay_paths=[path])
    assert report.crashes == 0 and path.exists()
    assert report.results[0].replay == str(path)


def test_summarize_aggregates_games_decisions_and_latency():
    from harness.arena import MatchResult

    results = [
        MatchResult(0, "win", 2, 1, 0, 10, 1.0, 0.0, (), calls_a=100, max_call_a=0.05),
        MatchResult(1, "loss", 0, 2, 0, 8, 3.0, 0.0, (), calls_a=300, max_call_a=0.2),
        MatchResult(0, "crash", 0, 0, 0, 0, 0.0, 0.0, ("CRASH",), "worker died"),
    ]
    report = summarize("a", "b", results)
    assert (report.games_won, report.games_lost) == (2, 3)
    assert report.game_rate == pytest.approx(0.4)
    assert report.calls_a == 400 and report.total_secs_a == pytest.approx(4.0)
    assert report.ms_per_call_a == pytest.approx(10.0)
    assert report.max_call_a == pytest.approx(0.2)
    assert report.results == tuple(results)


def test_summarize_without_decisions_does_not_divide_by_zero():
    report = summarize("a", "b", [])
    assert report.game_rate == 0.0 and report.ms_per_call_a == 0.0


def test_deck_id_distinguishes_decks_and_marks_agent_chosen():
    assert deck_id(None) == "agent-chosen"
    assert deck_id(default_deck()) != deck_id(tuple([3] * 60))


@pytest.mark.slow
def test_greedy_baseline_beats_random():
    report = run_arena("agents/greedy.py", "random", 60, procs=2)
    assert report.low > 0.5, format_report(report)
