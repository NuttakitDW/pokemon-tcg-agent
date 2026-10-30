import json
from pathlib import Path

import pytest

from harness.cli import main
from harness.decks import default_deck, save_deck


def test_arena_command_reports_and_exits_zero(capsys):
    assert main(["arena", "--a", "random", "--b", "first", "-n", "2", "--procs", "0"]) == 0
    assert "win rate" in capsys.readouterr().out


def test_arena_command_exits_nonzero_on_faults(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("999999\n" * 60)
    code = main(["arena", "--a", "random", "--b", "random", "-n", "2", "--procs", "1", "--deck-a", str(bad)])
    assert code == 1 and "FAULTS" in capsys.readouterr().out


def test_playground_command_prints_a_table_and_saves_the_run(tmp_path, capsys):
    out = tmp_path / "run"
    args = ["playground", "random", "--vs", "first", "-n", "2", "--procs", "0", "--replays", "1", "--out", str(out)]
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "first" in text and "win rate" in text and str(out) in text
    assert (out / "report.json").exists() and len(list((out / "replays").glob("*.html"))) == 1


def test_playground_defaults_to_every_other_agent_in_the_pool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the pool is read from ./agents, so give it a known one
    (tmp_path / "agents").mkdir()
    bot = (
        "from kaggle_environments.envs.cabt.cabt import deck as DECK\n"
        "def agent(obs):\n"
        "    return list(DECK) if obs['select'] is None else list(range(obs['select']['maxCount']))\n"
    )
    for name in ("alpha", "beta"):
        (tmp_path / "agents" / f"{name}.py").write_text(bot)

    assert main(["playground", "agents/alpha.py", "-n", "2", "--procs", "0", "--replays", "0", "--out", "run"]) == 0
    data = json.loads((tmp_path / "run" / "report.json").read_text())
    assert [o["spec_b"] for o in data["opponents"]] == ["random", "first", "agents/beta.py"]
    assert not (tmp_path / "run" / "replays").exists()


def test_playground_exits_nonzero_on_faults_but_still_saves_the_report(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("999999\n" * 60)
    args = ["playground", "random", "--vs", "first", "-n", "2", "--procs", "1", "--deck", str(bad), "--out", str(tmp_path / "run")]
    assert main(args) == 1 and "FAULTS" in capsys.readouterr().out
    saved = json.loads((tmp_path / "run" / "report.json").read_text())
    assert saved["opponents"][0]["faults"] == ["A:INVALID"]


def test_playground_exits_two_when_the_report_cannot_be_saved(tmp_path, monkeypatch, capsys):
    def full_disk(self, *args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_text", full_disk)
    args = ["playground", "random", "--vs", "first", "-n", "2", "--procs", "0", "--replays", "0", "--out", str(tmp_path / "run")]
    assert main(args) == 2
    assert "not saved" in capsys.readouterr().out  # the table still printed


@pytest.mark.parametrize(
    "flags", [["-n", "0"], ["-n", "-3"], ["--replays", "-1"], ["--bo", "0"], ["-n", "many"]]
)
def test_playground_rejects_nonsense_counts(flags, tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["playground", "random", "--out", str(tmp_path / "run"), *flags])  # --out: a regression must not litter ./runs
    assert not (tmp_path / "run").exists()
    assert exit_info.value.code == 2 and "error" in capsys.readouterr().err


def test_playground_missing_deck_is_a_clean_error(tmp_path, capsys):
    args = ["playground", "random", "--deck", str(tmp_path / "nope.csv"), "-n", "1", "--procs", "0", "--out", str(tmp_path / "run")]
    assert main(args) == 2
    assert "error:" in capsys.readouterr().err
    assert main(["arena", "--a", "random", "--deck-a", str(tmp_path / "nope.csv"), "-n", "1", "--procs", "0"]) == 2


def test_playground_missing_agent_is_a_clean_error(tmp_path, capsys):
    assert main(["playground", "no/such/agent.py", "-n", "1", "--procs", "0", "--out", str(tmp_path / "run")]) == 2
    assert "error:" in capsys.readouterr().err


def test_package_then_validate_roundtrip(tmp_path, capsys):
    deck = tmp_path / "deck.csv"
    save_deck(default_deck(), deck)
    out = tmp_path / "s.tar.gz"
    assert main(["package", "agents/greedy.py", "--deck", str(deck), "--out", str(out), "--games", "2"]) == 0
    assert main(["validate", str(out), "--games", "2"]) == 0
    assert "[FAIL]" not in capsys.readouterr().out


def test_package_without_validation_skips_the_engine(tmp_path, capsys):
    deck = tmp_path / "deck.csv"
    save_deck(default_deck(), deck)
    out = tmp_path / "s.tar.gz"
    assert main(["package", "agents/greedy.py", "--deck", str(deck), "--out", str(out), "--no-validate"]) == 0
    assert out.exists() and "[ok]" not in capsys.readouterr().out


def test_missing_agent_is_a_clean_error(capsys):
    assert main(["arena", "--a", "no/such/agent.py", "-n", "1", "--procs", "0"]) == 2
    assert "error:" in capsys.readouterr().err


def test_validate_with_too_few_games_is_a_clean_error(tmp_path, capsys):
    deck = tmp_path / "deck.csv"
    save_deck(default_deck(), deck)
    out = tmp_path / "s.tar.gz"
    assert main(["package", "agents/greedy.py", "--deck", str(deck), "--out", str(out), "--no-validate"]) == 0
    assert main(["validate", str(out), "--games", "1"]) == 2
