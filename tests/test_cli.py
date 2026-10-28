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
