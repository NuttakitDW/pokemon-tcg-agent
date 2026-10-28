import io
import tarfile
from pathlib import Path

import pytest

from harness.decks import default_deck, save_deck
from harness import package
from harness.package import PackageError, build_submission, validate_submission


@pytest.fixture
def deck_csv(tmp_path):
    path = tmp_path / "deck.csv"
    save_deck(default_deck(), path)
    return path


@pytest.fixture
def agent_py(tmp_path):
    path = tmp_path / "src" / "bot.py"
    path.parent.mkdir()
    path.write_text("def agent(obs):\n    return list(range(obs['select']['maxCount']))\n")
    return path


def test_layout_has_main_and_deck_at_the_root(tmp_path, agent_py, deck_csv):
    out = build_submission(agent_py, deck_csv, tmp_path / "out" / "s.tar.gz")
    with tarfile.open(out) as tar:
        assert set(tar.getnames()) == {"main.py", "deck.csv", "agent_impl.py"}


def test_build_is_byte_for_byte_deterministic(tmp_path, agent_py, deck_csv):
    a = build_submission(agent_py, deck_csv, tmp_path / "a.tar.gz").read_bytes()
    b = build_submission(agent_py, deck_csv, tmp_path / "b.tar.gz").read_bytes()
    assert a == b


def test_include_keeps_paths_relative_to_the_agent(tmp_path, agent_py, deck_csv):
    weights = agent_py.parent / "weights" / "w.npz"
    weights.parent.mkdir()
    weights.write_bytes(b"\x00")
    out = build_submission(agent_py, deck_csv, tmp_path / "s.tar.gz", include=(weights,))
    with tarfile.open(out) as tar:
        assert "weights/w.npz" in tar.getnames()


def test_short_deck_is_rejected(tmp_path, agent_py):
    short = tmp_path / "short.csv"
    short.write_text("3\n" * 10)
    with pytest.raises(ValueError):
        build_submission(agent_py, short, tmp_path / "s.tar.gz")


def test_missing_agent_file_is_rejected(tmp_path, deck_csv):
    with pytest.raises(PackageError):
        build_submission(tmp_path / "nope.py", deck_csv, tmp_path / "s.tar.gz")


def test_valid_submission_passes_every_check(tmp_path, agent_py, deck_csv):
    out = build_submission(agent_py, deck_csv, tmp_path / "s.tar.gz")
    report = validate_submission(out, games=2)
    assert report.ok, report.checks


def _tar_with(path, files):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_validation_catches_the_dunder_file_trap(tmp_path, deck_csv):
    # The bug that produced an ERROR submission in the Simulation division.
    main = b"import os\nHERE = os.path.dirname(__file__)\ndef agent(obs):\n    return []\n"
    out = _tar_with(tmp_path / "s.tar.gz", {"main.py": main, "deck.csv": deck_csv.read_bytes()})
    report = validate_submission(out, games=2)
    assert not report.ok
    assert any("__file__" in c.detail for c in report.checks)


def test_validation_catches_nested_layout(tmp_path, deck_csv):
    out = _tar_with(tmp_path / "s.tar.gz", {"sub/main.py": b"x=1", "sub/deck.csv": deck_csv.read_bytes()})
    report = validate_submission(out, games=2)
    assert not report.ok and report.checks[1].name == "layout"


def test_validation_catches_a_non_agent_last_callable(tmp_path, deck_csv):
    main = b"def agent(obs):\n    return []\n\ndef helper():\n    pass\n"
    out = _tar_with(tmp_path / "s.tar.gz", {"main.py": main, "deck.csv": deck_csv.read_bytes()})
    report = validate_submission(out, games=2)
    assert not report.ok


def test_dot_slash_member_names_are_accepted(tmp_path, agent_py, deck_csv):
    # `tar czf x.tar.gz -C dir .` prefixes every member with ./
    built = build_submission(agent_py, deck_csv, tmp_path / "built.tar.gz")
    with tarfile.open(built) as src:
        files = {f"./{m.name}": src.extractfile(m).read() for m in src.getmembers()}
    out = _tar_with(tmp_path / "dot.tar.gz", files)
    assert validate_submission(out, games=2).ok


def test_corrupt_archive_is_a_failed_check_not_a_traceback(tmp_path):
    bad = tmp_path / "bad.tar.gz"
    bad.write_bytes(b"not a tarball")
    report = validate_submission(bad, games=2)
    assert not report.ok and report.checks[-1].name == "layout"


def test_oversize_archive_is_reported_without_unpacking(tmp_path, agent_py, deck_csv, monkeypatch):
    out = build_submission(agent_py, deck_csv, tmp_path / "s.tar.gz")
    monkeypatch.setattr(package, "MAX_BYTES", 10)
    report = validate_submission(out, games=2)
    assert [c.name for c in report.checks] == ["size"] and not report.ok


def test_validating_zero_games_is_refused(tmp_path, agent_py, deck_csv):
    out = build_submission(agent_py, deck_csv, tmp_path / "s.tar.gz")
    for games in (0, 1):
        with pytest.raises(PackageError):
            validate_submission(out, games=games)


def test_cwd_relative_file_access_fails_validation(tmp_path, deck_csv):
    # In validation the cwd is an empty directory, so cwd-relative loads surface here.
    src = tmp_path / "src"
    src.mkdir()
    (src / "bot.py").write_text(
        "def agent(obs):\n    open('weights.txt').read()\n    return [0]\n"
    )
    (src / "weights.txt").write_text("w")
    out = build_submission(src / "bot.py", deck_csv, tmp_path / "s.tar.gz", include=(src / "weights.txt",))
    report = validate_submission(out, games=2)
    assert not report.ok
    assert any("weights.txt" in c.detail for c in report.checks)
