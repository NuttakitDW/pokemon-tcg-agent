import pytest

from harness.agents import AgentLoadError, load_agent, timed, with_deck

DECK = tuple([3] * 60)


def write(tmp_path, body):
    path = tmp_path / "bot.py"
    path.write_text(body)
    return str(path)


def test_builtin_agents_load():
    assert load_agent("random").fn is not load_agent("first").fn


def test_unknown_spec_is_an_error(tmp_path):
    with pytest.raises(AgentLoadError):
        load_agent(str(tmp_path / "missing.py"))


def test_path_agent_prefers_agent_over_last_callable(tmp_path):
    loaded = load_agent(write(tmp_path, "def agent(obs):\n    return [0]\n\ndef helper():\n    pass\n"))
    assert loaded.fn({}) == [0]
    assert loaded.name == "bot"


def test_path_agent_falls_back_to_last_callable(tmp_path):
    loaded = load_agent(write(tmp_path, "def first(obs):\n    return [1]\n\ndef second(obs):\n    return [2]\n"))
    assert loaded.fn({}) == [2]


def test_module_body_has_no_dunder_file(tmp_path):
    # Kaggle exec()s agents without __file__; the harness must fail the same way.
    with pytest.raises(AgentLoadError):
        load_agent(write(tmp_path, "import os\nHERE = os.path.dirname(__file__)\ndef agent(obs):\n    return []\n"))


def test_deck_constant_is_validated_and_exposed(tmp_path):
    loaded = load_agent(write(tmp_path, f"DECK = {[3] * 60}\ndef agent(obs):\n    return []\n"))
    assert loaded.deck == DECK
    with pytest.raises(Exception):
        load_agent(write(tmp_path, "DECK = [1, 2]\ndef agent(obs):\n    return []\n"))


def test_with_deck_only_intercepts_deck_selection():
    fn = with_deck(lambda obs: [7], DECK)
    assert fn({"select": None}) == list(DECK)
    assert fn({"select": {"option": []}}) == [7]


def test_timed_accumulates_even_when_the_agent_raises():
    def boom(obs):
        raise RuntimeError("x")

    fn, clock = timed(boom)
    with pytest.raises(RuntimeError):
        fn({})
    assert clock["calls"] == 1 and clock["secs"] >= 0.0


def test_a_tarball_is_extracted_once_per_process(tmp_path, monkeypatch):
    import tempfile
    from pathlib import Path

    from harness import agents
    from harness.package import build_submission

    tarball = build_submission(Path("agents/greedy.py"), Path("decks/default.csv"), tmp_path / "s.tar.gz")
    real_mkdtemp, extractions = tempfile.mkdtemp, []
    monkeypatch.setattr(agents.tempfile, "mkdtemp", lambda **kw: extractions.append(kw) or real_mkdtemp(**kw))

    first, second = load_agent(str(tarball)), load_agent(str(tarball))
    assert len(extractions) == 1  # a worker plays many matches; each must not leave its own copy
    assert first.deck == second.deck and first.fn is not second.fn  # but the agent code is fresh per load


def test_timed_keeps_the_slowest_single_call():
    import time

    delays = iter([0.03, 0.0, 0.0])
    fn, clock = timed(lambda obs: time.sleep(next(delays)) or [0])
    for _ in range(3):
        fn({})
    assert clock["calls"] == 3
    assert 0.03 <= clock["max"] <= clock["secs"]


def test_sibling_modules_do_not_leak_between_agents(tmp_path):
    # Two agents that each ship a `shared.py` must each get their own.
    for name, value in (("v1", 1), ("v2", 2)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "shared.py").write_text(f"VALUE = {value}\n")
        (folder / "bot.py").write_text("from shared import VALUE\ndef agent(obs):\n    return [VALUE]\n")
    first = load_agent(str(tmp_path / "v1" / "bot.py"))
    second = load_agent(str(tmp_path / "v2" / "bot.py"))
    assert first.fn({}) == [1] and second.fn({}) == [2]


def test_load_leaves_sys_path_untouched(tmp_path):
    import sys

    before = list(sys.path)
    load_agent(write(tmp_path, "def agent(obs):\n    return [0]\n"))
    assert sys.path == before


def test_tarball_spec_loads_main_py_and_deck(tmp_path):
    from harness.decks import default_deck, save_deck
    from harness.package import build_submission

    deck = tmp_path / "deck.csv"
    save_deck(default_deck(), deck)
    src = tmp_path / "bot.py"
    src.write_text("def agent(obs):\n    return [7]\n")
    loaded = load_agent(str(build_submission(src, deck, tmp_path / "s.tar.gz")))
    assert loaded.deck == default_deck()
    assert loaded.fn({"select": {"option": []}}) == [7]
    assert loaded.fn({"select": None}) == list(default_deck())


def test_two_tarballs_do_not_share_agent_impl(tmp_path):
    from harness.decks import default_deck, save_deck
    from harness.package import build_submission

    deck = tmp_path / "deck.csv"
    save_deck(default_deck(), deck)
    loaded = []
    for value in (1, 2):
        src = tmp_path / f"bot{value}.py"
        src.write_text(f"def agent(obs):\n    return [{value}]\n")
        loaded.append(load_agent(str(build_submission(src, deck, tmp_path / f"s{value}.tar.gz"))))
    assert [l.fn({"select": {"option": []}}) for l in loaded] == [[1], [2]]


def test_tarball_missing_files_is_an_error(tmp_path):
    import io
    import tarfile

    path = tmp_path / "s.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("main.py")
        tar.addfile(info, io.BytesIO(b""))
    with pytest.raises(AgentLoadError):
        load_agent(str(path))
