"""Load agents the way Kaggle does, and wrap them with a fixed deck and a stopwatch.

An agent is `agent(obs) -> list[int]`. While `obs["select"]` is None it must return the
60-card deck; afterwards it returns `maxCount` indices into `obs["select"]["option"]`.

A spec is a builtin name ("random", "first"), a path to a .py file, or a path to a
submission.tar.gz (so the arena can play exactly the artifact that will be uploaded).
"""
import atexit
import os
import shutil
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from harness.decks import load_deck, validate_deck

AgentFn = Callable[[dict], list[int]]
BUILTINS = ("random", "first")
AGENT_ROOT_ENV = "PTCG_AGENT_ROOT"  # where a packaged main.py finds deck.csv outside Kaggle


class AgentLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedAgent:
    name: str
    fn: AgentFn
    deck: tuple[int, ...] | None


def load_agent(spec: str) -> LoadedAgent:
    if spec in BUILTINS:
        from kaggle_environments.envs.cabt.cabt import agents

        return LoadedAgent(spec, agents[spec], None)

    path = Path(spec)
    if not path.is_file():
        raise AgentLoadError(f"{spec!r} is neither a builtin ({', '.join(BUILTINS)}) nor a file")
    if spec.endswith((".tar.gz", ".tgz")):
        return _load_tarball(path)
    return _exec_agent_file(path, path.stem)


def _exec_agent_file(path: Path, name: str, env: dict[str, str] | None = None) -> LoadedAgent:
    """Mirror kaggle_environments.agent.get_last_callable: exec into a bare namespace (so no
    __file__), with the agent's folder on sys.path only while the module body runs."""
    namespace: dict = {}
    folder = path.resolve().parent
    modules_before = set(sys.modules)
    saved_env = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    sys.path.append(str(folder))
    try:
        exec(compile(path.read_text(), str(path), "exec"), namespace)
    except Exception as exc:
        raise AgentLoadError(f"{path}: {exc!r}") from exc
    finally:
        sys.path[:] = [entry for entry in sys.path if entry != str(folder)]  # main.py may add it too
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _forget_modules_under(folder, keep=modules_before)

    fn = namespace.get("agent")
    if not callable(fn):
        callables = [v for v in namespace.values() if callable(v)]
        if not callables:
            raise AgentLoadError(f"{path} defines no callable")
        fn = callables[-1]
    deck = validate_deck(namespace["DECK"]) if "DECK" in namespace else None
    return LoadedAgent(name, fn, deck)


def _forget_modules_under(folder: Path, keep: set[str]) -> None:
    """Helpers imported while loading one agent must not be served to the next one.

    Two agents that each ship a `shared.py` would otherwise share whichever loaded first.
    The loaded agent keeps its own references, so dropping the cache entries is safe.
    """
    for name in set(sys.modules) - keep:
        module_file = getattr(sys.modules[name], "__file__", None)
        if module_file and Path(module_file).resolve().is_relative_to(folder):
            del sys.modules[name]


def _load_tarball(tarball: Path) -> LoadedAgent:
    root = Path(tempfile.mkdtemp(prefix="agent-"))
    atexit.register(shutil.rmtree, root, True)
    try:
        with tarfile.open(tarball) as tar:
            tar.extractall(root, filter="data")
    except (tarfile.TarError, OSError) as exc:
        raise AgentLoadError(f"{tarball}: {exc}") from exc
    if not (root / "main.py").is_file() or not (root / "deck.csv").is_file():
        raise AgentLoadError(f"{tarball} needs main.py and deck.csv at its root")
    loaded = _exec_agent_file(root / "main.py", tarball.name, env={AGENT_ROOT_ENV: str(root)})
    return LoadedAgent(loaded.name, loaded.fn, load_deck(root / "deck.csv"))


def with_deck(fn: AgentFn, deck: tuple[int, ...]) -> AgentFn:
    """Answer the deck-selection call with `deck`, delegate everything else to `fn`."""

    def agent(obs):
        if obs["select"] is None:
            return list(deck)
        return fn(obs)

    return agent


def timed(fn: AgentFn) -> tuple[AgentFn, dict]:
    """Wrap `fn`; the returned dict accumulates `secs` and `calls` across the match."""
    clock = {"secs": 0.0, "calls": 0}

    def agent(obs):
        started = time.perf_counter()
        try:
            return fn(obs)
        finally:
            clock["secs"] += time.perf_counter() - started
            clock["calls"] += 1

    return agent, clock
