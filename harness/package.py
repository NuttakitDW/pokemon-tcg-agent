"""Build and validate `submission.tar.gz`.

Contract (from the Simulation division, assumed unchanged for the Playground): a tarball
with `main.py` and `deck.csv` at its root, extracted to /kaggle_simulations/agent/, at most
197.7 MiB. Kaggle runs main.py through exec(), so `__file__` does not exist inside it and the
LAST callable it defines is the agent. The generated main.py respects both, and `validate`
re-checks the finished tarball in a clean directory with Kaggle's own loader.
"""
import gzip
import io
import json
import os
import posixpath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from harness.decks import DeckError, load_deck

MAX_BYTES = int(197.7 * 1024 * 1024)
AGENT_MODULE = "agent_impl"
RESERVED = frozenset({"main.py", "deck.csv", f"{AGENT_MODULE}.py"})
VALIDATE_SCRIPT = Path(__file__).with_name("validate_run.py")
VALIDATE_TIMEOUT_SECS = 600

MAIN_TEMPLATE = f'''\
import os
import sys

_ROOT = "/kaggle_simulations/agent"
if not os.path.exists(_ROOT):
    _ROOT = os.environ.get("PTCG_AGENT_ROOT", os.getcwd())  # set by local validation
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from {AGENT_MODULE} import agent as _policy

with open(os.path.join(_ROOT, "deck.csv")) as _f:
    _DECK = [int(_line) for _line in _f if _line.strip()]


def agent(obs):
    if obs["select"] is None:
        return list(_DECK)
    return _policy(obs)
'''


class PackageError(ValueError):
    pass


def _add(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def build_submission(
    agent_py: Path, deck_csv: Path, out: Path, include: tuple[Path, ...] = ()
) -> Path:
    """Write a deterministic tarball. `include` files keep their paths relative to agent_py."""
    load_deck(deck_csv)  # raises DeckError on anything but 60 integer IDs
    if not agent_py.is_file():
        raise PackageError(f"{agent_py} not found")

    files: dict[str, bytes] = {
        "main.py": MAIN_TEMPLATE.encode(),
        "deck.csv": deck_csv.read_bytes(),
        f"{AGENT_MODULE}.py": agent_py.read_bytes(),
    }
    for extra in include:
        relative = extra.resolve().relative_to(agent_py.resolve().parent).as_posix()
        if relative in RESERVED:
            raise PackageError(f"{relative} is reserved by the packager")
        files[relative] = extra.read_bytes()

    # Build in memory: gzip would otherwise stamp the output filename and time into the header.
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tar:
            for name in sorted(files):
                _add(tar, name, files[name])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(buffer.getvalue())
    return out


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


def _structure_checks(tarball: Path, root: Path) -> list[Check]:
    size = tarball.stat().st_size
    size_check = Check("size", size <= MAX_BYTES, f"{size / 1024 / 1024:.2f} MiB of 197.7")
    if not size_check.ok:
        return [size_check]  # do not unpack an oversize archive just to report on it
    try:
        with tarfile.open(tarball) as tar:
            members = tar.getmembers()
            names = [posixpath.normpath(m.name) for m in members]
            escaping = [n for n in names if n.startswith(("/", "..")) or "/../" in n]
            unpacked = sum(m.size for m in members)
            if escaping:
                return [size_check, Check("layout", False, f"unsafe member paths: {escaping}")]
            if unpacked > MAX_BYTES:
                return [size_check, Check("size", False, f"unpacks to {unpacked / 1024 / 1024:.0f} MiB")]
            tar.extractall(root, filter="data")
    except (tarfile.TarError, OSError) as exc:
        return [size_check, Check("layout", False, f"cannot read archive: {exc}")]
    top_level = {n for n in names if "/" not in n and n != "."}  # tolerate `tar czf x -C dir .`
    checks = [
        size_check,
        Check("layout", {"main.py", "deck.csv"} <= top_level, "main.py and deck.csv at the archive root"),
    ]
    try:
        load_deck(root / "deck.csv")
        checks.append(Check("deck", True, "60 integer card IDs"))
    except (DeckError, OSError) as exc:
        checks.append(Check("deck", False, str(exc)))
    return checks


def _play_checks(root: Path, games: int) -> list[Check]:
    # Clean interpreter (-I: no PYTHONPATH leakage) started in an EMPTY directory, so an agent
    # that opens files relative to the cwd fails here instead of silently passing.
    with tempfile.TemporaryDirectory(prefix="validate-cwd-") as empty_cwd:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(VALIDATE_SCRIPT), str(root / "main.py"), str(games)],
                cwd=empty_cwd, capture_output=True, text=True, timeout=VALIDATE_TIMEOUT_SECS,
                env={**os.environ, "PTCG_AGENT_ROOT": str(root)},
            )
        except subprocess.TimeoutExpired:
            return [Check("load + play", False, f"no result after {VALIDATE_TIMEOUT_SECS}s")]
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
        return [Check("load + play", False, " | ".join(tail) or f"exit code {proc.returncode}")]
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    return [Check(c["name"], c["ok"], c["detail"]) for c in report["checks"]]


def validate_submission(tarball: Path, games: int = 4) -> ValidationReport:
    if games < 2:
        raise PackageError("validate at least 2 matches so both seats are exercised")
    root = Path(tempfile.mkdtemp(prefix="submission-"))
    try:
        checks = _structure_checks(tarball, root)
        if all(c.ok for c in checks):
            checks += _play_checks(root, games)
        return ValidationReport(tuple(checks))
    finally:
        shutil.rmtree(root, ignore_errors=True)
