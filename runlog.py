"""Provenance ledger — every experiment run records what ran, when, and what came out.

    with RunRecord("P1-E1", "probe_representation.py", cfg) as rec:
        ...
        rec.result(stage="base", probe_acc=0.97)

Writes two files, both append-only:
  results/runs.jsonl  — one JSON object per run (machine-readable, diffable)
  results/RUNLOG.md   — the same run as a readable entry (what a human opens first)

Captures, without being asked: UTC start/end and duration, git commit + dirty flag +
branch, the full frozen config, library and GPU versions, the command line, and the
outcome (ok / failed + traceback summary). A FAILED run is recorded too — a run that
crashed is evidence, and silently leaving it out of the ledger is how "which version
produced this number?" becomes unanswerable.

Why this exists: four separate results in this project were retracted because a number
looked plausible and nobody could later say which code produced it. See RESULTS.md O-42.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict

__all__ = ["RunRecord", "git_state", "env_state"]

_JSONL = "results/runs.jsonl"
_MD = "results/RUNLOG.md"


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def git_state() -> Dict[str, Any]:
    """Commit the run was produced by. `dirty` means uncommitted edits were present —
    the commit alone does NOT identify the code in that case."""
    sha = _sh(["git", "rev-parse", "HEAD"])
    return {
        "commit": sha or "unknown",
        "short": sha[:7] if sha else "unknown",
        "branch": _sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "unknown",
        "dirty": bool(_sh(["git", "status", "--porcelain"])),
    }


def env_state() -> Dict[str, Any]:
    """Library and hardware versions. torch<2.5 silently disables the transformers torch
    backend, and tokenizer behaviour has shifted between transformers releases, so both
    are part of what identifies a result."""
    out: Dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "transformers", "numpy", "sklearn"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - a missing optional dep is itself worth recording
            out[mod] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        out["cuda"] = out["gpu"] = None
    return out


def _cfg_dict(cfg: Any) -> Any:
    if cfg is None:
        return None
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)
    return repr(cfg)


class RunRecord:
    """Context manager that writes one ledger entry per run."""

    def __init__(self, experiment: str, script: str, cfg: Any = None,
                 question: str = "", notes: str = "") -> None:
        self.experiment = experiment
        self.script = script
        self.question = question
        self.notes = notes
        self.cfg = _cfg_dict(cfg)
        self.results: list[Dict[str, Any]] = []
        self._t0 = 0.0
        self.started = ""

    def result(self, **kv: Any) -> None:
        """Record one result row (typically one per stage). Values must be JSON-safe;
        numpy scalars and arrays are converted."""
        self.results.append({k: _jsonable(v) for k, v in kv.items()})

    def __enter__(self) -> "RunRecord":
        self._t0 = time.time()
        self.started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        entry = {
            "experiment": self.experiment,
            "script": self.script,
            "question": self.question,
            "started_utc": self.started,
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "duration_s": round(time.time() - self._t0, 1),
            "status": "failed" if exc_type else "ok",
            "error": f"{exc_type.__name__}: {exc}" if exc_type else None,
            "argv": " ".join(sys.argv),
            "git": git_state(),
            "env": env_state(),
            "config": self.cfg,
            "results": self.results,
            "notes": self.notes,
        }
        os.makedirs("results", exist_ok=True)
        with open(_JSONL, "a") as f:
            f.write(json.dumps(entry) + "\n")
        _append_md(entry)
        return False  # never swallow the exception


def _jsonable(v: Any) -> Any:
    # NaN/Inf serialise as bare NaN/Infinity, which is not valid strict JSON and breaks
    # downstream readers. A missing measurement is None. (base has no valid direction, so
    # its ablated rate is genuinely absent rather than zero.)
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.round(6).tolist() if v.size <= 64 else f"<array shape={list(v.shape)}>"
    except ImportError:
        pass
    return v


def _append_md(e: Dict[str, Any]) -> None:
    new = not os.path.exists(_MD)
    with open(_MD, "a") as f:
        if new:
            f.write("# Run log\n\nAppend-only. One entry per execution of an experiment "
                    "script, newest at the bottom. Written automatically by `runlog.py` — "
                    "do not hand-edit.\n\nExperiment ids: `E02` is the pilot (this repo's "
                    "original experiment); `P1-E1`…`P1-E7` are the experiments of paper P1, "
                    "specified in `Writing/P1-Coupling-Not-Capability.md` in the vault.\n")
        g, v = e["git"], e["env"]
        mark = "OK" if e["status"] == "ok" else "**FAILED**"
        f.write(f"\n---\n\n## {e['experiment']} · {e['started_utc']} · {mark}\n\n")
        if e["question"]:
            f.write(f"> {e['question']}\n\n")
        f.write(f"- **script** `{e['script']}` — `{e['argv']}`\n")
        f.write(f"- **code** `{g['short']}` on `{g['branch']}`"
                f"{' ⚠️ DIRTY WORKING TREE — commit does not identify this code' if g['dirty'] else ''}\n")
        f.write(f"- **duration** {e['duration_s']}s\n")
        f.write(f"- **env** torch {v.get('torch')} · transformers {v.get('transformers')} · "
                f"sklearn {v.get('sklearn')} · gpu {v.get('gpu')}\n")
        if e["error"]:
            f.write(f"- **error** `{e['error']}`\n")
        if e["results"]:
            f.write("\n| " + " | ".join(e["results"][0].keys()) + " |\n")
            f.write("|" + "---|" * len(e["results"][0]) + "\n")
            for r in e["results"]:
                f.write("| " + " | ".join(str(x) for x in r.values()) + " |\n")
        if e["notes"]:
            f.write(f"\n{e['notes']}\n")
