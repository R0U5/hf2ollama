"""Resumable state tracking.

Each pipeline step writes/updates a JSON state file describing whether it is
'done', 'skipped' or 'failed', an ISO8601 timestamp, and any produced path(s).

On re-run, any step whose status is 'done' AND whose recorded paths still
exist on disk is skipped. If a recorded path is missing the step is marked
'failed' so the caller re-runs that step and everything downstream.

Schema (version 1):

{
  "version": 1,
  "input": { "kind": "hf|local", "ref": "...", "revision": "..." },
  "steps": {
    "resolve":     { "status": "...", "at": "ISO8601", ... },
    "download":    { "status": "...", "at": "...", "path": "..." },
    "probe":       { "status": "...", "at": "...", "architecture": "..." },
    "convert_fp16":{ "status": "...", "at": "...", "path": "..." },
    "quantize":    { "status": "...", "at": "...", "path": "...", "type": "..." },
    "modelfile":   { "status": "...", "at": "...", "path": "..." },
    "ollama_create": { "status": "...", "at": "...", "name": "..." }
  }
}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import StateError

STEP_KEYS = (
    "resolve",
    "download",
    "probe",
    "convert_fp16",
    "quantize",
    "modelfile",
    "ollama_create",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _existing(path: Any) -> bool:
    if path is None:
        return False
    return Path(path).exists()


@dataclass
class StateStore:
    """In-memory + on-disk representation of pipeline state."""

    path: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "StateStore":
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                raise StateError(f"could not read state file {path}: {exc}") from exc
            if not isinstance(data, dict) or data.get("version") != 1:
                # Incompatible/accepted: start fresh rather than guess.
                raise StateError(
                    f"state file {path} exists but is not a version-1 hf2ollama state file. "
                    "Remove it (or point --state-file elsewhere) to start fresh."
                )
        else:
            data = {"version": 1, "input": None, "steps": {}}
        return cls(path=path, data=data)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=2)
            os.replace(tmp, self.path)
        except OSError as exc:
            raise StateError(f"could not write state file {self.path}: {exc}") from exc

    # ---- step helpers -----------------------------------------------------

    def step(self, key: str) -> dict[str, Any]:
        return self.data.setdefault("steps", {}).setdefault(key, {})

    def mark_done(self, key: str, **extra: Any) -> None:
        s = self.step(key)
        s.update({"status": "done", "at": now_iso()})
        s.update(extra)
        self.save()

    def mark_skipped(self, key: str, **extra: Any) -> None:
        s = self.step(key)
        s.update({"status": "skipped", "at": now_iso()})
        s.update(extra)
        self.save()

    def mark_failed(self, key: str, reason: str) -> None:
        s = self.step(key)
        s.update({"status": "failed", "at": now_iso(), "reason": reason})
        self.save()

    def set_input(self, kind: str, ref: str, revision: str) -> None:
        self.data["input"] = {"kind": kind, "ref": ref, "revision": revision}
        self.save()

    def is_done(self, key: str) -> bool:
        return self.step(key).get("status") == "done"

    def is_skipped(self, key: str) -> bool:
        return self.step(key).get("status") == "skipped"

    def is_failed(self, key: str) -> bool:
        return self.step(key).get("status") == "failed"

    def should_skip(self, key: str, paths: list[Any]) -> bool:
        """Skip a step if it is done and every recorded artifact still exists."""
        if not self.is_done(key):
            return False
        if any(not _existing(p) for p in paths):
            self.mark_failed(key, reason="recorded artifact missing on disk; will re-run")
            # Strip the 'done' label but keep provenance fields.
            self.step(key)["status"] = "failed"
            self.save()
            return False
        return True
