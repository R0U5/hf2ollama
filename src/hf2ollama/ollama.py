"""Ollama integration.

Detects `ollama` on PATH; if absent, warn and exit cleanly (do not fail the
run — the GGUF + Modelfile are already the primary deliverable). If present,
runs `ollama create <name> -f <modelfile>` with stdout streamed, from inside
--output-dir so a relative FROM path resolves correctly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from rich.console import Console

from .config import Config
from .errors import OllamaNotFoundError

console = Console()


def find_ollama() -> str | None:
    return shutil.which("ollama")


def _warn_missing(cfg: Config, modelfile: Path, reason: str = "") -> None:
    extra = f" ({reason})" if reason else ""
    console.print(
        f"[yellow]ollama not found on PATH{extra} — skipping 'ollama create'. "
        "Your .gguf and Modelfile are ready. Register later with:[/yellow]"
    )
    console.print(
        f"  cd {cfg.output_dir} && ollama create {cfg.ollama_name} -f {modelfile.name}"
    )


def _run_create(cfg: Config, modelfile: Path) -> None:
    """Run `ollama create <name> -f <modelfile>` non-interactively, streaming stdout."""
    binary = find_ollama()
    if binary is None:
        raise OllamaNotFoundError("ollama binary not found on PATH")

    console.rule(f"[bold]ollama create[/bold] {cfg.ollama_name}")
    try:
        proc = subprocess.Popen(
            [binary, "create", cfg.ollama_name, "-f", str(modelfile)],
            cwd=str(cfg.output_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.strip():
                console.print(line)
        proc.wait()
    except OSError as exc:
        raise OllamaNotFoundError(f"failed to run 'ollama create': {exc}") from exc

    if proc.returncode != 0:
        raise OllamaNotFoundError(
            f"'ollama create {cfg.ollama_name}' exited with code {proc.returncode}. "
            "See the streamed output above; the GGUF and Modelfile are already produced."
        )


def register_with_ollama(cfg: Config, modelfile: Path, state, display=None) -> None:
    """Entry point. Best-effort: missing ollama is a warning, not a failure."""

    if cfg.skip_ollama_create:
        if display is not None:
            display.skipped("ollama_create", "--skip-ollama-create")
        return

    # Resumable: never re-run a completed registration.
    if state.is_done("ollama_create"):
        if display is not None:
            display.skipped("ollama_create", "from state file")
        return

    if display is not None:
        display.start("ollama_create", f"ollama create {cfg.ollama_name}")
    try:
        _run_create(cfg, modelfile)
        state.mark_done("ollama_create", name=cfg.ollama_name)
        if display is not None:
            display.done("ollama_create", cfg.ollama_name)
    except OllamaNotFoundError as exc:
        # Missing ollama: warn and continue (do not fail).
        _warn_missing(cfg, modelfile, str(exc))
        state.mark_skipped("ollama_create", name=cfg.ollama_name, reason="ollama not found on PATH")
        if display is not None:
            display.skipped("ollama_create", "ollama not on PATH")
