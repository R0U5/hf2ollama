"""Shared UI primitives — single Rich Console, step renderer, progress adapter.

Everything user-facing routes through here so we get a consistent look and
can swap implementations in tests. Two ideas in this file:

1. **Step renderer.** Every pipeline step calls `step_panel("download", "...")`,
   `step_done(...)`, `step_failed(...)`. We render a Live display with a
   checklist-style status column so the user always knows:
     - which step is running
     - which steps are done / skipped / pending
     - what's happening inside the active step (progress bar, spinner)

2. **HF download adapter.** `snapshot_download` already shows a tqdm bar;
   we wrap it in a `rich.progress.Progress` so the user sees one coherent
   bar that updates in place instead of tqdm overwriting lines on stderr.
"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


@dataclass
class StepRecord:
    name: str
    status: str = "pending"  # pending | active | done | skipped | failed
    detail: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0


STEP_ORDER = (
    "resolve",
    "download",
    "probe",
    "convert_fp16",
    "quantize",
    "modelfile",
    "ollama_create",
)

STEP_LABELS = {
    "resolve": "Resolve input",
    "download": "Download weights",
    "probe": "Probe architecture",
    "convert_fp16": "Convert to fp16 GGUF",
    "quantize": "Quantize GGUF",
    "modelfile": "Generate Modelfile",
    "ollama_create": "Register with Ollama",
}


class StepDisplay:
    """Manages a Live display of pipeline steps with embedded progress.

    Usage:
        sd = StepDisplay(console)
        sd.start("download", "fetching 6 files")
        with sd.download_progress() as progress:
            task = progress.add_task("downloading", total=N)
            ... do the download, advance the task ...
        sd.done("download", "6 files -> .hf-cache/...")
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self._steps: dict[str, StepRecord] = {n: StepRecord(name=n) for n in STEP_ORDER}
        self._live: Live | None = None
        self._lock = threading.Lock()
        # A dedicated progress sub-display that the download / convert steps
        # can write into without interfering with the checklist rendering.
        self._progress: Progress | None = None

    # ---- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "StepDisplay":
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    # ---- step transitions -------------------------------------------------

    def start(self, name: str, detail: str = "") -> None:
        with self._lock:
            s = self._steps[name]
            s.status = "active"
            s.detail = detail
        self.refresh()

    def done(self, name: str, detail: str = "") -> None:
        with self._lock:
            s = self._steps[name]
            s.status = "done"
            s.detail = detail
        self.refresh()

    def skipped(self, name: str, detail: str = "") -> None:
        with self._lock:
            s = self._steps[name]
            s.status = "skipped"
            s.detail = detail
        self.refresh()

    def failed(self, name: str, detail: str = "") -> None:
        with self._lock:
            s = self._steps[name]
            s.status = "failed"
            s.detail = detail
        self.refresh()

    # ---- sub-progress for the active step --------------------------------

    @contextmanager
    def sub_progress(
        self,
        text: str = "working",
        total: float | None = None,
        transient: bool = True,
    ) -> Iterator["Progress"]:
        """Yield a Rich Progress that the caller drives. Renders below the checklist."""
        progress = Progress(
            TextColumn(f"[bold cyan]{text}[/bold cyan]"),
            BarColumn(bar_width=30),
            DownloadColumn() if total is None else TextColumn("{task.completed}/{task.total}"),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=transient,
        )
        if total is not None:
            progress.total = total

        # Mount the sub-progress into the Live render so the user sees
        # checklist + bar together.
        prev_render = self._live.renderable if self._live else self._render()
        if self._live is not None:
            self._live.update(Group(self._render(), progress))
        try:
            yield progress
        finally:
            if self._live is not None:
                self._live.update(prev_render)

    # ---- final summary panel ---------------------------------------------

    def summary(self, title: str, rows: list[tuple[str, str]]) -> None:
        """Print a summary table once the Live display exits."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", justify="right")
        table.add_column(style="white")
        for k, v in rows:
            table.add_row(k, v)
        self.console.print(Panel(table, title=f"[bold green]✓[/bold green] {title}", border_style="green"))

    # ---- internal rendering ----------------------------------------------

    def _render(self):
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=3, justify="center")
        grid.add_column(style="bold", min_width=22)
        grid.add_column(style="dim")
        for name in STEP_ORDER:
            s = self._steps[name]
            grid.add_row(*_step_row(s))
        return Panel(grid, title="[bold]hf2ollama[/bold]", border_style="cyan")


def _step_row(s: StepRecord) -> tuple[str, str, str]:
    icon, style = _icon(s.status)
    label = STEP_LABELS.get(s.name, s.name)
    detail = s.detail or ""
    return (f"[{style}]{icon}[/{style}]", f"[{style}]{label}[/{style}]", detail)


def _icon(status: str) -> tuple[str, str]:
    return {
        "pending": ("·", "dim"),
        "active":  ("◐", "cyan"),
        "done":    ("✓", "green"),
        "skipped": ("↷", "yellow"),
        "failed":  ("✗", "red"),
    }.get(status, ("?", "white"))


# ---- Public helpers for non-step output ------------------------------------

def error_panel(console: Console, title: str, body: str, hint: str | None = None) -> None:
    text = Text(body)
    if hint:
        text.append("\n\n")
        text.append("Hint: ", style="bold")
        text.append(hint)
    console.print(Panel(text, title=f"[bold red]✗ {title}[/bold red]", border_style="red"))


def dry_run_panel(console: Console, cfg) -> None:
    """Render the dry-run plan as a single readable table."""
    table = Table(title="hf2ollama plan (dry run)", show_header=False, padding=(0, 2))
    table.add_column(style="cyan", justify="right")
    table.add_column(style="white")
    table.add_row("input", cfg.input_ref)
    table.add_row("quantize", cfg.quant)
    table.add_row("output_dir", str(cfg.output_dir))
    table.add_row("modelfile_name", cfg.modelfile_name)
    table.add_row(
        "ollama_name",
        cfg.ollama_name if not cfg.skip_ollama_create else "(skipped)",
    )
    table.add_row("keep_fp16", str(cfg.keep_fp16))
    table.add_row("state_file", str(cfg.state_file))
    chain = " → ".join(s for s in STEP_ORDER if s != "ollama_create" or not cfg.skip_ollama_create)
    table.add_row("actions", chain)
    console.print(table)
