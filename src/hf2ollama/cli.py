"""CLI entrypoint for hf2ollama.

Responsibility:
  - parse flags into a Config
  - derive defaults (names, paths)
  - run the resumable pipeline, streaming progress to the terminal
  - translate every Hf2OllamaError into a specific, colored error panel

Reading order note: this file wires everything together. Lower-level behavior
lives in resolver.py / probe.py / pipeline.py / modelfile.py / ollama.py /
state.py / progress.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, Prompt

from . import __version__
from .config import Config, KNOWN_QUANTS, safe_name, _normalize_quant
from .errors import Hf2OllamaError, ResolveError
from .progress import StepDisplay, dry_run_panel, error_panel


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hf2ollama",
        description=(
            "Convert a HuggingFace model id (or local HF model directory) into a "
            "quantized .gguf file ready for Ollama, generating a Modelfile. "
            "Delegates conversion/quantization to llama.cpp's tools."
        ),
    )
    p.add_argument("input", nargs="?",
                   help="HF repo id (org/name) or a local HF model directory path.")
    p.add_argument("--quantize", "-q", default="q4_K_M",
                   help="Quant type; default q4_K_M. (try q5_K_M for higher quality, q4_K_S for smaller)")
    p.add_argument("--output-dir", "-o", default="./out",
                   help="Output directory; default ./out.")
    p.add_argument("--modelfile-name", default=None,
                   help="Base name for output files (default: derived from repo id).")
    p.add_argument("--ollama-name", default=None,
                   help="Model name registered in Ollama (default: same as modelfile-name).")
    p.add_argument("--skip-ollama-create", action="store_true",
                   help="Produce .gguf + Modelfile but do not register with Ollama.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit without executing.")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip interactive confirmation prompts.")
    p.add_argument("--hf-token", default=None,
                   help="HF token; else reads HF_TOKEN env. For gated/private repos.")
    p.add_argument("--llama-cpp-dir", default=None,
                   help="Directory containing convert_hf_to_gguf.py and llama-quantize.")
    p.add_argument("--keep-fp16", action="store_true",
                   help="Keep the intermediate fp16 GGUF after quantization.")
    p.add_argument("--state-file", default=None,
                   help="State JSON path (default: <output-dir>/.hf2ollama-state.json).")
    p.add_argument("--interactive", "-i", action="store_true",
                   help="Walk the user through the choices with prompts. Good for first-time use.")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable the live progress display (use plain text output). "
                        "Useful for logs and CI.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def derive_config(args: argparse.Namespace) -> Config:
    llama_cpp_dir = Path(args.llama_cpp_dir) if args.llama_cpp_dir else None
    if llama_cpp_dir is None:
        env_dir = os.environ.get("LLAMA_CPP_DIR")
        llama_cpp_dir = Path(env_dir) if env_dir else None

    output_dir = Path(args.output_dir)
    last_seg = (args.input or "model").split("/")[-1].split("\\")[-1]
    base = safe_name(args.modelfile_name) if args.modelfile_name else safe_name(last_seg)
    ollama_name = args.ollama_name or base
    state_file = (
        Path(args.state_file)
        if args.state_file
        else (output_dir / ".hf2ollama-state.json")
    )

    quant = _normalize_quant(args.quantize)
    if quant not in KNOWN_QUANTS:
        known = ", ".join(sorted(KNOWN_QUANTS))
        raise ResolveError(
            f"Unknown quantization type '{args.quantize}'. Valid types: {known}"
        )

    token = args.hf_token or os.environ.get("HF_TOKEN")
    return Config(
        input_ref=args.input or "",
        quant=quant,
        output_dir=output_dir,
        modelfile_name=safe_name(base),
        ollama_name=ollama_name,
        skip_ollama_create=args.skip_ollama_create,
        dry_run=args.dry_run,
        yes=args.yes,
        hf_token=token,
        llama_cpp_dir=llama_cpp_dir,
        keep_fp16=args.keep_fp16,
        state_file=state_file,
    )


def interactive_prompt(args: argparse.Namespace, console: Console) -> argparse.Namespace:
    """Walk a first-time user through the choices. Mutates and returns args."""
    console.print("[bold cyan]hf2ollama[/bold cyan] — interactive setup")
    console.print()

    if not args.input:
        example = "sshleifer/tiny-gpt2" if sys.stdin.isatty() else None
        prompt = "HuggingFace repo id (org/name) or local path"
        args.input = Prompt.ask(
            f"[bold]{prompt}[/bold]" + (f" [dim](e.g. {example})[/dim]" if example else ""),
            console=console,
        ).strip()
        if not args.input:
            raise ResolveError("input is required")

    if args.quantize == "q4_K_M":  # default; only ask if user didn't override
        console.print()
        console.print("[dim]Quantization choices (smaller file, more loss):[/dim]")
        console.print("  [cyan]q4_K_M[/cyan]  — 4-bit, medium quality  [default, recommended]")
        console.print("  [cyan]q5_K_M[/cyan]  — 5-bit, higher quality  (larger file)")
        console.print("  [cyan]q4_K_S[/cyan]  — 4-bit, smaller    (more loss)")
        console.print("  [cyan]q8_0[/cyan]    — 8-bit, near-lossless (large file)")
        console.print("  [cyan]f16[/cyan]     — full precision     (huge file, no quant step)")
        args.quantize = Prompt.ask(
            "[bold]Quantization[/bold]", default="q4_K_M", console=console
        ).strip()

    if not args.yes and not args.dry_run:
        console.print()
        derive_config(args)  # validate before prompting for confirmation
        dry_run_panel(console, derive_config(args))
        if not Confirm.ask("Proceed?", default=True, console=console):
            raise SystemExit(0)
    return args


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    try:
        if args.interactive:
            args = interactive_prompt(args, console)
        cfg = derive_config(args)
        run(cfg, console, no_progress=args.no_progress)
        return 0
    except Hf2OllamaError as exc:
        error_panel(console, "hf2ollama failed", str(exc))
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        return 130
    except SystemExit as exc:
        # from interactive_prompt on user decline
        return int(exc.code) if isinstance(exc.code, int) else 0
    except Exception as exc:
        error_panel(
            console, "unexpected error",
            f"{type(exc).__name__}: {exc}",
            hint="This is a bug in hf2ollama. Please open an issue with the full traceback.",
        )
        return 2


def run(cfg: Config, console: Console, no_progress: bool = False) -> None:
    """Top-level driver. Streams progress to the terminal as a Live checklist."""
    if cfg.dry_run:
        dry_run_panel(console, cfg)
        console.print()
        console.print("[dim]Re-run without --dry-run to execute the plan above.[/dim]")
        return

    # Lazy heavy imports kept out of module import path.
    from .resolver import resolve_input
    from .probe import probe_architecture
    from .pipeline import convert_to_fp16, quantize
    from .modelfile import generate_modelfile
    from .ollama import register_with_ollama
    from .state import StateStore

    state = StateStore.load(cfg.state_file)

    if no_progress:
        _run_plain(cfg, state, console)
        return

    with StepDisplay(console) as sd:
        try:
            # Mark each step active BEFORE the underlying call so the user
            # always sees what was in-flight when an error occurred.
            sd.start("resolve", cfg.input_ref)
            resolved = resolve_input(cfg, state, display=sd)
            sd.start("probe")
            probe_architecture(cfg, resolved, state, display=sd)
            sd.start("convert_fp16", str(resolved.local_dir))
            fp16_path = convert_to_fp16(cfg, resolved, state, display=sd)
            sd.start("quantize", cfg.quant)
            quant_path = quantize(cfg, fp16_path, state, display=sd)
            sd.start("modelfile")
            modelfile_path = generate_modelfile(cfg, resolved, quant_path, state, display=sd)
            if not cfg.skip_ollama_create:
                sd.start("ollama_create", cfg.ollama_name)
                register_with_ollama(cfg, modelfile_path, state, display=sd)
            else:
                sd.skipped("ollama_create", "--skip-ollama-create")
        except Exception as exc:
            for step_name in ("resolve", "download", "probe", "convert_fp16",
                              "quantize", "modelfile", "ollama_create"):
                rec = sd._steps.get(step_name)
                if rec and rec.status == "active":
                    sd.failed(step_name, type(exc).__name__)
            raise

        if cfg.keep_fp16 is False and fp16_path is not None and fp16_path.exists():
            fp16_path.unlink()

        # Final summary panel
        sd.summary(
            "Convert complete",
            [
                ("gguf", str(quant_path)),
                ("Modelfile", str(modelfile_path)),
                ("size", _human_size(quant_path)),
                ("ollama name", cfg.ollama_name if not cfg.skip_ollama_create else "(not registered)"),
                ("", ""),
                ("next", f"  ollama run {cfg.ollama_name}" if not cfg.skip_ollama_create
                          else f"  cd {cfg.output_dir} && ollama create {cfg.ollama_name} -f {modelfile_path.name}"),
            ],
        )


def _run_plain(cfg: Config, state, console: Console) -> None:
    """Non-Live path: for --no-progress / logs / CI. Prints step labels as text."""
    from .resolver import resolve_input
    from .probe import probe_architecture
    from .pipeline import convert_to_fp16, quantize
    from .modelfile import generate_modelfile
    from .ollama import register_with_ollama

    def step(name):
        console.print(f"[cyan]>>>[/cyan] {name}")

    step(f"resolve  {cfg.input_ref}")
    resolved = resolve_input(cfg, state)
    step(f"download -> {resolved.local_dir}")
    probe_architecture(cfg, resolved, state)
    fp16_path = convert_to_fp16(cfg, resolved, state)
    quant_path = quantize(cfg, fp16_path, state)
    modelfile_path = generate_modelfile(cfg, resolved, quant_path, state)
    if not cfg.skip_ollama_create:
        register_with_ollama(cfg, modelfile_path, state)
    if cfg.keep_fp16 is False and fp16_path is not None and fp16_path.exists():
        fp16_path.unlink()
    console.print(f"[green]Done.[/green] gguf={quant_path} modelfile={modelfile_path}")


def _human_size(path: Path) -> str:
    try:
        n = path.stat().st_size
    except OSError:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"
