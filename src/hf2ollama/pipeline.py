"""Convert (HF -> unquantized GGUF) and Quantize (GGUF -> quantized GGUF).

Both steps invoke llama.cpp binaries via subprocess, streaming stderr to the
user, and fail with specific errors on non-zero exit.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from .config import Config, ResolvedInput, _normalize_quant
from .errors import ConverterError, QuantizeError
from .probe import find_llama_cpp_dir

console = Console()


def _find_executable(name: str, cfg: Config, kind: str) -> Path:
    """Locate a llama.cpp binary: in --llama-cpp-dir first, else on PATH."""
    ll_dir = cfg.llama_cpp_dir
    if ll_dir is not None:
        cand = ll_dir / name
        if cand.is_file():
            return cand
        # Windows: check for .exe extension (Path.is_file doesn't auto-resolve)
        if os.name == 'nt':
            cand_exe = ll_dir / f"{name}.exe"
            if cand_exe.is_file():
                return cand_exe
    on_path = shutil.which(name)
    if on_path:
        return Path(on_path)
    raise ConverterError(
        f"could not find '{name}' for {kind}. Install/build llama.cpp and either point "
        "--llama-cpp-dir at the build/bin directory or add it to PATH."
    )


def _stream_subprocess(cmd: list[str], cwd: Path, label: str, env: dict[str, str] | None = None) -> None:
    """Run a command, streaming stderr to the console, raising on non-zero exit.

    Returns nothing on success; raises the appropriate error type otherwise.
    """
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, **env} if env else None,
        )
    except OSError as exc:
        raise ConverterError(f"failed to start {label}: {exc}") from exc

    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        if line.strip():
            console.print(line)
    proc.wait()
    if proc.returncode != 0:
        tail = "\n".join(lines[-60:])
        raise ConverterError(
            f"{label} exited with code {proc.returncode}.\n" + (tail if tail else "(no output)")
        )


def convert_to_fp16(cfg: Config, resolved: ResolvedInput, state, display=None) -> Path:
    """Run convert_hf_to_gguf.py <model_dir> --outfile <out>.fp16.gguf --outtype f16."""

    fp16_path = cfg.fp16_gguf
    if state.should_skip("convert_fp16", [fp16_path]) and state.is_done("convert_fp16"):
        if display is not None:
            display.skipped("convert_fp16", str(fp16_path))
        return fp16_path

    model_dir = resolved.local_dir
    if model_dir is None:
        raise ConverterError(
            "no model directory available to convert (resolve step did not succeed)."
        )

    ll_dir = find_llama_cpp_dir(cfg)
    conv = ll_dir / "convert_hf_to_gguf.py"
    if not conv.is_file():
        raise ConverterError(f"{conv} not found. Point --llama-cpp-dir at a llama.cpp checkout.")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(conv),
        str(model_dir),
        "--outfile", str(fp16_path),
        "--outtype", "f16",
    ]
    if display is not None:
        display.start("convert_fp16", f"{conv.name} {model_dir.name} -> {fp16_path.name}")
    _stream_subprocess(cmd, cfg.output_dir, "convert_hf_to_gguf.py")

    if not fp16_path.is_file():
        raise ConverterError(
            f"conversion reported success but {fp16_path} was not produced."
        )
    state.mark_done("convert_fp16", path=str(fp16_path))
    if display is not None:
        display.done("convert_fp16", str(fp16_path))
    return fp16_path


def quantize(cfg: Config, fp16_path: Path, state, display=None) -> Path:
    """Run llama-quantize <fp16>.gguf <out>.<quant>.gguf <quant>.

    If the requested quant is f16/f32, the fp16 output already satisfies it and
    this step is skipped.
    """

    quant = _normalize_quant(cfg.quant)
    if quant in ("f16", "f32"):
        # fp16 already produced by the converter; nothing further to do.
        state.mark_skipped("quantize", reason=f"--quantize {quant} satisfied by converter output")
        if display is not None:
            display.skipped("quantize", f"fp16 satisfies --quantize {quant}")
        return fp16_path

    out_path = cfg.output_dir / f"{cfg.modelfile_name}.{quant}.gguf"
    if state.should_skip("quantize", [out_path]) and state.is_done("quantize"):
        if display is not None:
            display.skipped("quantize", str(out_path))
        return out_path

    qbin = _find_executable("llama-quantize", cfg, "quantization")
    if not fp16_path.is_file():
        raise QuantizeError(
            f"expected input GGUF {fp16_path} is missing. Re-run the convert step."
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(qbin), str(fp16_path), str(out_path), quant]
    if display is not None:
        display.start("quantize", f"llama-quantize {quant}")
    # Windows: DLLs live alongside the exe — pass ll_dir in PATH so Popen finds them
    ll_dir = cfg.llama_cpp_dir
    env = {"PATH": str(ll_dir)} if ll_dir is not None else None
    _stream_subprocess(cmd, cfg.output_dir, "llama-quantize", env=env)

    if not out_path.is_file():
        raise QuantizeError(
            f"llama-quantize reported success but {out_path} was not produced."
        )
    state.mark_done("quantize", path=str(out_path), type=quant)
    if display is not None:
        display.done("quantize", str(out_path))
    return out_path
