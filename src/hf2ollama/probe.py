"""Architecture probing against convert_hf_to_gguf.py.

The converter has two relevant surfaces (verified against llama.cpp master):
  - `--print-supported-models` prints the registered model classes (an
    allow-list of architecture names). We parse this to learn what the local
    llama.cpp build can handle.
  - When given a real model dir it prints `Model architecture: <X>` and, if
    unsupported, logs `Model <X> is not supported` to stderr and exits 1.

We combine both: resolve the canonical architecture name from the model's
config.json (the `architectures` field), then check it against the parsed
supported set. This gives a clean, actionable error before spending time on
conversion. If the allow-list cannot be parsed we fall back to letting the
converter fail and detecting the error on stderr (no silent success).
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Config, ResolvedInput
from .errors import ConverterError, UnsupportedArchitectureError
from .state import StateStore

_SUPPORTED_RE = re.compile(r"^\s*ModelType\.\w+:\s+(.*)$")
_ARCH_RE = re.compile(r"Model architecture:\s*(\S+)")
_UNSUPPORTED_RE = re.compile(r"Model\s+(\S+)\s+is not supported")


@dataclass
class ProbeResult:
    architecture: str | None
    supported: bool
    from_allowlist: bool  # True if we trusted --print-supported-models parse


def model_architecture_name(model_dir: Path) -> str | None:
    """Read config.json's `architectures` list and return the first entry."""
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        import json
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        arch = cfg.get("architectures")
        if isinstance(arch, list) and arch:
            return str(arch[0])
        if isinstance(arch, str) and arch:
            return arch
    except Exception:
        return None
    return None


def find_llama_cpp_dir(cfg: Config) -> Path:
    d = cfg.llama_cpp_dir
    if d is None:
        raise ConverterError(
            "could not locate llama.cpp. Pass --llama-cpp-dir or set LLAMA_CPP_DIR to a "
            "directory containing convert_hf_to_gguf.py and llama-quantize."
        )
    if not d.is_dir():
        raise ConverterError(f"--llama-cpp-dir {d} is not a directory.")
    conv = d / "convert_hf_to_gguf.py"
    if not conv.is_file():
        raise ConverterError(
            f"convert_hf_to_gguf.py not found in {d}. Point --llama-cpp-dir at a llama.cpp "
            "checkout (master) which contains the tool."
        )
    return d


def parse_supported_models(help_out: str) -> set[str]:
    """Best-effort parse of --print-supported-models output.

    llama.cpp prints lines like:
        ModelType.Llama: LlamaForCausalLM, MistralForCausalLM, ...
    We lowercase and return the set of known architecture tokens plus the
    canonical ModelType names as a fallback.
    """
    supported: set[str] = set()
    for line in help_out.splitlines():
        line = line.strip()
        # capture the canonical ModelType name
        m = re.match(r"ModelType\.(\w+)\s*:", line)
        if m:
            supported.add(m.group(1).lower())
        # capture architecture tokens after the colon
        m2 = re.match(r"ModelType\.\w+\s*:\s*(.*)$", line)
        if m2:
            for token in m2.group(1).split(","):
                token = token.strip()
                if token:
                    supported.add(token.lower())
    return supported


def _prefer_allowlist(cfg: Config, model_dir: Path) -> ProbeResult | None:
    """Try --print-supported-models; return None if we cannot trust the parse."""
    ll_dir = find_llama_cpp_dir(cfg)
    conv = ll_dir / "convert_hf_to_gguf.py"
    try:
        run = subprocess.run(
            [sys.executable, str(conv), "--print-supported-models"],
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        return None
    except OSError as exc:
        raise ConverterError(f"failed to invoke {conv}: {exc}") from exc
    out = run.stdout or ""
    supported = parse_supported_models(out)
    if not supported:
        return None
    arch = model_architecture_name(model_dir)
    if arch is None:
        raise UnsupportedArchitectureError(None)
    ok = arch.lower() in supported or _model_type_token_matches(arch, supported)
    return ProbeResult(architecture=arch, supported=ok, from_allowlist=True)


def _model_type_token_matches(arch: str, supported: set[str]) -> bool:
    # Some conversions are registered under a ModelType rather than an exact
    # architecture string; accept a strict lowercased token match either way.
    return arch.lower() in supported


def probe_architecture(cfg: Config, resolved: ResolvedInput, state, display=None) -> ProbeResult:
    """Entry point. Resumable; returns a ProbeResult."""

    model_dir = resolved.local_dir
    if model_dir is None or not (model_dir / "config.json").is_file():
        raise ConverterError("no model directory available to probe.")

    if state.should_skip("probe", []) and state.is_done("probe"):
        arch = state.step("probe").get("architecture")
        if display is not None:
            display.skipped("probe", arch or "(from state)")
        return ProbeResult(architecture=arch, supported=True, from_allowlist=False)

    # Determine llama.cpp directory once (may raise if missing).
    find_llama_cpp_dir(cfg)

    # Preferred: allow-list parse.
    if display is not None:
        display.start("probe", "checking llama.cpp supported list")
    via_allowlist = _prefer_allowlist(cfg, model_dir)
    if via_allowlist is not None:
        state.mark_done("probe", architecture=via_allowlist.architecture)
        if display is not None:
            if via_allowlist.supported:
                display.done("probe", via_allowlist.architecture or "(supported)")
            else:
                display.done("probe", f"{via_allowlist.architecture} (NOT in allowlist)")
        return via_allowlist

    # Fallback: run converter on a tiny/real dir and detect error on stderr.
    result = _probe_by_execution(cfg, model_dir, state)
    if display is not None:
        display.done("probe", result.architecture or "(detected)")
    return result


def _probe_by_execution(cfg: Config, model_dir: Path, state: StateStore) -> ProbeResult:
    """Detect unsupported architecture by running the converter and scraping stderr."""
    ll_dir = find_llama_cpp_dir(cfg)
    conv = ll_dir / "convert_hf_to_gguf.py"
    # We do NOT pass --outfile: we just want the probe to either succeed (exit 0)
    # or print 'Model X is not supported' on stderr (exit 1). To avoid writing a
    # full GGUF, add --dry-run? --dry-run only prints a split plan for splitting;
    # it still fully processes. We accept that this fallback is heavier and only
    # reached when allow-list parsing failed.
    cmd = [
        sys.executable, str(conv),
        str(model_dir),
        "--outtype", "f16",
        "--outfile", str(cfg.fp16_gguf),
    ]
    try:
        run = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise ConverterError(f"failed to invoke {conv}: {exc}") from exc

    combined = (run.stdout or "") + "\n" + (run.stderr or "")
    unsupported = _UNSUPPORTED_RE.search(combined)
    if unsupported:
        arch = unsupported.group(1)
        state.mark_done("probe", architecture=arch)
        raise UnsupportedArchitectureError(arch)
    arch = model_architecture_name(model_dir)
    if run.returncode != 0:
        # Some other failure; surface stderr tail rather than guessing.
        raise ConverterError(
            f"{conv} failed during architecture probe.\n"
            + (combined.strip()[-2000:] if combined.strip() else "(no output)")
        )
    state.mark_done("probe", architecture=arch)
    if not arch:
        raise UnsupportedArchitectureError(None)
    return ProbeResult(architecture=arch, supported=True, from_allowlist=False)
