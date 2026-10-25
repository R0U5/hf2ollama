"""Configuration dataclasses and derived path handling.

Central place for the flags the CLI accepts, default-value derivation, and
validation of the quantization enum. Kept free of any I/O so it is trivially
unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# The quantization methods llama-quantize accepts. This is the 2026 K-quant /
# modern enum. F16 and F32 are special-cased in the pipeline (no downstream
# quantize step is needed because the converter already produced that type).
# q8_0 is kept here but is typically requested downstream rather than as an
# --outtype; it is included so users can ask for its output string.
KNOWN_QUANTS: frozenset[str] = frozenset({
    # unquantized passthrough
    "f16", "f32",
    # legacy linear
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
    # k-quants (modern default family)
    "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_k_s", "q4_k_m", "q4_k_l",
    "q5_k_s", "q5_k_m", "q5_k_l", "q6_k",
    # i-quants (importance-matrix quants)
    "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq2_m",
    "iq3_xxs", "iq3_xs", "iq3_s", "iq3_m", "iq4_xs", "iq4_nl",
    # ternary
    "tq1_0", "tq2_0",
})

# Quant names that need no downstream llama-quantize step because the
# converter's --outtype already produces that exact type.
TRIVIAL_QUANTS: frozenset[str] = frozenset({"f16", "f32"})

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(name: str) -> str:
    """Derive a filesystem/ollama-safe name from a repo id or local dir name."""
    cleaned = _SAFE_NAME_RE.sub("-", name).strip("-.")
    return cleaned or "model"


def quant_is_valid(quant: str) -> bool:
    return _normalize_quant(quant) in KNOWN_QUANTS


def _normalize_quant(quant: str) -> str:
    # Accept BOTH the modern lowercase llama-quantize spellings ("q4_k_m")
    # and the legacy/user README style uppercase ("Q4_K_M"). llama-quantize
    # is case-insensitive, but we normalize to lowercase for the enum check.
    return quant.replace("_", "_").lower()


@dataclass
class ResolvedInput:
    """Outcome of the resolve step."""

    kind: str  # "hf" | "local"
    ref: str  # repo id (hf) or directory path (local)
    revision: str = "main"  # commit sha / branch / tag (hf only)
    local_dir: Path | None = None  # where weights live on disk
    resolved: bool = False


@dataclass
class Config:
    """Fully-derived run configuration."""

    input_ref: str
    quant: str
    output_dir: Path
    modelfile_name: str
    ollama_name: str
    skip_ollama_create: bool
    dry_run: bool
    yes: bool
    hf_token: str | None
    llama_cpp_dir: Path | None
    keep_fp16: bool
    state_file: Path
    # plugin hooks (defaults provided in config.py:defaults())
    hf_api_factory: object | None = field(default=None, repr=False)
    is_hf: bool | None = None  # True/False/None(decide by shape)

    # Convenience accessors used by downstream modules.
    @property
    def fp16_gguf(self) -> Path:
        return self.output_dir / f"{self.modelfile_name}.fp16.gguf"

    @property
    def quant_gguf(self) -> Path:
        if _normalize_quant(self.quant) in TRIVIAL_QUANTS:
            return self.fp16_gguf
        return self.output_dir / f"{self.modelfile_name}.{_normalize_quant(self.quant)}.gguf"

    def asdict(self):  # for dry-run printing
        return {
            "input": self.input_ref,
            "quantize": _normalize_quant(self.quant),
            "output_dir": str(self.output_dir),
            "modelfile_name": self.modelfile_name,
            "ollama_name": self.ollama_name,
            "skip_ollama_create": self.skip_ollama_create,
            "keep_fp16": self.keep_fp16,
            "state_file": str(self.state_file),
        }
