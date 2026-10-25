"""Actionable error types.

Every failure mode in the pipeline maps to a specific exception subclass so
the CLI can render a precise, actionable message instead of a bare traceback.
All inherit from Hf2OllamaError which the CLI catches at the top level.
"""

from __future__ import annotations


class Hf2OllamaError(Exception):
    """Base class for all expected hf2ollama failures."""


class UnsupportedArchitectureError(Hf2OllamaError):
    """The local llama.cpp converter does not support this model architecture."""

    def __init__(self, architecture: str | None, llama_version: str | None = None) -> None:
        self.architecture = architecture
        self.llama_version = llama_version
        detail = f"architecture '{architecture}'" if architecture else "the detected architecture"
        hint = (
            "llama.cpp does not (yet) support this model architecture. "
            "Common workarounds: use a branch/tag of llama.cpp that added support, "
            "or use a model whose config.json architecture is in llama.cpp's supported list. "
            "See https://github.com/ggml-org/llama.cpp/issues for tracker."
        ) if llama_version else (
            "llama.cpp does not (yet) support this model architecture. "
            "Check which architectures your llama.cpp build registers with: "
            "python convert_hf_to_gguf.py --print-supported-models"
        )
        super().__init__(f"Unsupported model {detail}. {hint}")


class OllamaNotFoundError(Hf2OllamaError):
    """ollama binary could not be found on PATH."""


class ResolveError(Hf2OllamaError):
    """Input could not be resolved (repo missing/gated, or not a model dir)."""


class AuthError(Hf2OllamaError):
    """HuggingFace credential missing or rejected for a gated/private repo."""


class ConverterError(Hf2OllamaError):
    """convert_hf_to_gguf.py exited non-zero."""


class QuantizeError(Hf2OllamaError):
    """llama-quantize exited non-zero."""


class StateError(Hf2OllamaError):
    """State file could not be read or written."""
