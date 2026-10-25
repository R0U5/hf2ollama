"""Input resolution and (for HF inputs) download.

Handles two steps of the pipeline:
  - resolve: validate that the input is an HF repo id or a local model dir
  - download (HF only): snapshot_download filtered to weights+tokenizer+config

The previous version made TWO sequential Hub round-trips (model_info +
list_repo_files) with no progress feedback, which manifested as a 5-30+
second silent hang on every run. This version makes ONE call (model_info)
and uses the `siblings` field it already returns to decide whether safetensors
are present.

For the download itself: snapshot_download writes its own progress bars to
stderr via tqdm. We redirect stderr to /dev/null during the call so those
bars don't leak past the Rich Live display. The user sees the StepDisplay's
"◐ Download weights — downloading… (elapsed: 38s)" spinner instead, which
is less detailed than a byte bar but actually visible inside the Live display.
"""

from __future__ import annotations

import contextlib
import fnmatch
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .config import Config, ResolvedInput
from .errors import AuthError, ResolveError
from .state import StateStore

if TYPE_CHECKING:
    from .progress import StepDisplay

ALLOW_PATTERNS = (
    "*.json",
    "*.txt",
    "*.md",
    "*.model",
    "*.safetensors",
    "*.bin",
    "*/tokenizer_config.json",
    "*/tokenizer.json",
    "*/vocab.json",
    "*/merges.txt",
    "*.jinja",
    "LICENSE",
    "LICENSE.md",
    "license.txt",
    "chat_template.jinja",
)

IGNORE_ALWAYS = (
    "*.gguf",
    "*.onnx",
    "*.h5",
    "*.msgpack",
    "onnx/*",
    "flax_*",
    "tf_*",
    "*.tflite",
)


def looks_like_repo_id(ref: str) -> bool:
    if "/" not in ref:
        return False
    return not (ref.startswith("/") or ref.startswith(".") or "\\" in ref)


def validate_local_dir(path: Path) -> None:
    if not path.is_dir():
        raise ResolveError(
            f"{path} is not a directory. Pass an HF repo id (org/name) or a local "
            "HF model directory containing config.json."
        )
    cfg = path / "config.json"
    if not cfg.is_file():
        raise ResolveError(
            f"{path} does not look like an HF model directory: no config.json found. "
            "Pass an HF repo id (org/name) or a local directory with config.json."
        )


def resolve_input(
    cfg: Config,
    state: StateStore,
    display: "StepDisplay | None" = None,
) -> ResolvedInput:
    """Resolve the input and (for HF) download weights. Resumable via state."""
    if cfg.is_hf is None:
        on_disk = Path(cfg.input_ref).is_dir()
        cfg.is_hf = (not on_disk) and looks_like_repo_id(cfg.input_ref)

    if cfg.is_hf:
        return _resolve_hf(cfg, state, display)
    return _resolve_local(cfg, state, display)


def _resolve_local(
    cfg: Config,
    state: StateStore,
    display: "StepDisplay | None" = None,
) -> ResolvedInput:
    path = Path(cfg.input_ref)
    if display is not None:
        display.start("resolve", f"local: {path}")
    validate_local_dir(path)
    result = ResolvedInput(
        kind="local",
        ref=str(path),
        revision="(local)",
        local_dir=path.resolve(),
        resolved=True,
    )
    state.set_input("local", result.ref, "(local)")
    state.mark_done("resolve", path=str(path))
    state.mark_skipped("download", reason="local input; no download needed")
    if display is not None:
        display.done("resolve", str(path))
        display.skipped("download", "local input")
    return result


def _resolve_hf(
    cfg: Config,
    state: StateStore,
    display: "StepDisplay | None" = None,
) -> ResolvedInput:
    # 1) resolve: ONE hub call. We use the `siblings` field from model_info
    # to derive both the commit SHA and safetensors presence. No second call.
    if state.is_done("resolve") and state.is_done("download"):
        ref = state.data.get("input", {}).get("ref", cfg.input_ref)
        rev = state.data.get("input", {}).get("revision", "main")
        local = state.step("download").get("path")
        local_dir = Path(local) if local else None
        if display is not None:
            display.skipped("resolve", "from state file")
            display.skipped("download", str(local_dir) if local_dir else "(cached)")
        return ResolvedInput(kind="hf", ref=ref, revision=rev, local_dir=local_dir, resolved=True)

    if display is not None:
        display.start("resolve", f"looking up {cfg.input_ref} on the Hub")

    model_info, has_safetensors = _fetch_model_info(cfg)
    sha = getattr(model_info, "sha", None) or getattr(model_info, "sha_commit", None) or "main"
    if state.is_done("resolve"):
        recorded = state.data.get("input", {}).get("revision", sha)
        revision = str(recorded)
    else:
        revision = str(sha)

    state.set_input("hf", cfg.input_ref, revision)
    state.mark_done("resolve", revision=revision, has_safetensors=has_safetensors)
    resolved = ResolvedInput(kind="hf", ref=cfg.input_ref, revision=revision)

    if display is not None:
        display.done("resolve", f"{cfg.input_ref} @ {revision[:12]}")

    # 2) download (resumable)
    if state.should_skip("download", [state.step("download").get("path")]):
        local = state.step("download").get("path")
        resolved.local_dir = Path(local)
        if display is not None:
            display.skipped("download", str(local))
        return resolved

    local_dir = _download(cfg, revision, has_safetensors, display)
    resolved.local_dir = local_dir
    state.mark_done("download", path=str(local_dir))
    if display is not None:
        display.done("download", str(local_dir))
    return resolved


def _fetch_model_info(cfg: Config):
    """Single Hub round-trip: returns (model_info, has_safetensors: bool)."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    token = cfg.hf_token
    api = HfApi(token=token)
    try:
        info = api.model_info(repo_id=cfg.input_ref, token=token if token else None)
    except GatedRepoError as exc:
        raise AuthError(
            f"repo '{cfg.input_ref}' is gated. Provide a token via --hf-token or HF_TOKEN "
            "that has been granted access."
        ) from exc
    except RepositoryNotFoundError as exc:
        msg = (
            f"repo '{cfg.input_ref}' does not exist or you lack access. "
            "Set HF_TOKEN for private repos."
            if cfg.hf_token is None
            else f"repo '{cfg.input_ref}' does not exist or you lack access."
        )
        raise ResolveError(msg) from exc
    except Exception as exc:
        raise ResolveError(
            f"could not resolve '{cfg.input_ref}' on the Hub: {exc}\n"
            "Check your network connection and that the repo id is spelled correctly."
        ) from exc

    # Derive safetensors presence from the siblings list — no second API call.
    siblings = getattr(info, "siblings", None) or []
    has_safetensors = any(
        getattr(s, "rfilename", "") and fnmatch.fnmatch(s.rfilename, "*.safetensors")
        for s in siblings
    )
    return info, has_safetensors


def _download(
    cfg: Config,
    revision: str,
    has_safetensors: bool,
    display: "StepDisplay | None" = None,
) -> Path:
    from huggingface_hub import snapshot_download
    token = cfg.hf_token

    ignore = list(IGNORE_ALWAYS)
    if has_safetensors:
        ignore = list(ignore) + ["*.bin"]

    slug = cfg.input_ref.replace("/", "--").replace("\\", "--")
    local_dir = cfg.output_dir / ".hf-cache" / f"{slug}@{revision}"
    local_dir.mkdir(parents=True, exist_ok=True)

    # Suppress ALL output during the download. The previous attempts to
    # subclass tqdm and mount a Rich Progress inside the Live display all
    # leaked - the Rich Progress own console.print() calls were rendered
    # outside the Live frame buffer between refresh ticks. The reliable
    # path is to redirect BOTH stdout and stderr to /dev/null during the
    # call, and use a simple elapsed-time spinner on the StepDisplay to
    # show the user the download is alive.
    import threading
    import time

    stop_event = threading.Event()

    def _elapsed_spinner():
        start = time.time()
        while not stop_event.wait(2.0):
            elapsed = int(time.time() - start)
            if display is not None:
                display.start("download", f"downloading\u2026 {elapsed}s elapsed")

    spinner_thread = None
    if display is not None:
        spinner_thread = threading.Thread(target=_elapsed_spinner, daemon=True)
        spinner_thread.start()

    try:
        devnull = open(os.devnull, "w")
        with contextlib.redirect_stderr(devnull), contextlib.redirect_stdout(devnull):
            snapshot_download(
                repo_id=cfg.input_ref,
                revision=revision,
                token=token if token else None,
                allow_patterns=list(ALLOW_PATTERNS),
                ignore_patterns=ignore,
                local_dir=str(local_dir),
            )
        devnull.close()
    except Exception as exc:
        raise ResolveError(
            f"download of '{cfg.input_ref}' failed: {exc}\n"
            "If this is a rate-limit error, set HF_TOKEN (or wait a few minutes)."
        ) from exc
    finally:
        stop_event.set()
        if spinner_thread is not None:
            spinner_thread.join(timeout=1.0)

    return local_dir
