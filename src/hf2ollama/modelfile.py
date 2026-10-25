"""Generate an Ollama Modelfile for the quantized GGUF.

Provenance is critical (spec): the header comments record the source repo id,
the source revision (commit sha), the quantization type, the timestamp, and the
exact llama.cpp commit/version used.

The chat template comes from chat_template.jinja if present, else from
tokenizer_config.json's `chat_template` field. PARAMETERs are only emitted when
the model's generation config clearly sets them (we read generation_config.json
or config.json sampling fields and only write values that are explicitly
present, never inventing values).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import Config, ResolvedInput, _normalize_quant
from .errors import ConverterError

PARAMETER_SOURCES = (
    # (config key, ollama PARAMETER name)
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("repeat_penalty", "repeat_penalty"),
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def llama_cpp_commit(ll_dir: Path) -> str:
    """Best-effort llama.cpp commit/version for provenance.

    Uses `git -C <ll_dir> rev-parse --short HEAD` only if the dir is a git
    checkout and git is on PATH. Never fails the pipeline; returns a marker
    otherwise. This is the single allowed use of git, purely for provenance.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(ll_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return "git " + proc.stdout.strip()
    except Exception:
        pass
    return "(llama.cpp commit unknown; verify your checkout)"

def llama_cpp_version(ll_dir: Path) -> str:
    commit = llama_cpp_commit(ll_dir)
    # Some builds embed a version via `llama-server --version`, but that is
    # heavy; the commit is the more precise provenance we need.
    return commit


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _chat_template(model_dir: Path) -> str:
    """Return the chat template string, or '' if none is found."""
    jinja = model_dir / "chat_template.jinja"
    if jinja.is_file():
        try:
            return jinja.read_text(encoding="utf-8")
        except OSError:
            pass
    tok = model_dir / "tokenizer_config.json"
    data = _read_json(tok)
    if data is not None:
        ct = data.get("chat_template")
        if isinstance(ct, str) and ct.strip():
            return ct
        # Some configs keep it as a list of dicts (template dicts); prefer plain string.
        if isinstance(ct, list) and ct:
            first = ct[0]
            if isinstance(first, dict) and first.get("template"):
                return str(first["template"])
    cfg = model_dir / "config.json"
    data = _read_json(cfg)
    if data is not None:
        ct = data.get("chat_template")
        if isinstance(ct, str) and ct.strip():
            return ct
    return ""


def _parameters(model_dir: Path) -> list[tuple[str, str]]:
    """Collect explicit PARAMETER directives only when clearly indicated.

    We read generation_config.json (then config.json as a fallback) and emit a
    PARAMETER only for sampling fields the config explicitly sets. Values are
    passed through verbatim; we never invent sampling defaults.
    """
    bulk = {}
    for fname in ("generation_config.json", "config.json"):
        data = _read_json(model_dir / fname)
        if data:
            bulk.update(data)

    params: list[tuple[str, str]] = []
    for key, param_name in PARAMETER_SOURCES:
        val = bulk.get(key)
        if val is None:
            continue
        if isinstance(val, float):
            params.append((param_name, f"{val:g}"))
        elif isinstance(val, int):
            params.append((param_name, str(val)))
    return params


def _license(model_dir: Path) -> str | None:
    for name in ("LICENSE", "LICENSE.md", "license.txt"):
        p = model_dir / name
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            # Keep license sane: cap length to avoid enormous Modelfiles.
            if len(text) > 40_000:
                text = text[:40_000] + "\n[...truncated by hf2ollama]"
            return text
    return None


def _template_block(template: str) -> str:
    """Wrap template for TEMPLATE \"\"\"...\"\"\" without mangling internal quotes."""
    # Rich/Modelfile uses triple-double-quote; escape a lone closing sequence.
    # Ollama TEMPLATE can use single quotes or triple quotes. We use the
    # triple-double-quote form and defensively escape any `"""` in content.
    safe = template.replace('"""', '\\"\\"\\"')
    return f'TEMPLATE """{safe}"""'


def _license_block(license_text: str) -> str:
    safe = license_text.replace('"""', '\\"\\"\\"')
    return f'LICENSE """\n{safe}\n"""'


def _write_modelfile(
    cfg: Config,
    resolved: ResolvedInput,
    quant_path: Path,
    template: str,
    params: list[tuple[str, str]],
    license_text: str | None,
) -> Path:
    quant = _normalize_quant(cfg.quant)
    ll_dir = cfg.llama_cpp_dir
    ll_version = llama_cpp_commit(ll_dir) if ll_dir else "(llama.cpp dir not provided)"
    rev = resolved.revision or "main"

    lines: list[str] = []
    lines.append("# hf2ollama generated Modelfile")
    lines.append("# provenance — do not edit: delete and regenerate with hf2ollama")
    lines.append(f"#   source        : {resolved.ref}")
    lines.append(f"#   revision      : {rev}")
    lines.append(f"#   quantization  : {quant}")
    lines.append(f"#   generated     : {now_iso()}")
    lines.append(f"#   llama.cpp     : {ll_version}")
    lines.append("")

    # FROM uses a relative path so the Modelfile is portable from --output-dir.
    gguf_rel = quant_path.name  # relative because ollama create runs in output-dir
    lines.append(f"FROM ./{gguf_rel}")
    lines.append("")

    if template:
        lines.append(_template_block(template))
        lines.append("")

    for name, value in params:
        lines.append(f"PARAMETER {name} {value}")
    if params:
        lines.append("")

    if license_text:
        lines.append(_license_block(license_text))
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    modelfile = cfg.output_dir / cfg.modelfile_name
    modelfile_path = modelfile.with_suffix(".Modelfile")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    # Use the exact filename the flag doc implies: <modelfile-name>.Modelfile.
    modelfile_path.write_text(text, encoding="utf-8")
    return modelfile_path


def generate_modelfile(cfg: Config, resolved: ResolvedInput, quant_path: Path, state) -> Path:
    """Entry point. Resumable via state."""

    modelfile_path = cfg.output_dir / f"{cfg.modelfile_name}.Modelfile"
    if state.should_skip("modelfile", [modelfile_path]) and state.is_done("modelfile"):
        return modelfile_path

    model_dir = resolved.local_dir
    if model_dir is None:
        raise ConverterError("no model directory available to read config for Modelfile.")

    template = _chat_template(model_dir)
    params = _parameters(model_dir)
    license_text = _license(model_dir)

    path = _write_modelfile(cfg, resolved, quant_path, template, params, license_text)
    state.mark_done("modelfile", path=str(path))
    return path
