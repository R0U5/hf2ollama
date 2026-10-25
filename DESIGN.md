# hf2ollama Design

## Research summary

The project targets users who want a HuggingFace model usable in Ollama with
one command, bridging the HF ↔ GGUF ↔ Ollama gap that no single existing tool
covers cleanly. The design delegates all GGUF encoding to llama.cpp's own
`convert_hf_to_gguf.py` and `llama-quantize` via subprocess, avoiding any
dependency on torch, transformers, or the gguf-py package. This means the tool
can be installed as a pure-Python package (`huggingface_hub`, `requests`, `rich`,
`tqdm`) while still producing format-correct GGUF files.

## Target: llama.cpp v0.3.0 (commit c1d0e7a)

Chosen because:
- Latest tagged release; `convert_hf_to_gguf.py` ships with `--print-supported-models`
  which enables offline architecture validation without running the full converter.
- Stable subprocess interface; v0.3.0 is widely deployed in prebuilt binaries.

## Surprises

- `convert_hf_to_gguf.py` hard-depends on torch ≥ 2.0 in its own environment;
  we must never import it, only shell out.
- `snapshot_download` with `local_dir` avoids HF symlink cache, critical on
  Windows without Developer Mode enabled.
- Some quant values (`f16`, `f32`) satisfy the request without a separate
  `llama-quantize` step — the pipeline skips it.
- `ollama create` must run from inside `--output-dir` so the relative `FROM`
  path in the Modelfile resolves correctly.

## File tree

```
hf2ollama/
  pyproject.toml          package metadata, entry point, ruff/pytest config
  README.md               install, quickstart, flag ref, troubleshooting, lossy note
  DESIGN.md               this file — research findings and design rationale
  .gitignore              standard Python ignore
  src/hf2ollama/
    __init__.py           exposes __version__
    cli.py                argparse, derive_config(), run() orchestration, dry-run table
    config.py             KNOWN_QUANTS list, Config/ResolvedInput dataclasses
    errors.py             Hf2OllamaError hierarchy (7 named subclasses)
    resolver.py           HF/local input classification, snapshot_download with filtering
    probe.py              architecture probing via --print-supported-models + stderr fallback
    pipeline.py           convert_to_fp16() and quantize() subprocess wrappers
    modelfile.py          Modelfile generation (FROM, TEMPLATE, PARAMETER, LICENSE, provenance)
    ollama.py             ollama detect + best-effort ollama create
    state.py              StateStore JSON load/save/skip-decisions for resumable pipeline
  tests/
    conftest.py           shared fixture (copyable local HF model dir)
    fixtures/             minimal config.json, chat_template.jinja, etc.
    test_modelfile.py     unit tests for Modelfile generation (incl. snapshot test)
    test_state.py         unit tests for state resume/skip logic
    test_probe.py         unit tests for architecture probing + unsupported error path
    test_resolver.py      unit tests for input classification and filtering
    test_dry_run.py       dry-run + quant-rejection tests (full pipeline test: skipped)
```

## Biggest risk

Unsupported model architectures surfacing only at conversion time. Mitigated
by a two-layer check: parse `--print-supported-models` allow-list first (cheap,
pre-download), then fall back to running the converter and detecting the
`Model X is not supported` stderr pattern. Both layers run before the user
spends time on a full conversion.
