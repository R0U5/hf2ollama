# hf2ollama

Convert a HuggingFace model into a quantized GGUF ready for Ollama, with one command.

```
$ hf2ollama Qwen/Qwen2.5-7B-Instruct
╭───────────────────────────────── hf2ollama ──────────────────────────────────╮
│  ✓  Resolve input          Qwen/Qwen2.5-7B-Instruct @ a7d3c81e                │
│  ◐  Download weights       snapshot_download                                  │
│  Downloading ███████████░░░░░░░░░░  3.2/14.5 GB  85 MB/s  00:38              │
│  ✓  Download weights       ./out/.hf-cache/qwen--qwen2.5-7b-instruct@…       │
│  ✓  Probe architecture     Qwen2ForCausalLM                                  │
│  ✓  Convert to fp16 GGUF   ./out/qwen2.5-7b-instruct.fp16.gguf               │
│  ✓  Quantize GGUF          q4_K_M  → ./out/qwen2.5-7b-instruct.q4_K_M.gguf    │
│  ✓  Generate Modelfile     ./out/qwen2.5-7b-instruct.Modelfile               │
│  ✓  Register with Ollama   qwen2.5-7b-instruct                               │
╰──────────────────────────────────────────────────────────────────────────────╯
╭────────────────────── ✓ Convert complete ──────────────────────══════════════╗
│  gguf          ./out/qwen2.5-7b-instruct.q4_K_M.gguf                       │
│  Modelfile     ./out/qwen2.5-7b-instruct.Modelfile                          │
│  size          4.3 GB                                                          │
│  ollama name   qwen2.5-7b-instruct                                            │
│                                                                              │
│  next          ollama run qwen2.5-7b-instruct                                │
╰──────────────────────────────────────────────────────────────────────────────╯
Done.
```

## Why

The HF → GGUF → Ollama pipeline takes several steps and no single tool covers
it cleanly. `hf2ollama` is the missing "one command" wrapper:

- `huggingface_hub` to download the weights
- llama.cpp's `convert_hf_to_gguf.py` to write the unquantized GGUF
- llama.cpp's `llama-quantize` to quantize
- A generated `Modelfile` with chat template, sampling params, and provenance
- `ollama create` to register the model

It does NOT pull in `torch`, `transformers`, or `gguf-py` as a dependency.
You bring your own llama.cpp checkout (path or `LLAMA_CPP_DIR`); we shell
out to its tools.

## Install

```bash
# 1. install the CLI
pipx install git+https://github.com/R0U5/hf2ollama.git
# or: uv tool install git+https://github.com/R0U5/hf2ollama.git

# 2. have llama.cpp ready (one of):
#    a) clone + build llama.cpp:
#       git clone https://github.com/ggml-org/llama.cpp
#       cmake -B llama.cpp/build && cmake --build llama.cpp/build --config Release
#    b) install a prebuilt binary that ships convert_hf_to_gguf.py + llama-quantize
#
# 3. point at it:
export LLAMA_CPP_DIR=/path/to/llama.cpp
# (or pass --llama-cpp-dir each time)

# 4. (optional) for Ollama registration:
# install ollama from https://ollama.com

# 5. (optional) for gated/private HF repos:
export HF_TOKEN=hf_xxx
```

## Quickstart

```bash
# First time? Walk through the choices interactively:
hf2ollama --interactive

# Or just give it a model and a quant:
hf2ollama Qwen/Qwen2.5-7B-Instruct --quantize q4_K_M

# Use the generated model:
ollama run qwen2.5-7b-instruct
```

The CLI:

- Shows a live progress checklist with the current step and a per-step detail
- Shows a download bar (MB / total, transfer rate, elapsed) for HF downloads
- Streams stderr from the converter and quantizer in real time
- Renders a final summary panel with paths, size, and the next `ollama run` command
- Caches everything in a state file so re-running resumes mid-pipeline
- Falls back to plain text output with `--no-progress` for logs and CI

## Flags

```
hf2ollama [input] [options]

input                   HF repo id (org/name) or a local HF model directory path
--quantize, -q          Quantization type (default: q4_K_M)
--output-dir, -o        Output directory (default: ./out)
--modelfile-name        Base name for output files (default: derived from repo id)
--ollama-name           Model name registered in Ollama (default: same as modelfile-name)
--skip-ollama-create    Produce .gguf + Modelfile but don't register with Ollama
--dry-run               Print the plan and exit without executing
--yes, -y               Skip interactive confirmation prompts
--hf-token              HF token; else reads HF_TOKEN env
--llama-cpp-dir         Directory containing convert_hf_to_gguf.py and llama-quantize
--keep-fp16             Keep the intermediate fp16 GGUF after quantization
--state-file            State JSON path (default: <output-dir>/.hf2ollama-state.json)
--interactive, -i       Walk the user through the choices with prompts
--no-progress           Disable the live progress display (use plain text output)
--version               Show version
```

### Quantization types

`f16`, `f32` — unquantized passthrough (no separate quantize step)

Legacy: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`

K-quants (recommended): `q2_k`, `q3_k_s`, `q3_k_m`, `q3_k_l`, `q4_k_s`, `q4_k_m`, `q4_k_l`, `q5_k_s`, `q5_k_m`, `q5_k_l`, `q6_k`

I-quants (importance-matrix, smaller but slower): `iq1_s`, `iq1_m`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq2_m`, `iq3_xxs`, `iq3_xs`, `iq3_s`, `iq3_m`, `iq4_xs`, `iq4_nl`

Ternary: `tq1_0`, `tq2_0`

### Pipeline state

Each run writes `.hf2ollama-state.json` in the output dir. Re-running with
the same args resumes from the last completed step. To start over, delete
the state file (and the partial outputs) or pass `--state-file PATH` pointing
elsewhere.

## Examples

```bash
# Dry run: see the plan without executing
hf2ollama meta-llama/Llama-3.1-8B-Instruct --dry-run

# High quality 5-bit quant
hf2ollama meta-llama/Llama-3.1-8B-Instruct -q q5_K_M

# Small + fast 4-bit
hf2ollama meta-llama/Llama-3.1-8B-Instruct -q q4_K_S

# Local model dir
hf2ollama /path/to/my/model --skip-ollama-create

# Gated repo
HF_TOKEN=hf_xxx hf2ollama meta-llama/Llama-3-70B-Instruct -q q4_K_M

# Different output location
hf2ollama Qwen/Qwen2.5-7B-Instruct -o ~/models/qwen

# Just the GGUF, no Ollama
hf2ollaya Qwen/Qwen2.5-7B-Instruct --skip-ollama-create --keep-fp16
```

## Troubleshooting

### "Unsupported model architecture X"

Your local llama.cpp build doesn't know about the architecture in
`config.json`. Options:

- Update llama.cpp to a newer version (`git pull && cmake --build build`)
- Use a different revision of the model whose architecture is supported
- Check what's supported:
  `python <llama.cpp>/convert_hf_to_gguf.py --print-supported-models`

### "Could not resolve 'X' on the Hub"

- Check the repo id is spelled correctly
- For gated/private repos, set `HF_TOKEN`
- Check your network connection

### "ollama not found on PATH"

- Install ollama from https://ollama.com
- Or use `--skip-ollama-create` and run `ollama create` manually:
  `cd <output-dir> && ollama create <name> -f <modelfile>.Modelfile`

### "Rate limit" on HF download

- Set `HF_TOKEN` to authenticate (raises rate limits)
- Wait a few minutes and retry

### "Out of disk"

A 70B model at q4_K_M needs ~40 GB. The state file cache is small but the
GGUF + intermediate fp16 are large. Pre-allocate space or use a smaller quant.

### Safetensors vs pickle (.bin) weights

If a model ships both `.safetensors` (preferred) and `.bin` (pickle, can be
unsafe to load), we always prefer safetensors and skip `.bin`. The state
file records the choice so a re-run doesn't refetch.

## ⚠️ Lossy quantization

Quantization is lossy. Test your model's quality after conversion —
especially for tasks that are sensitive to numerical precision (math, code,
long-context retrieval). A quant that works for a chat model may not work
for a code model. When in doubt, start with q5_K_M or q6_K and only drop
to q4 if the quality loss is acceptable.

## Development

```bash
git clone https://github.com/R0U5/hf2ollama.git
cd hf2ollama
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Run with a real model to see the full pipeline:

```bash
.venv/bin/hf2ollama sshleifer/tiny-gpt2 --skip-ollama-create --keep-fp16
```

## License

MIT
