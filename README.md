# NeuroGlyph

NeuroGlyph is a Python prototype for constrained generation over formal languages. It combines a local language model with a symbolic language service that supplies grammar-aware predictions during decoding.

The current implementation is designed around a language specification received from an external compiler service.

## Features

- constrained decoding with token masks derived from compiler predictions
- support for fixed tokens and lexer-machine based token classes
- semantic lexeme hints for selected contexts
- weighted decoding biases for preferred roots, semantic hints, and stopping
- gRPC client for language specification, next-token prediction, input analysis, and evaluation
- experiment suites for comparing constrained and unconstrained generation

## Project structure

src/
  compiler_client/   gRPC client and language-spec DTOs
  generator/         model wrapper, system prompt, weight engine
  models/            model registry and loading checks
  semantics/         semantic hint cache
  syntax/            mask engine, token tries, lexer machine runner

experiments/         experiment runners and result output
tests/               unit tests for syntax infrastructure
compiler.proto       compiler service API

## Requirements

* Python 3.12+
* CUDA-capable GPU for local model inference
* a running Glykon-compatible compiler service
* valid TLS certificate file for the gRPC connection

Python dependencies are declared in `pyproject.toml`.

## Setup

Install dependencies:

```bash
uv sync
```

Generate gRPC bindings:

```bash
uv run gen-grpc
```

Start the compiler service separately. By default, NeuroGlyph expects it at:

```text
localhost:7162
```

The repository expects the root certificate at:

```text
./cert.pem
```

## Usage

Run the sample generator entry point:

```bash
uv run python src/main.py
```

The default entry point loads a configured model from the registry, retrieves the language specification from the compiler service, and generates a constrained Glykon program for the sample task.

To use the generator directly:

```python
from generator.gemma3_code_generator import Gemma3CodeGenerator, Gemma3Config

gen = Gemma3CodeGenerator(
    Gemma3Config(
        model_id="google/gemma-3-4b-it",
        max_new_tokens=448,
    )
)

code = gen.generate("Write a function named add that adds two integers.")
print(code)
```

## Model registry

Models are defined in `src/models/registry.py`. The registry includes Gemma 3, Qwen2.5 Instruct, and Qwen2.5-Coder variants, including 4-bit models for larger checkpoints.

Before loading, the registry can estimate whether a model is likely to fit into available CUDA memory.

## Compiler service API

The external compiler service must implement the API defined in `compiler.proto`:

* `GetLanguageSpec`
* `PredictNext`
* `GetSemanticHints`
* `AnalyzeInput`
* `EvaluateInput`

NeuroGlyph uses these endpoints to build masks, request semantic hints, analyze generated input, and evaluate runnable programs.

## Experiments

Experiment code is stored in `experiments/`. The shared runner supports:

* constrained and unconstrained generation
* syntax, parse, and semantic error counts
* evaluation success
* output correctness
* runtime measurement
* JSON/JSONL result output

Run experiment modules directly with Python after starting the compiler service.

## Tests

Run tests with:

```bash
uv run pytest
```

The current tests cover fixed-token trie behavior and lexer-machine execution.
