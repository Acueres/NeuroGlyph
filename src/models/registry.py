import torch

from typing import Optional
from .spec import ModelSpec


GEMMA3_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="gemma3-4b-it",
        family="gemma3",
        model_id="google/gemma-3-4b-it",
        display_name="Gemma 3 4B IT",
        params_b=4.0,
        loader_kind="gemma3",
        quantized_4bit=False,
    ),
    ModelSpec(
        key="gemma3-12b-it-4bit",
        family="gemma3",
        model_id="unsloth/gemma-3-12b-it-bnb-4bit",
        display_name="Gemma 3 12B IT (4-bit)",
        params_b=12.0,
        loader_kind="gemma3",
        quantized_4bit=True,
    ),
    ModelSpec(
        key="gemma3-27b-it-4bit",
        family="gemma3",
        model_id="unsloth/gemma-3-27b-it-bnb-4bit",
        display_name="Gemma 3 27B IT (4-bit)",
        params_b=27.0,
        loader_kind="gemma3",
        quantized_4bit=True,
    ),
)

QWEN25_INSTRUCT_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="qwen25-0.5b-instruct",
        family="qwen25",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        display_name="Qwen2.5 0.5B Instruct",
        params_b=0.5,
        loader_kind="qwen25",
        quantized_4bit=False,
    ),
    ModelSpec(
        key="qwen25-1.5b-instruct",
        family="qwen25",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        display_name="Qwen2.5 1.5B Instruct",
        params_b=1.5,
        loader_kind="qwen25",
        quantized_4bit=False,
    ),
    ModelSpec(
        key="qwen25-3b-instruct",
        family="qwen25",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        display_name="Qwen2.5 3B Instruct",
        params_b=3.0,
        loader_kind="qwen25",
        quantized_4bit=False,
    ),
    ModelSpec(
        key="qwen25-7b-instruct",
        family="qwen25",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        display_name="Qwen2.5 7B Instruct",
        params_b=7.0,
        loader_kind="qwen25",
        quantized_4bit=False,
    ),
    ModelSpec(
        key="qwen25-14b-instruct-4bit",
        family="qwen25",
        model_id="unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        display_name="Qwen2.5 14B Instruct (4-bit)",
        params_b=14.0,
        loader_kind="qwen25",
        quantized_4bit=True,
    ),
    ModelSpec(
        key="qwen25-32b-instruct-4bit",
        family="qwen25",
        model_id="unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
        display_name="Qwen2.5 32B Instruct (4-bit)",
        params_b=32.0,
        loader_kind="qwen25",
        quantized_4bit=True,
    ),
)

QWEN25_CODER_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="qwen25-coder-0.5b-instruct",
        family="qwen25_coder",
        model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        display_name="Qwen2.5-Coder 0.5B Instruct",
        params_b=0.5,
        loader_kind="qwen25_coder",
        quantized_4bit=False,
    ),
    ModelSpec(
        key="qwen25-coder-1.5b-instruct",
        family="qwen25_coder",
        model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        display_name="Qwen2.5-Coder 1.5B Instruct",
        params_b=1.5,
        loader_kind="qwen25_coder",
        quantized_4bit=False,
    ),
    ModelSpec(
        key="qwen25-coder-3b-instruct",
        family="qwen25_coder",
        model_id="Qwen/Qwen2.5-Coder-3B-Instruct",
        display_name="Qwen2.5-Coder 3B Instruct",
        params_b=3.0,
        loader_kind="qwen25_coder",
        quantized_4bit=False,
    ),
    ModelSpec(
        key="qwen25-coder-7b-instruct",
        family="qwen25_coder",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        display_name="Qwen2.5-Coder 7B Instruct",
        params_b=7.0,
        loader_kind="qwen25_coder",
        quantized_4bit=False,
    ),
    ModelSpec(
        key="qwen25-coder-14b-instruct-4bit",
        family="qwen25_coder",
        model_id="unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit",
        display_name="Qwen2.5-Coder 14B Instruct (4-bit)",
        params_b=14.0,
        loader_kind="qwen25_coder",
        quantized_4bit=True,
    ),
    ModelSpec(
        key="qwen25-coder-32b-instruct-4bit",
        family="qwen25_coder",
        model_id="unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit",
        display_name="Qwen2.5-Coder 32B Instruct (4-bit)",
        params_b=32.0,
        loader_kind="qwen25_coder",
        quantized_4bit=True,
    ),
)

ALL_MODELS: tuple[ModelSpec, ...] = (
    *GEMMA3_MODELS,
    *QWEN25_INSTRUCT_MODELS,
    *QWEN25_CODER_MODELS,
)


def get_model(key: str) -> ModelSpec:
    for m in ALL_MODELS:
        if m.key == key:
            return m
    raise KeyError(f"Unknown model key: {key}")


def current_cuda_vram_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024**3)


def can_attempt_load(
    spec: ModelSpec, safety_fraction: float = 0.90
) -> tuple[bool, str | None]:
    vram_gb = current_cuda_vram_gb()
    if vram_gb is None:
        return False, "CUDA device not available"

    budget = vram_gb * safety_fraction
    estimated = spec.estimated_vram_gb()

    if estimated > budget:
        return False, f"estimated {estimated:.1f} GB > safe budget {budget:.1f} GB"

    return True, None


def release_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
