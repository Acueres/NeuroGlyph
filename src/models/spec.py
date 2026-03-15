from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    family: str
    model_id: str
    display_name: str
    params_b: float
    loader_kind: str
    quantized_4bit: bool = False
    max_new_tokens: int = 448
    use_auto_device: bool = True

    def estimated_vram_gb(self) -> float:
        """
        Conservative heuristic for preflight skipping.
        - normal models: bf16/fp16-ish estimate
        - 4-bit models: substantially reduced footprint, but still leave overhead
        """
        if self.quantized_4bit:
            # generous estimate for 4-bit weights + runtime overhead on a 24 GB card
            return self.params_b * 0.75
        return self.params_b * 2.7
