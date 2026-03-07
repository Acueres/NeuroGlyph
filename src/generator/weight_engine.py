import torch

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
from transformers.generation.logits_process import LogitsProcessor
from syntax.mask_engine import MaskEngine


@dataclass(frozen=True, slots=True)
class WeightConfig:
    root_boost: float = 1.5
    semantic_boost: float = 3.0
    stop_boost: float = 0.8


def _get_bool(obj: Any, *names: str, default: bool = False) -> bool:
    for n in names:
        if hasattr(obj, n):
            try:
                return bool(getattr(obj, n))
            except Exception:
                pass
    return default


class WeightEngine:
    """
    Computes logit biases for special tokens
    """

    def __init__(
        self,
        engine: MaskEngine,
        *,
        preferred_lexemes: Sequence[str],
        config: WeightConfig | None = None,
        root_lexemes: Sequence[str],
    ) -> None:
        self._engine = engine
        self._cfg = config or WeightConfig()

        self._preferred_lexemes = tuple(
            str(x) for x in preferred_lexemes if str(x).strip()
        )
        self._root_lexemes = tuple(str(x) for x in root_lexemes if str(x).strip())

        # Cache: lexeme -> token_ids that can start spelling that lexeme.
        self._start_ids_cache: dict[str, tuple[int, ...]] = {}

    def _lexeme_start_ids(self, lexeme: str) -> tuple[int, ...]:
        lexeme = str(lexeme)
        cached = self._start_ids_cache.get(lexeme)
        if cached is not None:
            return cached

        table = self._engine.compile_lexeme_table(lexeme)

        ids = tuple(int(tid) for tid in table.allowed_start[0])
        self._start_ids_cache[lexeme] = ids
        return ids

    def compute_biases(
        self,
        *,
        allowed_token_ids: Iterable[int],
        stop_ids: Sequence[int],
        can_end_input: bool,
        root_start: bool,
        semantic_symbol_context: bool,
    ) -> dict[int, float]:
        """
        Returns token_id -> bias. Caller applies these to logits after legality masking.
        """
        allowed = set(int(x) for x in allowed_token_ids)

        biases: dict[int, float] = {}

        if semantic_symbol_context:
            boost = self._cfg.semantic_boost
            for lex in self._preferred_lexemes:
                for tid in self._lexeme_start_ids(lex):
                    if tid in allowed:
                        biases[tid] = biases.get(tid, 0.0) + boost

        if root_start:
            boost = self._cfg.root_boost
            for lex in self._root_lexemes:
                for tid in self._lexeme_start_ids(lex):
                    if tid in allowed:
                        biases[tid] = biases.get(tid, 0.0) + boost

        if can_end_input and stop_ids:
            boost = self._cfg.stop_boost
            for tid in stop_ids:
                tid = int(tid)
                if tid in allowed:
                    biases[tid] = biases.get(tid, 0.0) + boost

        return biases


class WeightLogitsProcessor(LogitsProcessor):
    def __init__(self, current_biases: Mapping[int, float]) -> None:
        super().__init__()
        self._current_biases = current_biases

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # scores: (batch, vocab)
        if not self._current_biases:
            return scores

        # Apply in-place additions to avoid extra allocations
        for tid, b in self._current_biases.items():
            if b == 0.0:
                continue
            scores[:, int(tid)] += float(b)

        return scores
