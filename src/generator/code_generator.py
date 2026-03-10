import torch

from dataclasses import dataclass
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    LogitsProcessorList,
)
from compiler_client.responses import PredictResponse
from compiler_client.fetchers import (
    get_language_spec,
    get_expected,
    get_semantic_hints,
)
from syntax.mask_engine import MaskEngine
from semantics.semantic_hints import SemanticHintsCache
from .weight_engine import WeightEngine, WeightLogitsProcessor, WeightConfig
from .system_prompt import build_system_prompt

MODEL_ID = "google/gemma-3-4b-it"


@dataclass
class Gemma3Config:
    model_id: str = MODEL_ID
    max_new_tokens: int = 256
    use_auto_device: bool = True


class Gemma3CodeGenerator:
    def __init__(self, config: Gemma3Config | None = None) -> None:
        self.config = config or Gemma3Config()

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.config.model_id,
            device_map="auto" if self.config.use_auto_device else None,
            dtype=dtype,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(self.config.model_id)
        self.tokenizer = self.processor.tokenizer

        lang_spec_response = get_language_spec(
            "localhost:7162", root_cert_pem="./cert.pem"
        )
        self.engine = MaskEngine(lang_spec_response.spec, self.tokenizer)
        self.system_prompt = build_system_prompt(
            lang_spec_response.spec, language_name=lang_spec_response.language
        )

        semantic_hints_reply = get_semantic_hints(
            "localhost:7162", root_cert_pem="./cert.pem"
        )
        self.semantic_cache = SemanticHintsCache(
            self.engine, semantic_hints_reply.preferred_lexemes
        )

        self._current_biases: dict[int, float] = {}

        self.weight_engine = WeightEngine(
            self.engine,
            preferred_lexemes=semantic_hints_reply.preferred_lexemes,
            config=WeightConfig(
                root_boost=1.5,
                semantic_boost=3.0,
                stop_boost=0.8,
            ),
            root_lexemes=[root.literal for root in lang_spec_response.spec.root_tokens],
        )
        self.weight_processor = WeightLogitsProcessor(self._current_biases)

    def generate(self, user_text: str) -> str:
        prompt = f"{self.system_prompt}\n\n" f"Task:\n{user_text}\n"

        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.model.dtype)

        input_len = inputs["input_ids"].shape[-1]

        self.engine.reset()

        completion_parts: list[str] = []

        # Track how many tokens we have already "replayed" into the engine
        last_seen_len = input_len

        can_end = False
        stop_ids = get_stop_ids(self.tokenizer, self.model)

        def mask_fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
            nonlocal last_seen_len, can_end

            self._current_biases.clear()

            cur_len = int(input_ids.shape[-1])

            # Replay any newly generated tokens into the engine
            while last_seen_len < cur_len:
                tid = int(input_ids[last_seen_len].item())
                # append decoded text for server context
                completion_parts.append(
                    self.tokenizer.decode(
                        [tid],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                )

                # advance engine state
                self.engine.consume(tid)
                last_seen_len += 1

            response = None
            semantic_symbol_context = False
            root_start = False

            # If we’re at a boundary and have no active predictions, fetch new ones
            if self.engine.needs_predictions():
                response = predict_next_token_kinds("".join(completion_parts))
                self.engine.set_predictions(response.expected_token_kind_ids)
                can_end = response.can_end_input
                semantic_symbol_context = response.semantic_symbol_context
                root_start = response.root_start

            # If the current pattern is already in an accepting state, we also need "post" prediction
            # (what could come next *if we stop the pattern here*), so we can mask delimiters like ')', ',', '}', etc.
            elif self.engine.needs_post_predictions():
                response = predict_next_token_kinds("".join(completion_parts))
                self.engine.set_post_predictions(response.expected_token_kind_ids)
                can_end = response.can_end_input
                semantic_symbol_context = response.semantic_symbol_context
                root_start = response.root_start

            if semantic_symbol_context:
                self.semantic_cache.apply_type_hints()

            allowed = self.engine.allowed_token_ids()

            for sid in stop_ids:
                allowed.add(sid)

            if response is not None:
                self._current_biases.update(
                    self.weight_engine.compute_biases(
                        allowed_token_ids=allowed,
                        stop_ids=stop_ids,
                        can_end_input=can_end,
                        semantic_symbol_context=semantic_symbol_context,
                        root_start=root_start,
                    )
                )

            return list(allowed)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                eos_token_id=stop_ids,
                prefix_allowed_tokens_fn=mask_fn,
                logits_processor=LogitsProcessorList([self.weight_processor]),
                repetition_penalty=1.1,
                renormalize_logits=True,
            )[0]

        completion_ids = generated[input_len:]
        return self.processor.decode(completion_ids, skip_special_tokens=True)


def predict_next_token_kinds(prefix_code: str) -> PredictResponse:
    response = get_expected("localhost:7162", prefix_code, root_cert_pem="./cert.pem")
    return response


def get_stop_ids(tokenizer, model) -> list[int]:
    ids: list[int] = []

    # generation_config eos (can be int or list[int])
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(gen_eos, int):
        ids.append(gen_eos)
    elif isinstance(gen_eos, (list, tuple)):
        ids.extend(int(x) for x in gen_eos if x is not None)

    # tokenizer eos
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))

    # Gemma IT stop token: <end_of_turn>
    try:
        eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(eot, int) and eot >= 0:
            ids.append(int(eot))
    except Exception:
        pass

    # de-dup, keep order
    out = []
    seen = set()
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
