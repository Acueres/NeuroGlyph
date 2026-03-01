import torch

from dataclasses import dataclass
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)
from compiler_client.responses import PredictResponse
from compiler_client.fetchers import (
    fetch_language_spec,
    fetch_expected,
    fetch_semantic_hints,
)
from syntax.mask_engine import MaskEngine
from semantics.semantic_hints import SemanticHintsCache
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

        lang_spec_response = fetch_language_spec(
            "localhost:7162", root_cert_pem="./cert.pem"
        )
        self.engine = MaskEngine(lang_spec_response.spec, self.tokenizer)
        self.system_prompt = build_system_prompt(
            lang_spec_response.spec, language_name=lang_spec_response.language
        )

        semantic_hints_reply = fetch_semantic_hints(
            "localhost:7162", root_cert_pem="./cert.pem"
        )
        self.semantic_cache = SemanticHintsCache(
            self.engine, semantic_hints_reply.preferred_lexemes
        )

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

            is_type_context = False
            
            # If we’re at a boundary and have no active predictions, fetch new ones
            if self.engine.needs_predictions():
                response = predict_next_token_kinds("".join(completion_parts))
                self.engine.set_predictions(response.expected_token_kind_ids)
                can_end = bool(response.can_end_input)
                is_type_context = bool(response.type_name_context)


            # If the current pattern is already in an accepting state, we also need "post" prediction
            # (what could come next *if we stop the pattern here*), so we can mask delimiters like ')', ',', '}', etc.
            elif self.engine.needs_post_predictions():
                response = predict_next_token_kinds("".join(completion_parts))
                self.engine.set_post_predictions(response.expected_token_kind_ids)
                can_end = bool(response.can_end_input)
                is_type_context = bool(response.type_name_context)

            if is_type_context:
                self.semantic_cache.apply_type_hints()

            allowed = self.engine.allowed_token_ids()

            if can_end or not allowed:
                for sid in stop_ids:
                    allowed.add(sid)

            return list(allowed)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                eos_token_id=stop_ids,
                prefix_allowed_tokens_fn=mask_fn,
                repetition_penalty=1.1,
                renormalize_logits=True,
            )[0]

        completion_ids = generated[input_len:]
        return self.processor.decode(completion_ids, skip_special_tokens=True)


def predict_next_token_kinds(prefix_code: str) -> PredictResponse:
    response = fetch_expected("localhost:7162", prefix_code, root_cert_pem="./cert.pem")
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
