import torch

from dataclasses import dataclass
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)
from compiler_client.fetchers import fetch_language_spec, fetch_expected

MODEL_ID = "google/gemma-3-4b-it"


@dataclass
class Gemma3Config:
    model_id: str = MODEL_ID
    max_new_tokens: int = 256
    use_auto_device: bool = True


class Gemma3Chat:
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

    def chat(self, user_text: str, system_prompt: str | None = None) -> str:
        system_prompt = system_prompt or (
            "You are NeuroGlyph, a neuro-symbolic code assistant for the Glykon "
            f"language. Answer concisely and generate clean code when asked."
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.model.dtype)

        input_len = inputs["input_ids"].shape[-1]

        '''
        prefix_fn = make_stmt_keyword_prefix_fn(
            self.tokenizer, ALLOWED_STMT_KEYWORDS, input_len
        )
        '''

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                #prefix_allowed_tokens_fn=prefix_fn,
            )[0]

        completion_ids = generated[input_len:]
        return self.processor.decode(completion_ids, skip_special_tokens=True)


def main():
    '''
    chat = Gemma3Chat()
    reply = chat.chat(
        "Generate a main function in Glykon."
        "In the comments mention that it uses a compiler in the loop."
    )
    print("Model reply:\n")
    print(reply)
    '''
    spec = fetch_language_spec(
        "localhost:7162",
        root_cert_pem="./cert.pem"
    )
    print(spec.spec)

    expected = fetch_expected(
        "localhost:7162",
        "def add(a: int, b: int)",
        root_cert_pem="./cert.pem"
    )

    print(expected)


if __name__ == "__main__":
    main()
