import torch

from dataclasses import dataclass
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

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
            torch_dtype=dtype,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(self.config.model_id)

    def chat(self, user_text: str, system_prompt: str | None = None) -> str:
        system_prompt = system_prompt or (
            "You are NeuroGlyph, a neuro-symbolic code assistant for the Glykon "
            "language. Answer concisely and generate clean code when asked."
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

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )[0]

        completion_ids = generated[input_len:]
        return self.processor.decode(completion_ids, skip_special_tokens=True)

def main():
    chat = Gemma3Chat()
    reply = chat.chat(
        "Say one short sentence explaining what NeuroGlyph is. "
        "Mention that it uses a compiler in the loop."
    )
    print("Model reply:\n")
    print(reply)


if __name__ == "__main__":
    main()
