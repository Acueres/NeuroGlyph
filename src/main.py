import torch

from dataclasses import dataclass
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)

MODEL_ID = "google/gemma-3-4b-it"

ALLOWED_STMT_KEYWORDS = ["def", "let", "const", "if", "while", "for"]

def build_keyword_prefix_tables(keywords: list[str]) -> tuple[set[str], dict[str, set[str]]]:
    """
    Build:
      - prefixes: set of all prefixes (including "")
      - transitions[prefix] = set of all valid next pieces (not just whole tails)
        that keep the string being built a valid prefix of some keyword.
    """
    prefixes: set[str] = set()
    
    # Collect all valid prefixes
    for kw in keywords:
        for i in range(len(kw) + 1):
            prefixes.add(kw[:i])

    # Initialize transitions for every prefix
    transitions: dict[str, set[str]] = {p: set() for p in prefixes}

    # Populate transitions
    for kw in keywords:
        n = len(kw)
        # For every possible split point in the keyword (defining the prefix)
        for i in range(n + 1):
            prefix = kw[:i]
            
            # Add prefixes of the remaining tail
            for j in range(i + 1, n + 1):
                tail = kw[i:j]
                transitions[prefix].add(tail)

    return prefixes, transitions


def make_stmt_keyword_prefix_fn(tokenizer, keywords: list[str], input_len: int):
    """
    Creates a function to constrain generation to start with specific keywords.
    """

    prefixes, transitions = build_keyword_prefix_tables(keywords)
    
    # 2. Group all vocabulary tokens by their first character.
    tokens_by_start_char = {}
    
    # We decodes all tokens once. 
    for tok_id in range(tokenizer.vocab_size):
        text = tokenizer.decode([tok_id])
        if not text:
            continue
        
        first_char = text[0]
        if first_char not in tokens_by_start_char:
            tokens_by_start_char[first_char] = []
        tokens_by_start_char[first_char].append((tok_id, text))

    # Precompute allowed token IDs for every valid prefix state.
    valid_next_ids = {}

    for p in prefixes:
        # If 'p' itself is a keyword, we allow the user to 'break out'
        if p in keywords:
            valid_next_ids[p] = list(range(tokenizer.vocab_size))
            continue

        allowed = []
        possible_tails = transitions[p]
        
        # We only need to check tokens that start with the same characters as our valid tails
        # Get all unique starting characters for the possible tails
        valid_start_chars = {t[0] for t in possible_tails if t}

        for char in valid_start_chars:
            if char in tokens_by_start_char:
                for tok_id, tok_text in tokens_by_start_char[char]:
                    # A token is valid if:
                    # A) It is a prefix of a valid tail (e.g. prefix="con", tail="st", token="s")
                    #    -> New state will be "cons" (still in prefixes)
                    # B) It contains the valid tail as a prefix (e.g. prefix="con", tail="st", token="stant")
                    #    -> New state "constant" (completes the keyword "const" and adds more)
                    
                    is_valid_continuation = False
                    
                    # Check against all tails starting with this char
                    for tail in possible_tails:
                        if tok_text.startswith(tail) and tok_text in prefixes: 
                            # Case B: Token completes the tail (and maybe more)
                            is_valid_continuation = True
                            break
                        if tail.startswith(tok_text):
                            # Case A: Token is a sub-segment of the tail
                            is_valid_continuation = True
                            break
                    
                    if is_valid_continuation:
                        allowed.append(tok_id)
        
        valid_next_ids[p] = allowed

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
        # Decode the recent context to find the current line
        input_list = input_ids[input_len:].tolist()
        
        # Look at last ~20 tokens (sufficient for keywords) to find context
        chunk_ids = input_list[-20:] 
        text_chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)

        if '\n' in text_chunk:
            # Extract everything after the last newline
            current_line = text_chunk.split('\n')[-1]
        else:
            # If no newline found in chunk, we assume we are in the middle of a long line
            # UNLESS the total sequence length is short (beginning of file)
            if len(input_list) < 20:
                current_line = text_chunk
            else:
                # Mid-line context -> Allow everything
                return list(range(tokenizer.vocab_size))

        # Check strict prefix matching
        # If the current line exactly matches a known prefix state, return precomputed allowed tokens
        if current_line in valid_next_ids:
            return valid_next_ids[current_line]

        # Fallback
        # If the current_line is not in our prefix map, it might mean we have already
        # successfully typed a keyword and moved past it (e.g. "const a").
        # We verify if the line *starts* with a valid keyword.
        for kw in keywords:
            if current_line.startswith(kw):
                return list(range(tokenizer.vocab_size))

        # Allow everything
        return list(range(tokenizer.vocab_size))

    return prefix_allowed_tokens_fn


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
            f"language. Answer concisely and generate clean code when asked. Allowed keywords: {' '.join(ALLOWED_STMT_KEYWORDS)}"
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

        prefix_fn = make_stmt_keyword_prefix_fn(self.tokenizer, ALLOWED_STMT_KEYWORDS, input_len)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                prefix_allowed_tokens_fn=prefix_fn,
            )[0]

        completion_ids = generated[input_len:]
        return self.processor.decode(completion_ids, skip_special_tokens=True)


def main():
    chat = Gemma3Chat()
    reply = chat.chat(
        "Generate a main function in Glykon."
        "In the comments mention that it uses a compiler in the loop."
    )
    print("Model reply:\n")
    print(reply)


if __name__ == "__main__":
    main()
