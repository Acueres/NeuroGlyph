import torch

from generator.gemma3_code_generator import Gemma3CodeGenerator, Gemma3Config
from models.registry import get_model, can_attempt_load, release_cuda_memory


def main():
    try:
        spec = get_model("gemma3-12b-it-4bit")
        ok, reason = can_attempt_load(spec)
        if not ok:
            print(f"{spec.display_name}: {reason}")
            return

        gen = Gemma3CodeGenerator(
            Gemma3Config(model_id=spec.model_id, max_new_tokens=spec.max_new_tokens)
        )
        code = gen.generate(
            """Task: Generate a function 'add'.
            The function 'add' adds two ints that come as parameters, and returns an int.
            Use the 'println' function to print the result in the main function.
            Start functions with keyword def, not func."""
        )
        print("Model reply:\n")
        print(code)
    except torch.cuda.OutOfMemoryError:
        print(f"Error: CUDA OOM while loading")
        release_cuda_memory()
        return


if __name__ == "__main__":
    main()
