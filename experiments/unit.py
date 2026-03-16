import torch

from experiments.base import (
    ExperimentCase,
    ExperimentRunner,
    bootstrap_repo_root,
    set_torch_seed,
)
from typing import Callable


def main() -> None:
    bootstrap_repo_root()

    from generator.gemma3_code_generator import (
        Gemma3CodeGenerator,
        Gemma3Config,
        get_stop_ids,
    )
    from compiler_client.fetchers import analyze_input
    from compiler_client.responses import AnalyzeInputResponse
    from models.registry import get_model, can_attempt_load, release_cuda_memory

    def analyze_input_fn() -> Callable[[str], AnalyzeInputResponse]:
        def wrapper(text: str):
            return analyze_input("localhost:7162", text, root_cert_pem="./cert.pem")

        return wrapper

    spec = get_model("gemma3-12b-it-4bit")
    ok, reason = can_attempt_load(spec)
    if not ok:
        print(f"[SKIP] {spec.display_name}: {reason}")
        return

    try:
        gen = Gemma3CodeGenerator(
            Gemma3Config(model_id=spec.model_id, max_new_tokens=spec.max_new_tokens)
        )

        # --------
        # Unconstrained generation
        # --------
        def generate_unconstrained(case: ExperimentCase) -> str:
            set_torch_seed(case.seed)

            prompt = f"{gen.system_prompt}\n\nTask:\n{case.task}\n"
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            inputs = gen.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(gen.model.device, dtype=gen.model.dtype)

            input_len = inputs["input_ids"].shape[-1]
            stop_ids = get_stop_ids(gen.tokenizer, gen.model)

            with torch.inference_mode():
                out = gen.model.generate(
                    **inputs,
                    max_new_tokens=gen.config.max_new_tokens,
                    do_sample=True,
                    eos_token_id=stop_ids,
                    repetition_penalty=1.1,
                    renormalize_logits=True,
                )[0]

            completion_ids = out[input_len:]
            return gen.processor.decode(completion_ids, skip_special_tokens=True)

        # --------
        # Constrained generation
        # --------
        def generate_constrained(case: ExperimentCase) -> str:
            set_torch_seed(case.seed)
            return gen.generate(case.task)

        # Syntax prompts
        cases = [
            ExperimentCase(
                name="func_add_int",
                task="Generate a function `add(x: int, y: int) -> int` that returns `x + y`.",
                seed=1,
            ),
            ExperimentCase(
                name="func_add_real",
                task="Generate a function `add(x: real, y: real) -> real` that returns `x + y`.",
                seed=1,
            ),
            ExperimentCase(
                name="let_decl",
                task="Inside a function body, declare a variable `sum: int` initialized to `1 + 2`.",
                seed=2,
            ),
            ExperimentCase(
                name="return_stmt",
                task="Generate a function `id(x: int) -> int` whose body is a single return statement.",
                seed=3,
            ),
            ExperimentCase(
                name="if_else",
                task="Generate a function `max2(a: int, b: int) -> int` using an if/else and return.",
                seed=4,
            ),
            ExperimentCase(
                name="while_loop",
                task="Generate a function that uses a while-loop to sum integers from 1 to 10 and returns the sum.",
                seed=5,
            ),
            ExperimentCase(
                name="for_range",
                task="Generate a function that iterates `i` from 0 to 10 inclusive and calls `println(i as str)` each iteration.",
                seed=6,
            ),
            ExperimentCase(
                name="initializer_new",
                task="Generate a snippet that creates a 'Point' object with positional parameters 1 and 2.",
                seed=7,
            ),
            ExperimentCase(
                name="cast_as",
                task="Generate a snippet that casts `x` to type `int` using `as`.",
                seed=8,
            ),
            ExperimentCase(
                name="two_funcs_main",
                task="Generate two functions: `add(int,int)->int` and `main()` that calls add and prints the result using 'println' function. Use 'println(val as str)' to cast the value into a string",
                seed=9,
            ),
        ]

        runner = ExperimentRunner(
            constrained_fn=generate_constrained,
            unconstrained_fn=generate_unconstrained,
            analyze_input_fn=analyze_input_fn(),
        )
        results = runner.run(cases)

        ExperimentRunner.save_results(
            results,
            suite_name="unit",
            model_family=spec.family,
            model_key=spec.key,
            run_name="sampled__rep-1.1__seeds-10",
            metadata={
                "model_id": spec.model_id,
                "do_sample": True,
                "repetition_penalty": 1.1,
                "renormalize_logits": True,
                "max_new_tokens": spec.max_new_tokens,
            },
        )
    except torch.cuda.OutOfMemoryError:
        print(f"[SKIP] {spec.display_name}: CUDA OOM while loading")
        release_cuda_memory()
        return


if __name__ == "__main__":
    main()
