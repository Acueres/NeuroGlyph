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
    from compiler_client.fetchers import analyze_input, evaluate_input
    from compiler_client.responses import AnalyzeInputResponse, EvaluateInputResponse
    from models.registry import get_model, can_attempt_load, release_cuda_memory

    def analyze_input_fn() -> Callable[[str], AnalyzeInputResponse]:
        def wrapper(text: str):
            return analyze_input("localhost:7162", text, root_cert_pem="./cert.pem")

        return wrapper

    def evaluate_input_fn() -> Callable[[str], EvaluateInputResponse]:
        def wrapper(text: str):
            return evaluate_input("localhost:7162", text, root_cert_pem="./cert.pem")

        return wrapper

    spec = get_model("gemma3-27b-it-4bit")
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

        cases = [
            # Type drift / foreign type priors
            ExperimentCase(
                name="adv_float_wording",
                task=(
                    "Write a function named `add` that works with float values, and a `main` function "
                    "that prints the result of calling it."
                ),
                seed=1,
            ),
            ExperimentCase(
                name="adv_double_precision",
                task=(
                    "Write a function that adds two double-precision numbers, and a `main` function "
                    "that prints the result."
                ),
                seed=2,
            ),
            ExperimentCase(
                name="adv_string_wording",
                task=(
                    "Write a function that takes a string and returns a string, and a `main` function "
                    "that prints the returned value."
                ),
                seed=3,
            ),
            # Comment pressure
            ExperimentCase(
                name="adv_comment_every_line",
                task=(
                    "Write a function named `add` that adds two integers, and a `main` function that prints "
                    "the result of calling it. Add comments throughout the code."
                ),
                seed=4,
            ),
            ExperimentCase(
                name="adv_comment_heavy_loop",
                task=(
                    "Write a function that sums the integers from 1 to 10 using a while loop. "
                    "Add comments before the statements."
                ),
                seed=5,
            ),
            # Unsupported or foreign constructs
            ExperimentCase(
                name="adv_range_function",
                task=(
                    "Write a function that iterates over `range(0, 11)` and prints each value."
                ),
                seed=6,
            ),
            ExperimentCase(
                name="adv_fn_main",
                task=(
                    "Write an `add` function and an `fn main()` entry point that prints the result of calling it."
                ),
                seed=7,
            ),
            ExperimentCase(
                name="adv_generics",
                task=(
                    "Write a generic identity function and a `main` function that uses it with an integer."
                ),
                seed=8,
            ),
            ExperimentCase(
                name="adv_function_pointers",
                task=(
                    "Write a function that takes a function pointer and applies it to an integer."
                ),
                seed=9,
            ),
            ExperimentCase(
                name="adv_python_blocks",
                task=(
                    "Write a function using Python-style indentation instead of braces, with an if/else "
                    "and a return statement."
                ),
                seed=10,
            ),
        ]

        full_cases: list[ExperimentCase] = []

        for case in cases:
            full_cases.append(case)
            for seed_offset in (42, 100):
                full_cases.append(
                    ExperimentCase(
                        name=case.name,
                        task=case.task,
                        seed=case.seed + seed_offset,
                        expected_output=case.expected_output,
                    )
                )

        runner = ExperimentRunner(
            constrained_fn=generate_constrained,
            unconstrained_fn=generate_unconstrained,
            analyze_input_fn=analyze_input_fn(),
            evaluate_input_fn=evaluate_input_fn(),
        )

        results = runner.run(full_cases)

        ExperimentRunner.print_results(results)
        ExperimentRunner.save_results(
            results,
            suite_name="adversarial",
            model_family=spec.family,
            model_key=spec.key,
            run_name="sampled__seeds-10",
            metadata={
                "model_id": spec.model_id,
                "do_sample": True,
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
