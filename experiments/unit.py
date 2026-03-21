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

        # Syntax prompts
        cases = [
            ExperimentCase(
                name="func_add_int",
                task="Write a function named `add` that takes two integers and returns their sum.",
                seed=1,
            ),
            ExperimentCase(
                name="func_add_real",
                task="Write a function named `add` that takes two real numbers and returns their sum.",
                seed=1,
            ),
            ExperimentCase(
                name="let_decl",
                task="Write a function that contains a local integer variable named `sum` initialized to `1 + 2`.",
                seed=2,
            ),
            ExperimentCase(
                name="return_stmt",
                task="Write a function named `id` that takes an integer and returns it unchanged.",
                seed=3,
            ),
            ExperimentCase(
                name="if_else",
                task="Write a function named `max2` that returns the larger of two integers.",
                seed=4,
            ),
            ExperimentCase(
                name="while_loop",
                task="Write a function named `sum_to_ten` that returns the sum of the integers from 1 to 10 using a while loop.",
                seed=5,
            ),
            ExperimentCase(
                name="for_range",
                task="Write a function named `print_numbers` that prints the integers from 0 to 10, one per line.",
                seed=6,
            ),
            ExperimentCase(
                name="initializer_new",
                task="Define a struct named `Point` with integer fields `a` and `b`, and write a snippet that creates a `Point` value with `a = 1` and `b = 2`.",
                seed=7,
            ),
            ExperimentCase(
                name="cast_as",
                task="Write a snippet that defines a real value, converts it to an integer, and stores the result in a variable named `y`.",
                seed=8,
            ),
            ExperimentCase(
                name="two_funcs_main",
                task="Write a function named `add` that adds two integers, and a `main` function that prints the result of calling it with 1 and 2.",
                seed=9,
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
        )
        results = runner.run(full_cases)

        ExperimentRunner.print_results(results)
        ExperimentRunner.save_results(
            results,
            suite_name="unit",
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
