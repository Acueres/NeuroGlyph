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

    def generate_constrained(case: ExperimentCase) -> str:
        set_torch_seed(case.seed)
        return gen.generate(case.task)

    def analyze_input_fn() -> Callable[[str], AnalyzeInputResponse]:
        def wrapper(text: str):
            return analyze_input("localhost:7162", text, root_cert_pem="./cert.pem")

        return wrapper

    def evaluate_input_fn() -> Callable[[str], EvaluateInputResponse]:
        def wrapper(text: str):
            return evaluate_input("localhost:7162", text, root_cert_pem="./cert.pem")

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

        cases = [
            ExperimentCase(
                name="task_add_main_int",
                task=(
                    "Write a function named `add` that adds two integers. "
                    "Also write `main` that prints the result of calling it with 1 and 2."
                ),
                seed=1,
                expected_output="3",
            ),
            ExperimentCase(
                name="task_add_main_real",
                task=(
                    "Write a function named `add` that adds two real numbers. "
                    "Also write `main` that prints the result of calling it with 1.0 and 2.0."
                ),
                seed=2,
                expected_output="3",
            ),
            ExperimentCase(
                name="task_sum_to_ten_while",
                task=(
                    "Write a function named `sum_to_ten` that sums the integers from 1 to 10 using a while loop. "
                    "Also write `main` that prints its result."
                ),
                seed=3,
                expected_output="55",
            ),
            ExperimentCase(
                name="task_print_range_for",
                task=(
                    "Write a function named `print_numbers` that prints the integers from 0 to 10 inclusive, "
                    "one per line. Also write `main` that calls it."
                ),
                seed=4,
                expected_output="0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10",
            ),
            ExperimentCase(
                name="task_max2_if_else",
                task=(
                    "Write a function named `max2` that returns the larger of two integers using if/else. "
                    "Also write `main` that prints the result of calling it with 2 and 5."
                ),
                seed=5,
                expected_output="5",
            ),
            ExperimentCase(
                name="task_point_init",
                task=(
                    "Define a struct named `Point` with integer fields `a` and `b`. "
                    "Also write `main` that creates a point with values 3 and 2 and prints the sum of its fields."
                ),
                seed=6,
                expected_output="5",
            ),
            ExperimentCase(
                name="task_cast_print",
                task=(
                    "Write `main` that creates a real value 5.0, converts it to an integer, stores it in `y`, "
                    "and prints `y`."
                ),
                seed=7,
                expected_output="5",
            ),
            ExperimentCase(
                name="task_two_add_calls",
                task=(
                    "Write a function named `add` that adds two integers. "
                    "Also write `main` that prints the results of calling it with 1 and 2, and with 3 and 4."
                ),
                seed=8,
                expected_output="3\n7",
            ),
        ]

        runner = ExperimentRunner(
            constrained_fn=generate_constrained,
            unconstrained_fn=generate_unconstrained,
            analyze_input_fn=analyze_input_fn(),
            evaluate_input_fn=evaluate_input_fn(),
        )

        results = runner.run(cases)
        ExperimentRunner.print_results(results)
        ExperimentRunner.save_results(
            results,
            suite_name="tasks",
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
