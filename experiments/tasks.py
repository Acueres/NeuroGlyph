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
                    "Generate two functions: "
                    "`add(x: int, y: int) -> int` returning the sum, and "
                    "`main()` that calls add with 1 and 2 and prints the result using "
                    "`println(val as str)`."
                ),
                seed=1,
            ),
            ExperimentCase(
                name="task_add_main_real",
                task=(
                    "Generate two functions: "
                    "`add(x: real, y: real) -> real` returning the sum, and "
                    "`main()` that calls add with 1.0 and 2.0 and prints the result using "
                    "`println(val as str)`."
                ),
                seed=2,
            ),
            ExperimentCase(
                name="task_sum_to_ten_while",
                task=(
                    "Generate a function `sum_to_ten() -> int` that uses a while-loop to sum "
                    "integers from 1 to 10 and returns the sum."
                ),
                seed=3,
            ),
            ExperimentCase(
                name="task_print_range_for",
                task=(
                    "Generate a function `print_numbers()` that iterates from 0 to 10 inclusive "
                    "and calls `println(i as str)` for each value."
                ),
                seed=4,
            ),
            ExperimentCase(
                name="task_max2_if_else",
                task=(
                    "Generate a function `max2(a: int, b: int) -> int` using if/else and return."
                ),
                seed=5,
            ),
            ExperimentCase(
                name="task_point_init",
                task=(
                    "Generate a snippet that creates a `Point` object with values 1 and 2 and "
                    "stores it in variable `point`."
                ),
                seed=6,
            ),
            ExperimentCase(
                name="task_cast_print",
                task=(
                    "Generate a snippet that casts `x` to type `int` using `as`, stores it in "
                    "`y`, and prints `y as str`."
                ),
                seed=7,
            ),
            ExperimentCase(
                name="task_two_add_calls",
                task=(
                    "Generate a `main()` function that calls `add(1, 2)` and `add(3, 4)` and "
                    "prints both results using `println(val as str)`."
                ),
                seed=8,
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
    except torch.cuda.OutOfMemoryError:
        print(f"[SKIP] {spec.display_name}: CUDA OOM while loading")
        release_cuda_memory()
        return


if __name__ == "__main__":
    main()
