import torch
import os
import sys
import re
import json
import time

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence
from src.compiler_client.responses import AnalyzeInputResponse, EvaluateInputResponse

_FENCE_RE = re.compile(r"(?s)(```|~~~)(?:[^\n]*)\n(.*?)\1")


def extract_code(text: str) -> str:
    """
    Extracts code from a markdown-fenced block if present. Otherwise returns the trimmed text.
    """
    if not text:
        return ""
    matches = _FENCE_RE.findall(text)
    if not matches:
        return text.strip()

    best = ""
    for _, body in matches:
        candidate = (body or "").strip()
        if len(candidate) > len(best):
            best = candidate

    return best if best else text.strip()


def bootstrap_repo_root() -> Path:
    """
    Ensures `src/` is importable even if the package isn't installed in editable mode.
    Also sets CWD to repo root so relative paths like ./cert.pem keep working.
    """
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(root)
    return root


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True, slots=True)
class ExperimentCase:
    name: str
    task: str
    seed: int = 1
    expected_output: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    case: ExperimentCase

    constrained_output: str
    constrained_parse_ok: Optional[bool]
    constrained_syntax_errors: Optional[int]
    constrained_parse_errors: Optional[int]
    constrained_semantic_errors: Optional[int]
    constrained_eval_ok: Optional[bool] = None
    constrained_eval_output: Optional[str] = None
    constrained_output_correct: Optional[bool] = None
    constrained_end_to_end_correct: Optional[bool] = None
    constrained_runtime_s: float = 0

    unconstrained_output_raw: Optional[str] = None
    unconstrained_code: Optional[str] = None
    unconstrained_parse_ok: Optional[bool] = None
    unconstrained_syntax_errors: Optional[int] = None
    unconstrained_parse_errors: Optional[int] = None
    unconstrained_semantic_errors: Optional[int] = None
    unconstrained_eval_ok: Optional[bool] = None
    unconstrained_eval_output: Optional[str] = None
    unconstrained_output_correct: Optional[bool] = None
    unconstrained_end_to_end_correct: Optional[bool] = None
    unconstrained_runtime_s: float = 0


class ExperimentRunner:
    """
    Base experiment runner:
      - runs a list of ExperimentCase
      - optionally runs both constrained and unconstrained generation
      - prints results
    """

    def __init__(
        self,
        *,
        constrained_fn: Callable[[ExperimentCase], str],
        unconstrained_fn: Optional[Callable[[ExperimentCase], str]] = None,
        analyze_input_fn: Optional[Callable[[str], AnalyzeInputResponse]] = None,
        evaluate_input_fn: Optional[Callable[[str], EvaluateInputResponse]] = None,
    ) -> None:
        self._constrained_fn = constrained_fn
        self._unconstrained_fn = unconstrained_fn
        self._analyze_input_fn = analyze_input_fn
        self._evaluate_input_fn = evaluate_input_fn

    def run(self, cases: Iterable[ExperimentCase]) -> list[ExperimentResult]:
        results: list[ExperimentResult] = []

        for c in cases:
            # Constrained mode
            cons_start = time.perf_counter()
            constrained_output = self._constrained_fn(c)
            cons_elapsed_s = time.perf_counter() - cons_start

            constrained_input_check = (
                self._analyze_input_fn(constrained_output)
                if self._analyze_input_fn
                else None
            )
            constrained_ok = (
                constrained_input_check.ok if constrained_input_check else None
            )
            constrained_syntax_errors = (
                constrained_input_check.syntax_errors_number
                if constrained_input_check
                else None
            )
            constrained_parse_errors = (
                constrained_input_check.parse_errors_number
                if constrained_input_check
                else None
            )
            constrained_semantic_errors = (
                constrained_input_check.semantic_errors_number
                if constrained_input_check
                else None
            )
            constrained_input_evaluation = (
                self._evaluate_input_fn(constrained_output)
                if self._evaluate_input_fn
                else None
            )
            constrained_eval_ok = (
                constrained_input_evaluation.ok
                if constrained_input_evaluation
                else None
            )
            constrained_eval_output = (
                constrained_input_evaluation.output
                if constrained_input_evaluation
                else None
            )
            constrained_output_correct = (
                c.expected_output.strip() == constrained_eval_output.strip()
                if c.expected_output and constrained_eval_output
                else None
            )
            constrained_end_to_end_correct = (
                bool(constrained_eval_ok and constrained_output_correct is True)
                if c.expected_output is not None and constrained_eval_ok is not None
                else None
            )

            # Unconstrained mode
            uncon_start = time.perf_counter()
            unconstrained_raw = (
                self._unconstrained_fn(c) if self._unconstrained_fn else None
            )
            uncon_elapsed_s = time.perf_counter() - uncon_start

            unconstrained_code = (
                extract_code(unconstrained_raw)
                if unconstrained_raw is not None
                else None
            )
            unconstrained_input_check = (
                self._analyze_input_fn(unconstrained_code)
                if unconstrained_code and self._analyze_input_fn
                else None
            )
            unconstrained_ok = (
                unconstrained_input_check.ok if unconstrained_input_check else None
            )
            unconstrained_syntax_errors = (
                unconstrained_input_check.syntax_errors_number
                if unconstrained_input_check
                else None
            )
            unconstrained_parse_errors = (
                unconstrained_input_check.parse_errors_number
                if unconstrained_input_check
                else None
            )
            unconstrained_semantic_errors = (
                unconstrained_input_check.semantic_errors_number
                if unconstrained_input_check
                else None
            )
            unconstrained_input_evaluation = (
                self._evaluate_input_fn(unconstrained_code)
                if unconstrained_code and self._evaluate_input_fn
                else None
            )
            unconstrained_eval_ok = (
                unconstrained_input_evaluation.ok
                if unconstrained_input_evaluation
                else None
            )
            unconstrained_eval_output = (
                unconstrained_input_evaluation.output
                if unconstrained_input_evaluation
                else None
            )
            unconstrained_output_correct = (
                c.expected_output.strip() == unconstrained_eval_output.strip()
                if c.expected_output and unconstrained_eval_output
                else None
            )
            unconstrained_end_to_end_correct = (
                bool(unconstrained_eval_ok and unconstrained_output_correct is True)
                if c.expected_output is not None and unconstrained_eval_ok is not None
                else None
            )

            results.append(
                ExperimentResult(
                    case=c,
                    # constrained section
                    constrained_output=constrained_output,
                    constrained_parse_ok=constrained_ok,
                    constrained_syntax_errors=constrained_syntax_errors,
                    constrained_parse_errors=constrained_parse_errors,
                    constrained_semantic_errors=constrained_semantic_errors,
                    constrained_eval_ok=constrained_eval_ok,
                    constrained_eval_output=constrained_eval_output,
                    constrained_output_correct=constrained_output_correct,
                    constrained_end_to_end_correct=constrained_end_to_end_correct,
                    constrained_runtime_s=cons_elapsed_s,
                    # unconstrained section
                    unconstrained_output_raw=unconstrained_raw,
                    unconstrained_code=unconstrained_code,
                    unconstrained_parse_ok=unconstrained_ok,
                    unconstrained_syntax_errors=unconstrained_syntax_errors,
                    unconstrained_parse_errors=unconstrained_parse_errors,
                    unconstrained_semantic_errors=unconstrained_semantic_errors,
                    unconstrained_eval_ok=unconstrained_eval_ok,
                    unconstrained_eval_output=unconstrained_eval_output,
                    unconstrained_output_correct=unconstrained_output_correct,
                    unconstrained_end_to_end_correct=unconstrained_end_to_end_correct,
                    unconstrained_runtime_s=uncon_elapsed_s,
                )
            )

        return results

    @staticmethod
    def print_results(results: Iterable[ExperimentResult]) -> None:
        results = list(results)

        def _rate(values: list[Optional[bool]]) -> str:
            vals = [v for v in values if v is not None]
            if not vals:
                return "n/a"
            ok = sum(1 for v in vals if v)
            return f"{ok}/{len(vals)} = {ok / len(vals):.1%}"

        def _avg(values: Sequence[Optional[int | float]]) -> str:
            vals = [v for v in values if v is not None]
            if not vals:
                return "n/a"
            return f"{sum(vals) / len(vals):.2f}"

        def _print_variant(
            title: str,
            *,
            parse_ok: Optional[bool],
            syntax_errors: Optional[int],
            parse_errors: Optional[int],
            semantic_errors: Optional[int],
            output_raw: Optional[str],
            extracted: Optional[str],
            eval_ok: Optional[bool],
            eval_output: Optional[str],
            eval_output_correct: Optional[bool],
            end_to_end_correct: Optional[bool],
            runtime_s: float,
        ) -> None:
            print(f"[{title} EXECUTED IN] {runtime_s: .2f} s")
            print(f"[{title} PARSE OK] {parse_ok}")
            if syntax_errors is not None:
                print(f"[{title} SYNTAX ERRORS] {syntax_errors}")
            if parse_errors is not None:
                print(f"[{title} PARSE ERRORS] {parse_errors}")
            if semantic_errors is not None:
                print(f"[{title} SEMANTIC ERRORS] {semantic_errors}")

            if eval_ok is not None:
                print(f"[{title} EVAL OK] {eval_ok}")
            if eval_output_correct is not None:
                print(
                    f"[{title} EVAL {'CORRECT' if eval_output_correct else 'INCORRECT'}]"
                )
            if end_to_end_correct is not None:
                print(
                    f"[{title} END-TO-END {'CORRECT' if end_to_end_correct else 'INCORRECT'}]"
                )
            if eval_output:
                print(f"[{title} EVAL OUTPUT]")
                print(eval_output.rstrip())
                print("-" * 100)

            if output_raw is not None:
                print(f"[{title} OUTPUT]")
                print(output_raw.rstrip())
                print("-" * 100)

            if extracted is not None:
                print(f"[{title} CODE EXTRACTED]")
                print(extracted.rstrip())
                print("-" * 100)

        # Per-case output
        for r in results:
            print("=" * 100)
            print(f"[CASE] {r.case.name} | seed={r.case.seed}")
            print("-" * 100)
            print("[TASK]")
            print(r.case.task.rstrip())
            print("-" * 100)

            _print_variant(
                "UNCONSTRAINED",
                parse_ok=r.unconstrained_parse_ok,
                syntax_errors=r.unconstrained_syntax_errors,
                parse_errors=r.unconstrained_parse_errors,
                semantic_errors=r.unconstrained_semantic_errors,
                output_raw=r.unconstrained_output_raw,
                extracted=r.unconstrained_code,
                eval_ok=r.unconstrained_eval_ok,
                eval_output=r.unconstrained_eval_output,
                eval_output_correct=r.unconstrained_output_correct,
                end_to_end_correct=r.unconstrained_end_to_end_correct,
                runtime_s=r.unconstrained_runtime_s,
            )

            _print_variant(
                "CONSTRAINED",
                parse_ok=r.constrained_parse_ok,
                syntax_errors=r.constrained_syntax_errors,
                parse_errors=r.constrained_parse_errors,
                semantic_errors=r.constrained_semantic_errors,
                output_raw=r.constrained_output,
                extracted=None,
                eval_ok=r.constrained_eval_ok,
                eval_output=r.constrained_eval_output,
                eval_output_correct=r.constrained_output_correct,
                end_to_end_correct=r.constrained_end_to_end_correct,
                runtime_s=r.constrained_runtime_s,
            )

            print()

        # Summary
        uncon_ok = [
            r.unconstrained_parse_ok
            for r in results
            if r.unconstrained_output_raw is not None
        ]
        cons_ok = [r.constrained_parse_ok for r in results]

        uncon_syntax = [
            r.unconstrained_syntax_errors
            for r in results
            if r.unconstrained_output_raw is not None
        ]
        uncon_parse = [
            r.unconstrained_parse_errors
            for r in results
            if r.unconstrained_output_raw is not None
        ]
        uncon_sem = [
            r.unconstrained_semantic_errors
            for r in results
            if r.unconstrained_output_raw is not None
        ]

        cons_syntax = [r.constrained_syntax_errors for r in results]
        cons_parse = [r.constrained_parse_errors for r in results]
        cons_sem = [r.constrained_semantic_errors for r in results]

        uncon_eval = [
            r.unconstrained_eval_ok
            for r in results
            if r.unconstrained_output_raw is not None
        ]
        cons_eval = [r.constrained_eval_ok for r in results]

        uncon_eval_correct = [
            r.unconstrained_output_correct
            for r in results
            if r.unconstrained_output_raw is not None
        ]
        cons_eval_correct = [r.constrained_output_correct for r in results]

        uncon_end_to_end = [
            r.unconstrained_end_to_end_correct
            for r in results
            if r.unconstrained_output_raw is not None
        ]
        cons_end_to_end = [r.constrained_end_to_end_correct for r in results]

        uncon_runtimes = [r.unconstrained_runtime_s for r in results
            if r.unconstrained_output_raw is not None]
        cons_runtimes = [r.constrained_runtime_s for r in results]

        print("=" * 100)
        print("[SUMMARY]")
        print(f"Unconstrained average runtime s:   {_avg(uncon_runtimes)}")
        print(f"Unconstrained parse success:    {_rate(uncon_ok)}")
        print(f"Unconstrained avg syntax errs: {_avg(uncon_syntax)}")
        print(f"Unconstrained avg parse errs:  {_avg(uncon_parse)}")
        print(f"Unconstrained avg sem errs:    {_avg(uncon_sem)}")
        if any(v is not None for v in uncon_eval):
            print(f"Unconstrained eval success:    {_rate(uncon_eval)}")
        if any(v is not None for v in uncon_eval_correct):
            print(f"Unconstrained output correctness:    {_rate(uncon_eval_correct)}")
        if any(v is not None for v in uncon_end_to_end):
            print(f"Unconstrained end-to-end correctness: {_rate(uncon_end_to_end)}")
        print("-" * 100)
        print(f"Unconstrained average runtime s:   {_avg(cons_runtimes)}")
        print(f"Constrained   parse success:   {_rate(cons_ok)}")
        print(f"Constrained   avg syntax errs: {_avg(cons_syntax)}")
        print(f"Constrained   avg parse errs:  {_avg(cons_parse)}")
        print(f"Constrained   avg sem errs:    {_avg(cons_sem)}")
        if any(v is not None for v in cons_eval):
            print(f"Constrained   eval success:    {_rate(cons_eval)}")
        if any(v is not None for v in cons_eval_correct):
            print(f"Constrained   output correctness:    {_rate(cons_eval_correct)}")
        if any(v is not None for v in cons_end_to_end):
            print(f"Constrained   end-to-end correctness: {_rate(cons_end_to_end)}")

        print("=" * 100)

    @staticmethod
    def _rate_fraction(values: list[Optional[bool]]) -> dict[str, float | int | None]:
        vals = [v for v in values if v is not None]
        if not vals:
            return {"ok": None, "total": 0, "rate": None}
        ok = sum(1 for v in vals if v)
        total = len(vals)
        return {"ok": ok, "total": total, "rate": ok / total}

    @staticmethod
    def _avg_value(values: list[Optional[int]]) -> float | None:
        vals = [v for v in values if v is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    @classmethod
    def build_summary(cls, results: Iterable[ExperimentResult]) -> dict:
        results = list(results)

        uncon_ok = [
            r.unconstrained_parse_ok
            for r in results
            if r.unconstrained_output_raw is not None
        ]
        cons_ok = [r.constrained_parse_ok for r in results]

        uncon_syntax = [
            r.unconstrained_syntax_errors
            for r in results
            if r.unconstrained_output_raw is not None
        ]
        uncon_parse = [
            r.unconstrained_parse_errors
            for r in results
            if r.unconstrained_output_raw is not None
        ]
        uncon_sem = [
            r.unconstrained_semantic_errors
            for r in results
            if r.unconstrained_output_raw is not None
        ]

        cons_syntax = [r.constrained_syntax_errors for r in results]
        cons_parse = [r.constrained_parse_errors for r in results]
        cons_sem = [r.constrained_semantic_errors for r in results]

        uncon_eval = [
            r.unconstrained_eval_ok
            for r in results
            if r.unconstrained_output_raw is not None
        ]
        cons_eval = [r.constrained_eval_ok for r in results]

        uncon_eval_correct = [
            r.unconstrained_output_correct
            for r in results
            if r.unconstrained_output_raw is not None
        ]
        cons_eval_correct = [r.constrained_output_correct for r in results]

        uncon_end_to_end = [
            r.unconstrained_end_to_end_correct
            for r in results
            if r.unconstrained_output_raw is not None
        ]
        cons_end_to_end = [r.constrained_end_to_end_correct for r in results]

        return {
            "cases_total": len(results),
            "unconstrained": {
                "parse_success": cls._rate_fraction(uncon_ok),
                "avg_syntax_errors": cls._avg_value(uncon_syntax),
                "avg_parse_errors": cls._avg_value(uncon_parse),
                "avg_semantic_errors": cls._avg_value(uncon_sem),
                "eval_success": cls._rate_fraction(uncon_eval),
                "eval_correctness": cls._rate_fraction(uncon_eval_correct),
                "end_to_end_correctness": cls._rate_fraction(uncon_end_to_end),
            },
            "constrained": {
                "parse_success": cls._rate_fraction(cons_ok),
                "avg_syntax_errors": cls._avg_value(cons_syntax),
                "avg_parse_errors": cls._avg_value(cons_parse),
                "avg_semantic_errors": cls._avg_value(cons_sem),
                "eval_success": cls._rate_fraction(cons_eval),
                "eval_correctness": cls._rate_fraction(cons_eval_correct),
                "end_to_end_correctness": cls._rate_fraction(cons_end_to_end),
            },
        }

    @staticmethod
    def save_results(
        results: Iterable[ExperimentResult],
        *,
        suite_name: str,
        model_family: str,
        model_key: str,
        run_name: str,
        metadata: Optional[dict] = None,
        out_root: str | Path = "experiments/out",
    ) -> Path:
        """
        Saves:
          - cases__<run_name>.jsonl
          - summary__<run_name>.json

        Example path:
          experiments/out/unit/gemma3/gemma3-12b-it-4bit/
        """
        results = list(results)

        out_dir = Path(out_root) / suite_name / model_family / model_key
        out_dir.mkdir(parents=True, exist_ok=True)

        cases_path = out_dir / f"cases__{run_name}.jsonl"
        summary_path = out_dir / f"summary__{run_name}.json"

        # Save per-case results as JSONL
        with cases_path.open("w", encoding="utf-8") as f:
            for r in results:
                row = asdict(r)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Save aggregated summary as JSON
        summary = ExperimentRunner.build_summary(results)
        payload = {
            "suite_name": suite_name,
            "model_family": model_family,
            "model_key": model_key,
            "run_name": run_name,
            "metadata": metadata or {},
            "summary": summary,
            "files": {
                "cases": str(cases_path),
                "summary": str(summary_path),
            },
        }

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return out_dir
