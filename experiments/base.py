import torch
import os
import sys
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional
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


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    case: ExperimentCase

    constrained_output_raw: str
    constrained_code: str
    constrained_parse_ok: Optional[bool]
    constrained_syntax_errors: Optional[int]
    constrained_parse_errors: Optional[int]
    constrained_semantic_errors: Optional[int]
    constrained_eval_ok: Optional[bool] = None
    constrained_eval_output: Optional[str] = None

    unconstrained_output_raw: Optional[str] = None
    unconstrained_code: Optional[str] = None
    unconstrained_parse_ok: Optional[bool] = None
    unconstrained_syntax_errors: Optional[int] = None
    unconstrained_parse_errors: Optional[int] = None
    unconstrained_semantic_errors: Optional[int] = None
    unconstrained_eval_ok: Optional[bool] = None
    unconstrained_eval_output: Optional[str] = None


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
            constrained_raw = self._constrained_fn(c)
            constrained_code = extract_code(constrained_raw)
            constrained_input_check = (
                self._analyze_input_fn(constrained_code)
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
                self._evaluate_input_fn(constrained_code)
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

            unconstrained_raw = (
                self._unconstrained_fn(c) if self._unconstrained_fn else None
            )
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

            results.append(
                ExperimentResult(
                    case=c,
                    # constrained section
                    constrained_output_raw=constrained_raw,
                    constrained_code=constrained_code,
                    constrained_parse_ok=constrained_ok,
                    constrained_syntax_errors=constrained_syntax_errors,
                    constrained_parse_errors=constrained_parse_errors,
                    constrained_semantic_errors=constrained_semantic_errors,
                    constrained_eval_ok=constrained_eval_ok,
                    constrained_eval_output=constrained_eval_output,
                    # unconstrained section
                    unconstrained_output_raw=unconstrained_raw,
                    unconstrained_code=unconstrained_code,
                    unconstrained_parse_ok=unconstrained_ok,
                    unconstrained_syntax_errors=unconstrained_syntax_errors,
                    unconstrained_parse_errors=unconstrained_parse_errors,
                    unconstrained_semantic_errors=unconstrained_semantic_errors,
                    unconstrained_eval_ok=unconstrained_eval_ok,
                    unconstrained_eval_output=unconstrained_eval_output,
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

        def _avg(values: list[Optional[int]]) -> str:
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
            code: Optional[str],
            eval_ok: Optional[bool],
            eval_output: Optional[str],
        ) -> None:
            print(f"[{title} PARSE OK] {parse_ok}")
            if syntax_errors is not None:
                print(f"[{title} SYNTAX ERRORS] {syntax_errors}")
            if parse_errors is not None:
                print(f"[{title} PARSE ERRORS] {parse_errors}")
            if semantic_errors is not None:
                print(f"[{title} SEMANTIC ERRORS] {semantic_errors}")

            if eval_ok is not None:
                print(f"[{title} EVAL OK] {eval_ok}")
            if eval_output:
                print(f"[{title} EVAL OUTPUT]")
                print(eval_output.rstrip())
                print("-" * 100)

            if output_raw is not None:
                print(f"[{title} OUTPUT RAW]")
                print(output_raw.rstrip())
                print("-" * 100)

            if code is not None:
                print(f"[{title} CODE EXTRACTED]")
                print(code.rstrip())
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
                code=r.unconstrained_code,
                eval_ok=getattr(r, "unconstrained_eval_ok", None),
                eval_output=getattr(r, "unconstrained_eval_output", None),
            )

            _print_variant(
                "CONSTRAINED",
                parse_ok=r.constrained_parse_ok,
                syntax_errors=r.constrained_syntax_errors,
                parse_errors=r.constrained_parse_errors,
                semantic_errors=r.constrained_semantic_errors,
                output_raw=r.constrained_output_raw,
                code=r.constrained_code,
                eval_ok=getattr(r, "constrained_eval_ok", None),
                eval_output=getattr(r, "constrained_eval_output", None),
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
            getattr(r, "unconstrained_eval_ok", None)
            for r in results
            if r.unconstrained_output_raw is not None
        ]
        cons_eval = [getattr(r, "constrained_eval_ok", None) for r in results]

        print("=" * 100)
        print("[SUMMARY]")
        print(f"Unconstrained parse success:    {_rate(uncon_ok)}")
        print(f"Unconstrained avg syntax errs: {_avg(uncon_syntax)}")
        print(f"Unconstrained avg parse errs:  {_avg(uncon_parse)}")
        print(f"Unconstrained avg sem errs:    {_avg(uncon_sem)}")
        if any(v is not None for v in uncon_eval):
            print(f"Unconstrained eval success:    {_rate(uncon_eval)}")
        print("-" * 100)
        print(f"Constrained   parse success:   {_rate(cons_ok)}")
        print(f"Constrained   avg syntax errs: {_avg(cons_syntax)}")
        print(f"Constrained   avg parse errs:  {_avg(cons_parse)}")
        print(f"Constrained   avg sem errs:    {_avg(cons_sem)}")
        if any(v is not None for v in cons_eval):
            print(f"Constrained   eval success:    {_rate(cons_eval)}")
        print("=" * 100)
