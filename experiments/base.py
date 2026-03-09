import torch
import os
import sys
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional
from src.compiler_client.responses import CheckSyntaxResponse

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

    unconstrained_output_raw: Optional[str] = None
    unconstrained_code: Optional[str] = None
    unconstrained_parse_ok: Optional[bool] = None
    unconstrained_syntax_errors: Optional[int] = None
    unconstrained_parse_errors: Optional[int] = None


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
        parse_check_fn: Optional[Callable[[str], CheckSyntaxResponse]] = None,
    ) -> None:
        self._constrained_fn = constrained_fn
        self._unconstrained_fn = unconstrained_fn
        self._parse_check_fn = parse_check_fn

    def run(self, cases: Iterable[ExperimentCase]) -> list[ExperimentResult]:
        results: list[ExperimentResult] = []

        for c in cases:
            constrained_raw = self._constrained_fn(c)
            constrained_code = extract_code(constrained_raw)
            constrained_syntax_check = (
                self._parse_check_fn(constrained_code) if self._parse_check_fn else None
            )
            constrained_ok = (
                constrained_syntax_check.ok if constrained_syntax_check else None
            )
            constrained_syntax_errors = (
                constrained_syntax_check.syntax_errors_number
                if constrained_syntax_check
                else None
            )
            constrained_parse_errors = (
                constrained_syntax_check.parse_errors_number
                if constrained_syntax_check
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
            unconstrained_syntax_check = (
                self._parse_check_fn(unconstrained_code)
                if unconstrained_code and self._parse_check_fn
                else None
            )
            unconstrained_ok = (
                unconstrained_syntax_check.ok if unconstrained_syntax_check else None
            )
            unconstrained_syntax_errors = (
                unconstrained_syntax_check.syntax_errors_number
                if unconstrained_syntax_check
                else None
            )
            unconstrained_parse_errors = (
                unconstrained_syntax_check.parse_errors_number
                if unconstrained_syntax_check
                else None
            )

            results.append(
                ExperimentResult(
                    case=c,
                    constrained_output_raw=constrained_raw,
                    constrained_code=constrained_code,
                    constrained_parse_ok=constrained_ok,
                    constrained_syntax_errors=constrained_syntax_errors,
                    constrained_parse_errors=constrained_parse_errors,
                    unconstrained_output_raw=unconstrained_raw,
                    unconstrained_code=unconstrained_code,
                    unconstrained_parse_ok=unconstrained_ok,
                    unconstrained_syntax_errors=unconstrained_syntax_errors,
                    unconstrained_parse_errors=unconstrained_parse_errors,
                )
            )

        return results

    @staticmethod
    def print_results(results: Iterable[ExperimentResult]) -> None:
        results = list(results)
        unconstrained_syntax_errors = 0
        unconstrained_parse_errors = 0
        constrained_syntax_errors = 0
        constrained_parse_errors = 0

        for r in results:
            print("=" * 100)
            print(f"[CASE] {r.case.name} | seed={r.case.seed}")
            print("-" * 100)
            print("[TASK]")
            print(r.case.task.rstrip())
            print("-" * 100)

            if r.unconstrained_output_raw is not None:
                print(f"[UNCONSTRAINED PARSE OK] {r.unconstrained_parse_ok}")
                if r.unconstrained_syntax_errors and r.unconstrained_syntax_errors > 0:
                    print(
                        f"[UNCONSTRAINED SYNTAX ERRORS] {r.unconstrained_syntax_errors}"
                    )
                    unconstrained_syntax_errors += r.unconstrained_syntax_errors
                if r.unconstrained_parse_errors and r.unconstrained_parse_errors > 0:
                    print(
                        f"[UNCONSTRAINED PARSE ERRORS] {r.unconstrained_parse_errors}"
                    )
                    unconstrained_parse_errors += r.unconstrained_parse_errors
                print("[UNCONSTRAINED OUTPUT RAW]")
                print(r.unconstrained_output_raw.rstrip())
                print("-" * 100)
                print("[UNCONSTRAINED CODE EXTRACTED]")
                print(r.unconstrained_code.rstrip() if r.unconstrained_code else "")
                print("-" * 100)

            print(f"[CONSTRAINED PARSE OK] {r.constrained_parse_ok}")
            if r.constrained_syntax_errors and r.constrained_syntax_errors > 0:
                print(f"[UNCONSTRAINED SYNTAX ERRORS] {r.constrained_syntax_errors}")
                constrained_syntax_errors += r.constrained_syntax_errors
            if r.constrained_parse_errors and r.constrained_parse_errors > 0:
                print(f"[UNCONSTRAINED PARSE ERRORS] {r.constrained_parse_errors}")
                constrained_parse_errors += r.constrained_parse_errors
            print("-" * 100)
            print("[CONSTRAINED CODE EXTRACTED]")
            print(r.constrained_code.rstrip())
            print()

        # Summary
        def rate(ok_list: list[Optional[bool]]) -> str:
            vals = [x for x in ok_list if x is not None]
            if not vals:
                return "n/a"
            return f"{sum(1 for x in vals if x)}/{len(vals)} = {sum(1 for x in vals if x)/len(vals):.1%}"

        uncon_ok = [
            r.unconstrained_parse_ok
            for r in results
            if r.unconstrained_output_raw is not None
        ]
        cons_ok = [r.constrained_parse_ok for r in results]

        print("=" * 100)
        print("[SUMMARY]")
        print(f"Unconstrained parse success: {rate(uncon_ok)}")
        print(
            f"Unconstrained average syntax errors: {unconstrained_syntax_errors / len(uncon_ok)}"
        )
        print(
            f"Unconstrained average parse errors: {unconstrained_parse_errors / len(uncon_ok)}"
        )
        print(f"Constrained   parse success: {rate(cons_ok)}")
        print(
            f"Constrained   average syntax errors: {constrained_syntax_errors / len(cons_ok)}"
        )
        print(
            f"Constrained   average parse errors: {constrained_parse_errors / len(cons_ok)}"
        )
        print("=" * 100)
