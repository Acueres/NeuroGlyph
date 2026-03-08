import torch
import os
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


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
    constrained_output: str
    unconstrained_output: Optional[str] = None


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
    ) -> None:
        self._constrained_fn = constrained_fn
        self._unconstrained_fn = unconstrained_fn

    def run(self, cases: Iterable[ExperimentCase]) -> list[ExperimentResult]:
        results: list[ExperimentResult] = []
        for c in cases:
            constrained = self._constrained_fn(c)
            unconstrained = (
                self._unconstrained_fn(c) if self._unconstrained_fn else None
            )
            results.append(
                ExperimentResult(
                    case=c,
                    constrained_output=constrained,
                    unconstrained_output=unconstrained,
                )
            )
        return results

    @staticmethod
    def print_results(results: Iterable[ExperimentResult]) -> None:
        for r in results:
            print("=" * 100)
            print(f"[CASE] {r.case.name} | seed={r.case.seed}")
            print("-" * 100)
            print("[TASK]")
            print(r.case.task.rstrip())
            print("-" * 100)

            if r.unconstrained_output is not None:
                print("[UNCONSTRAINED OUTPUT]")
                print(r.unconstrained_output.rstrip())
                print("-" * 100)

            print("[CONSTRAINED OUTPUT]")
            print(r.constrained_output.rstrip())
            print()
