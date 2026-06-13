"""SYNTHETIC demo: compute and print the program (QI) + eval metrics — no real PHI.

Run as `python -m retina_screen.learning.demo`. Builds fabricated labeled
encounters, then prints the program metrics and model-vs-grader evaluation as
JSON. Demonstrates the learning loop end-to-end with zero real data.
"""

from retina_screen.learning.dto import EvalMetrics, ProgramMetrics
from retina_screen.learning.metrics import evaluate, program_metrics
from retina_screen.learning.synthetic import make_synthetic_labeled_encounters

__all__ = ["main", "run_demo"]


def run_demo(n: int = 200) -> tuple[ProgramMetrics, EvalMetrics]:
    """Build synthetic data and return (program metrics, eval metrics)."""
    labeled = make_synthetic_labeled_encounters(n)
    return program_metrics(labeled), evaluate(labeled)


def main() -> None:
    """Entry point: print the synthetic program + eval metrics as JSON."""
    program, evaluation = run_demo()
    print(program.model_dump_json(indent=2))
    print(evaluation.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
