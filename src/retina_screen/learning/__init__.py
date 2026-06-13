"""Production data contracts + metrics for the screening learning loop.

The *real* (production-shaped) model layer: the contracts a deployment populates
to close the loop — prediction -> grader label -> intervention -> outcome — plus
the eval / QI / active-learning computations over them.

This is part of an original reference implementation that carries **no real
PHI**; these are real *contracts*, not real *data*. See `dto.py` for the PHI
handling notes.
"""

__all__: list[str] = []
