"""SYNTHETIC demo of the screening metrics learning loop — no real PHI.

Models the ground-truth labels and outcomes that close the loop
(prediction -> grader label -> intervention -> outcome) and computes the eval +
program (QI) metrics over fabricated data. Nothing here touches real patient
data; the synthetic generators live in `synthetic.py`.
"""

__all__: list[str] = []
