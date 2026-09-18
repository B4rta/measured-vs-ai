"""Split conformal intervals with one maximum-residual score per project."""
import math
import numpy as np


def project_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=float)
    if not 0 < alpha < 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Finite calibration scores and 0 < alpha < 1 required")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    # Never substitute the largest observed score for an unattainable rank.
    return float(np.sort(scores)[rank - 1]) if rank <= len(scores) else float("inf")
