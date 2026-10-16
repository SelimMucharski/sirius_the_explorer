"""Matching of detected probes to ground truth and error statistics. Pure numpy/scipy."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class MatchGate:
    """A detection matches a probe if its centre is within max(min_dist, rel_dist * Z)."""

    min_dist: float = 0.25
    rel_dist: float = 0.25


def match(gt: np.ndarray, det: np.ndarray, gate: MatchGate = MatchGate()) -> List[Tuple[int, int]]:
    """Match probe centres gt (N x 3) and det (M x 3) -> [(gt index, det index)]."""
    if not len(gt) or not len(det):
        return []
    cost = np.linalg.norm(gt[:, None] - det[None], axis=2)
    out = []
    for g, d in zip(*linear_sum_assignment(cost)):
        if cost[g, d] <= max(gate.min_dist, gate.rel_dist * gt[g, 2]):
            out.append((int(g), int(d)))
    return out


def axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two axes [deg]; sign-sensitive (tip -> face is known)."""
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def in_view(p: np.ndarray, fx, fy, cx, cy, width, height, z_min=0.05) -> bool:
    if p[2] <= z_min:
        return False
    u, v = fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy
    return 0 <= u < width and 0 <= v < height


def summarize(values: Sequence[float]) -> str:
    v = np.asarray(values, float)
    if not len(v):
        return '-'
    return f'median {np.median(v):.0f}, p90 {np.percentile(v, 90):.0f}'
