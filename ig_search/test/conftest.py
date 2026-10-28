"""Wspólne fabryki terenów syntetycznych do testów (bez ROS)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ig_search.domain.params import SearchParams  # noqa: E402, I100
from ig_search.domain.terrain import Grid, HeightMap  # noqa: E402
import numpy as np  # noqa: E402

RES = 0.1


def make_heightmap(fn, size_x=20.0, size_y=20.0, res=RES, origin=(0.0, 0.0)):
    """Zbuduj mapę wysokości z funkcji h(x, y) na prostokącie [origin, origin + size]."""
    W, H = int(round(size_x / res)), int(round(size_y / res))
    g = Grid(res, origin[0], origin[1], H, W)
    cx, cy = g.centers()
    return HeightMap(np.asarray(fn(cx, cy), float).reshape(H, W), g)


def flat(cx, cy):
    return np.zeros_like(cx)


def ridge(x0=10.0, height=1.0, half_width=0.4):
    """Grzbiet wzdłuż osi y w x = x0 (przekrój trójkątny)."""
    def fn(cx, cy):
        return np.maximum(0.0, height * (1.0 - np.abs(cx - x0) / half_width))
    return fn


def default_params() -> SearchParams:
    return SearchParams()
