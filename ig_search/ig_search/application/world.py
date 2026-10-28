"""Statyczne modele planera zbudowane raz na misję: teren, widoczność, ruch, pozy obserwacji."""
from dataclasses import dataclass
import math

from ig_search.domain.belief import Belief
from ig_search.domain.detector_model import DetectorModel
from ig_search.domain.params import SearchParams
from ig_search.domain.terrain import Grid, HeightMap
from ig_search.domain.travel import TravelModel
from ig_search.domain.traversability import TraversabilityMap
from ig_search.domain.views import ViewSet
from ig_search.domain.visibility import Visibility
import numpy as np


@dataclass
class SearchWorld:
    params: SearchParams
    heightmap: HeightMap
    grid: Grid
    vis: Visibility                 # kamera przednia
    wrist_vis: Visibility           # kamera na nadgarstku (poza z TF)
    model: DetectorModel
    wrist_model: DetectorModel
    travel: TravelModel
    views: ViewSet
    allowed: np.ndarray = None      # komórki na mapie przejezdności (None - brak mapy)

    def prior(self, weights=None) -> np.ndarray:
        """
        p_i⁰: jednorodny na komórkach dostępnych dla próbek (teren nie za stromy).

        Z mapą przejezdności - tylko na niej: próbki są rozmieszczane na przejezdnym terenie
        (probe_supervisor, DESIGN §6). Mapa miejsc na próbki (bez miejsc, w których próbka znika
        w podłożu) to artefakt symulacji - prior jej nie uwzględnia (decyzja autora).
        """
        max_slope = math.tan(math.radians(self.params.belief.prior_max_slope_deg))
        active = self.vis.cell_valid & (np.nan_to_num(self.travel.slope, nan=np.inf)
                                        <= max_slope)
        if self.allowed is not None:
            active &= self.allowed
        return Belief.uniform_prior(active, self.params.belief.n_samples, weights)

    def new_belief(self, weights=None) -> Belief:
        return Belief(self.grid, self.prior(weights), self.params.belief, self.model)


def build_world(params: SearchParams, heightmap: HeightMap, cache_dir: str = '',
                trav_map: TraversabilityMap = None) -> SearchWorld:
    k = params.belief.cell_size / heightmap.grid.res
    if abs(k - round(k)) > 1e-6 or round(k) < 1:
        raise ValueError(f'cell_size {params.belief.cell_size} musi być wielokrotnością '
                         f'rozdzielczości mapy wysokości {heightmap.grid.res}')
    grid = heightmap.grid.coarsen(int(round(k)))
    vis = Visibility(heightmap, grid, params.camera, params.hfov, params.vfov,
                     base_radius=params.travel.robot_radius)
    wrist_vis = Visibility(heightmap, grid, params.wrist_camera, params.wrist_hfov,
                           params.wrist_vfov)
    model = DetectorModel(params.detector, params.camera, params.hfov)
    wrist_model = DetectorModel(params.wrist_detector, params.wrist_camera, params.wrist_hfov)
    allowed = None if trav_map is None else trav_map.allowed_at(*grid.centers())
    travel = TravelModel(heightmap, grid, params.travel, allowed)
    views = ViewSet.build(vis, travel, params.views, cache_dir)
    views.apply_model(model)
    return SearchWorld(params, heightmap, grid, vis, wrist_vis, model, wrist_model, travel,
                       views, allowed)
