"""Mapa przejezdności (format nav2_map_server) i jej użycie w ruchu i priorze (DESIGN §6)."""
from conftest import default_params, flat, make_heightmap, ridge
from ig_search.application.world import build_world
from ig_search.domain.traversability import TraversabilityMap
import numpy as np


def test_ridge_blocked_and_one_component():
    hm = make_heightmap(ridge(x0=10.0, height=1.5))
    tm = TraversabilityMap.from_heightmap(hm, default_params().travel, start=(4.0, 10.0))
    assert tm.allowed_at(4.0, 10.0)[0]
    assert not tm.allowed_at(10.0, 10.0)[0]               # zbocze grzbietu
    assert not tm.allowed_at(16.0, 10.0)[0]               # druga strona - inna składowa
    assert not tm.allowed_at(0.5, 10.0)[0]                # odstęp od krawędzi mapy


def test_save_load_roundtrip(tmp_path):
    hm = make_heightmap(flat)
    tm = TraversabilityMap.from_heightmap(hm, default_params().travel, start=(10.0, 10.0))
    tm.mask[:, :tm.mask.shape[1] // 2] = False            # lewa połowa zamknięta
    path = str(tmp_path / 'trav.yaml')
    tm.save(path)
    back = TraversabilityMap.load(path)
    assert back.grid == tm.grid
    assert np.array_equal(back.mask, tm.mask)
    assert back.allowed_at(15.0, 10.0)[0] and not back.allowed_at(5.0, 10.0)[0]


def test_world_respects_map():
    hm = make_heightmap(flat)
    tm = TraversabilityMap.from_heightmap(hm, default_params().travel, start=(10.0, 10.0))
    tm.mask[:, :tm.mask.shape[1] // 2] = False
    w = build_world(default_params(), hm, trav_map=tm)
    left = w.grid.flat_index(5.0, 10.0)
    right = w.grid.flat_index(15.0, 10.0)
    assert not w.travel.traversable[left] and w.travel.traversable[right]
    prior = w.prior()
    assert prior[left] == 0.0 and prior[right] > 0.0


def test_gentle_mound_excluded():
    def mound(cx, cy):                                    # kopiec 0,5 m o zboczach ~9°
        return 0.5 * np.clip(1.0 - np.hypot(cx - 12.0, cy - 10.0) / 3.0, 0.0, None)
    hm = make_heightmap(mound)
    p = default_params().travel
    with_hill = TraversabilityMap.from_heightmap(hm, p, start=(4.0, 10.0))
    without = TraversabilityMap.from_heightmap(hm, p, start=(4.0, 10.0), hill_height=0.0)
    assert without.allowed_at(12.0, 10.0)[0]              # zbocza w normie nachylenia
    assert not with_hill.allowed_at(12.0, 10.0)[0]
    assert with_hill.allowed_at(4.0, 10.0)[0]


def test_excluded_polygon():
    hm = make_heightmap(flat)
    zone = [(12.0, 8.0), (18.0, 8.0), (18.0, 12.0), (12.0, 12.0)]
    tm = TraversabilityMap.from_heightmap(hm, default_params().travel, start=(5.0, 10.0),
                                          exclude=[zone])
    assert tm.allowed_at(5.0, 10.0)[0]
    assert not tm.allowed_at(15.0, 10.0)[0]
    assert tm.allowed_at(15.0, 5.0)[0]


def test_buried_spots_removed_from_spawn_map_not_prior():
    hm = make_heightmap(flat)
    lowered = make_heightmap(lambda cx, cy: np.where(cx > 10.0, -0.05, 0.0))   # kolizja 5 cm niżej
    tm = TraversabilityMap.from_heightmap(hm, default_params().travel, start=(5.0, 10.0))
    spawn = tm.spawn_map(hm, lowered)
    assert spawn.allowed_at(5.0, 10.0)[0]
    assert not spawn.allowed_at(15.0, 10.0)[0]            # próbka zapadłaby się 5 cm
    assert not spawn.allowed_at(10.1, 10.0)[0]            # okno 0,5 m wokół
    w = build_world(default_params(), hm, trav_map=tm)
    prior = w.prior()
    # prior i jazda - tylko z mapy przejezdności (miejsca na próbki to artefakt symulacji)
    assert prior[w.grid.flat_index(15.0, 10.0)] == prior[w.grid.flat_index(5.0, 10.0)] > 0
    assert w.travel.traversable[w.grid.flat_index(15.0, 10.0)]
