"""M10: strategia z horyzontem ig_horizon (DESIGN §3.6)."""
from conftest import flat, make_heightmap
from ig_search.application.world import build_world
from ig_search.domain.params import from_dict
from ig_search.domain.strategies import make_strategy
from ig_search.domain.strategies.base import OBSERVE, PlanningContext
import numpy as np
import pytest

PARAMS = from_dict({'views': {'vp_step': 2.0}})


@pytest.fixture(scope='module')
def world():
    return build_world(PARAMS, make_heightmap(flat))


def ctx_for(world, belief, q, params=PARAMS):
    return PlanningContext(q, belief, world.views, world.travel,
                           world.travel.drive_times(q[0], q[1]), [], params)


def blob(world, belief, x, y, r, p):
    cx, cy = world.grid.centers()
    belief.ell[np.hypot(cx - x, cy - y) < r] = np.log(p / (1.0 - p))


def strategy(name, **planner):
    return make_strategy(from_dict({'planner': {'strategy': name, **planner}}))


def scene(world, near_p=0.1):
    """Mała plama za łazikiem, trzy plamy dalej przed nim."""
    b = world.new_belief()
    b.ell[:] = -12.0
    blob(world, b, 3.0, 10.0, 1.0, near_p)
    for y in (6.0, 10.0, 14.0):
        blob(world, b, 16.0, y, 1.0, 0.1)
    return b


def test_depth_one_is_greedy(world):
    b = scene(world)
    q = (7.0, 10.0, 0.0)
    a = strategy('ig_greedy').choose(ctx_for(world, b, q))
    h = strategy('ig_horizon', horizon_depth=1).choose(ctx_for(world, b, q))
    assert a.kind == h.kind == OBSERVE
    assert a.view == h.view
    assert abs(a.value - h.value) < 1e-9 * a.value


def test_horizon_skips_small_detour_for_cluster(world):
    b = scene(world)
    q = (7.0, 10.0, 0.0)
    a = strategy('ig_greedy').choose(ctx_for(world, b, q))
    h = strategy('ig_horizon').choose(ctx_for(world, b, q))
    assert a.x < q[0]                       # zachłannie: zawraca po mały zysk
    assert h.x > q[0] + 3.0                 # z horyzontem: jedzie do skupiska


def test_sequence_does_not_count_area_twice(world):
    b = world.new_belief()
    b.ell[:] = -12.0
    blob(world, b, 12.0, 10.0, 1.0, 0.3)
    s = strategy('ig_horizon')
    c = ctx_for(world, b, (7.0, 10.0, 0.0))
    node = s._plan(c)
    single = s._plan(c, first=node.views[0])
    assert len(node.views) == 3 and len(set(node.views)) == 3
    one = strategy('ig_horizon', horizon_depth=1)._plan(c, first=node.views[0])
    # dwie kolejne pozy dokładają mniej niż pierwsza, bo plama jest już „obejrzana”
    assert node.gain - one.gain < one.gain
    assert single.value == pytest.approx(node.value)


def test_value_of_matches_choice(world):
    b = scene(world)
    s = strategy('ig_horizon')
    c = ctx_for(world, b, (7.0, 10.0, 0.0))
    a = s.choose(c)
    assert s.value_of(c, a) == pytest.approx(a.value)


@pytest.mark.parametrize('depth', [2, 5])
def test_configurable_depth(world, depth):
    b = scene(world)
    s = strategy('ig_horizon', horizon_depth=depth)
    c = ctx_for(world, b, (7.0, 10.0, 0.0))
    node = s._plan(c)
    a = s.choose(c)
    assert len(node.views) == depth and len(set(node.views)) == depth
    assert len(a.plan) == depth
    assert a.view == node.views[0]
