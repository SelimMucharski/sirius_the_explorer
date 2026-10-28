"""Miary serii (M9): Kaplan-Meier, η, π, czasy kolejnych odnalezień."""
from ig_search_experiments.metrics import (
    aborted, episode_from_summary, kaplan_meier, km_median, summarize, time_to_k)
import numpy as np
import pytest


def summary(reports, elapsed=500.0, state='DONE'):
    return {'state': state, 'elapsed': elapsed, 'distance': 100.0, 'seed': 1,
            'variant': 'v', 'reports': [
                {'t': t, 'probe': p, 'error_now': e, 'grasped': False} for t, p, e in reports]}


def test_kaplan_meier_matches_hand_computation():
    # czasy 1, 2(cenz.), 3, 3, 4(cenz.): S(1) = 4/5, S(3) = 4/5 · (1 - 2/3) = 4/15
    t, s = kaplan_meier([1, 2, 3, 3, 4], [True, False, True, True, False])
    assert t.tolist() == [1.0, 3.0]
    assert s == pytest.approx([0.8, 0.8 / 3])
    assert km_median(t, s) == 3.0


def test_km_median_not_reached():
    t, s = kaplan_meier([5, 10, 10], [True, False, False])
    assert km_median(t, s) is None


def test_episode_counts_duplicates_and_far_reports_as_false():
    ep = episode_from_summary(summary([(100, 'a', 0.02), (200, 'a', 0.03),
                                       (300, 'b', 0.40), (400, 'c', 0.10)]))
    assert ep.times == [100.0, 400.0]
    assert ep.false_reports == 2
    assert ep.errors == pytest.approx([0.02, 0.10])


def test_time_to_k_censors_incomplete_and_aborted():
    eps = [episode_from_summary(summary([(50, 'a', 0.01), (90, 'b', 0.01), (120, 'c', 0.01)])),
           episode_from_summary(summary([(70, 'a', 0.01)], elapsed=900.0)),   # skończyła sama
           aborted('v', 3, elapsed=600.0)]
    d, e = time_to_k(eps, 3, 1800.0)
    assert d == [120.0, 1800.0, 600.0] and e == [True, False, False]
    d1, e1 = time_to_k(eps, 1, 1800.0)
    assert d1 == [50.0, 70.0, 600.0] and e1 == [True, True, False]


def test_summarize_rates():
    eps = [episode_from_summary(summary([(50, 'a', 0.01), (90, 'b', 0.01), (120, 'c', 0.01)])),
           episode_from_summary(summary([(70, 'a', 0.01), (80, 'x', 0.5)], elapsed=1800.0,
                                        state='TIMEOUT'))]
    s = summarize(eps)
    assert s['eta'] == pytest.approx(4 / 6)
    assert s['precision'] == pytest.approx(4 / 5)
    assert s['complete'] == 1
    assert s['t1_median'] == 50.0
    assert np.isclose(s['distance_mean'], 100.0)


def test_score_uses_erc_time_factor():
    # 3/3 w 7:30 - pełne punkty; 3/3 w 12:00 - 90 %; 2/3 po T_max - 60 % z 2/3; przerwany - brak
    full = episode_from_summary(summary([(50, 'a', 0.01), (90, 'b', 0.01), (450, 'c', 0.01)],
                                        elapsed=450.0))
    late = episode_from_summary(summary([(50, 'a', 0.01), (90, 'b', 0.01), (720, 'c', 0.01)],
                                        elapsed=720.0))
    two = episode_from_summary(summary([(50, 'a', 0.01), (90, 'b', 0.01)], elapsed=1800.0,
                                       state='TIMEOUT'))
    assert full.score() == pytest.approx(1.0)
    assert late.score() == pytest.approx(0.9)
    assert two.score() == pytest.approx(0.6 * 2 / 3)
    assert np.isnan(aborted('v', 1, 300.0).score())
    assert summarize([full, late, two, aborted('v', 2)])['score_mean'] == pytest.approx(
        (1.0 + 0.9 + 0.4) / 3)
