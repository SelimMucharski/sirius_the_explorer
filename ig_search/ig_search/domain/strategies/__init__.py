"""Strategie wyboru akcji; `make_strategy(params)` wybiera według planner.strategy."""
from ig_search.domain.params import SearchParams
from ig_search.domain.strategies.frontier import Chase, Frontier
from ig_search.domain.strategies.ig_greedy import IGGreedy
from ig_search.domain.strategies.ig_horizon import IGHorizon
from ig_search.domain.strategies.meander import Meander

STRATEGIES = {s.name: s for s in (IGGreedy, IGHorizon, Meander, Frontier, Chase)}


def make_strategy(params: SearchParams):
    name = params.planner.strategy
    if name not in STRATEGIES:
        raise ValueError(f'nieznana strategia {name!r}; dostępne: {sorted(STRATEGIES)}')
    return STRATEGIES[name](params)
