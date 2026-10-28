"""
Miary serii epizodów (DESIGN §4.3, M9). Bez ROS.

Epizod (summary.json z probe_watch) -> czasy kolejnych poprawnych zgłoszeń t₁ ≤ t₂ ≤ t₃,
fałszywe zgłoszenia, droga. Seria -> estymator Kaplana-Meiera czasu k-tego odnalezienia.
Cenzurowanie: misja zakończona (DONE, TIMEOUT) bez k poprawnych zgłoszeń - w T_max (sama już
nic nie znajdzie); epizod przerwany technicznie (ABORTED) - w chwili przerwania,
skuteczność η = n_pop / (M·K), precyzja π = n_pop / (n_pop + n_fał).
Punktacja epizodu - (n_pop / M) · f(t_końca) z f(t) regulaminu ERC (MissionParams.score_factor);
epizod przerwany technicznie nie ma punktacji.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ig_search.domain.params import MissionParams
import numpy as np

EPS = 0.15              # [m] zgłoszenie poprawne, gdy ŝ bliżej niż ε od próbki
M = 3                   # próbek w epizodzie


@dataclass
class Episode:
    variant: str
    seed: int
    state: str                      # DONE | TIMEOUT | ABORTED
    elapsed: float                  # [s] czas misji (koniec albo przerwanie)
    times: List[float]              # czasy kolejnych poprawnych zgłoszeń (różne próbki)
    false_reports: int
    reports: int
    distance: float
    errors: List[float] = field(default_factory=list)   # błędy poprawnych zgłoszeń [m]
    grasped: int = 0

    def score(self, mission: MissionParams = MissionParams()) -> float:
        """Ułamek maksymalnej punktacji: (n_pop / M) · f(czas ukończenia misji)."""
        if self.state == 'ABORTED':
            return float('nan')
        return len(self.times) / M * mission.score_factor(self.elapsed)


def episode_from_summary(s: dict, variant: str = '', seed: int = -1,
                         eps: float = EPS) -> Episode:
    """Epizod z summary.json; zgłoszenie tej samej próbki drugi raz liczy się jako fałszywe."""
    seen, times, errors, false = set(), [], [], 0
    for r in sorted(s['reports'], key=lambda r: r['t']):
        if r['error_now'] < eps and r['probe'] not in seen:
            seen.add(r['probe'])
            times.append(float(r['t']))
            errors.append(float(r['error_now']))
        else:
            false += 1
    return Episode(variant or s.get('variant') or s.get('strategy', ''),
                   int(s.get('seed', seed) if seed < 0 else seed), s['state'],
                   float(s['elapsed']), times, false, len(s['reports']),
                   float(s.get('distance', 0.0)), errors,
                   sum(bool(r.get('grasped')) for r in s['reports']))


def aborted(variant: str, seed: int, elapsed: float = 0.0) -> Episode:
    """Epizod przerwany bez summary.json (cenzurowany w chwili przerwania)."""
    return Episode(variant, seed, 'ABORTED', elapsed, [], 0, 0, 0.0)


def kaplan_meier(durations: Sequence[float],
                 observed: Sequence[bool]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estymator Kaplana-Meiera S(t) = P(T > t).

    Zwraca (czasy zdarzeń, S tuż po każdym z nich); S = 1 przed pierwszym zdarzeniem.
    Przy równych czasach zdarzenia poprzedzają cenzurowania.
    """
    d = np.asarray(durations, float)
    e = np.asarray(observed, bool)
    times, surv, s = [], [], 1.0
    for t in np.unique(d[e]):
        at_risk = int(np.sum(d >= t))
        events = int(np.sum((d == t) & e))
        s *= 1.0 - events / at_risk
        times.append(float(t))
        surv.append(s)
    return np.array(times), np.array(surv)


def km_median(times: np.ndarray, surv: np.ndarray) -> Optional[float]:
    """Mediana z krzywej KM: pierwszy czas, w którym S ≤ 0,5 (None - nie osiągnięta)."""
    idx = np.flatnonzero(surv <= 0.5)
    return float(times[idx[0]]) if len(idx) else None


def time_to_k(episodes: Sequence[Episode], k: int, t_max: float):
    """
    (czasy, zdarzenie?) k-tego odnalezienia.

    Brak: misja zakończona - cenzurowanie w T_max; przerwana technicznie - w chwili przerwania.
    """
    d, e = [], []
    for ep in episodes:
        if len(ep.times) >= k and ep.times[k - 1] <= t_max:
            d.append(ep.times[k - 1])
            e.append(True)
        elif ep.state == 'ABORTED' and ep.elapsed > 0:
            d.append(min(ep.elapsed, t_max))
            e.append(False)
        else:
            d.append(t_max)
            e.append(False)
    return d, e


def summarize(episodes: Sequence[Episode], t_max: float = 1800.0) -> Dict[str, float]:
    """Miary zbiorcze wariantu."""
    n = len(episodes)
    correct = sum(len(ep.times) for ep in episodes)
    false = sum(ep.false_reports for ep in episodes)
    out = {'episodes': n, 'eta': correct / (M * n) if n else float('nan'),
           'precision': correct / (correct + false) if correct + false else float('nan'),
           'false_reports': false, 'complete': sum(len(ep.times) >= M for ep in episodes),
           'aborted': sum(ep.state == 'ABORTED' for ep in episodes),
           'distance_mean': float(np.mean([ep.distance for ep in episodes])) if n else 0.0,
           'error_mean': (float(np.mean([e for ep in episodes for e in ep.errors]))
                          if correct else float('nan')),
           'grasped': sum(ep.grasped for ep in episodes)}
    scores = [ep.score() for ep in episodes if ep.state != 'ABORTED']
    out['score_mean'] = float(np.mean(scores)) if scores else float('nan')
    for k in range(1, M + 1):
        t, s = kaplan_meier(*time_to_k(episodes, k, t_max))
        out[f't{k}_median'] = km_median(t, s)
    return out
