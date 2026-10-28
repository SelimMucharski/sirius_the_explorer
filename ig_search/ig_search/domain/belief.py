"""
Belief o położeniu nieodnalezionych próbek na siatce komórek (DESIGN §3.2). Bez ROS.

p_i = P(w komórce i jest nieodnaleziona próbka) - zmienna Bernoulliego na komórkę,
przechowywana jako log-szanse ℓ_i. Dla obszaru A (skupisko, kula wokół ŝ):
P(≥ 1 próbka w A) = 1 − Π_{i∈A} (1 − p_i).

Aktualizacja po klatce detektora ze zbiorem detekcji {ẑ_k} (także pustym), dla każdej
komórki i objętej klatką:

    LR_i = (1 − P_d,i) + P_d,i · Σ_k 𝒩_ik / (λ_fa + c_k − P_d,i 𝒩_ik p_i),   ℓ_i ← ℓ_i + ln LR_i
    c_k  = Σ_j P_d,j 𝒩_jk p_j,        𝒩_ik = 𝒩(ẑ_k; c_i, σ_eff²)

Mianownik to gęstość detekcji z INNYCH źródeł niż komórka i: fałszywe alarmy i próbki
w pozostałych komórkach („explaining away”, normalizacja jak w filtrach PHD/MeMBer).
Komórka najlepiej tłumacząca detekcje dąży do p → 1, a jej sąsiedzi są „wyjaśnieni”
i maleją - jedna próbka nie rozlewa się na wiele pewnych komórek. Bez wykluczenia samej
komórki i z mianownika p zatrzymuje się w okolicy 0,5; bez normalizacji (c_k = 0,
formuła z DESIGN §3.2) przy małym λ_fa jedna próbka czyni pewnymi wiele sąsiednich
komórek, a po wyłączeniu otoczenia zgłoszonej próbki pozostałość wygląda jak kolejna
próbka. Parametr `explain_away` pozwala porównać oba warianty (ablacja). Przy pustym
beliefie (p ≈ 0) obie formuły są równe. Pominięcie dobrze potwierdzonej próbki obniża
p umiarkowanie (np. 0,99 -> 0,94), w przeciwieństwie do filtru PHD (D -> D·(1 − P_d)).

Bramka różnorodności widoku: komórka przyjmuje obserwację tylko z punktu widzenia
(kierunek kubełkowany, odległość kubełkowana logarytmicznie), z którego jeszcze jej
nie obserwowano. Fałszywe alarmy z tego samego miejsca są skorelowane, więc kolejne
klatki z tej samej pozy nie wnoszą informacji - ani pozytywnej, ani negatywnej.
"""
from dataclasses import dataclass
import math

from ig_search.domain.detector_model import DetectorModel
from ig_search.domain.params import BeliefParams
from ig_search.domain.terrain import Grid
from ig_search.domain.visibility import VisibleSet
import numpy as np


def bin_entropy(p):
    """Entropia binarna [bit]."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def logit(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1.0 - p))


def presence(p) -> float:
    """P(≥ 1 próbka) w obszarze o prawdopodobieństwach komórek p."""
    return float(-np.expm1(np.log1p(-np.clip(np.asarray(p, float), 0.0, 1.0 - 1e-15)).sum()))


@dataclass
class UpdateResult:
    n_updated: int          # komórki, które przeszły bramkę
    n_gated: int            # komórki odrzucone przez bramkę
    n_detections: int
    max_ell_gain: float     # największy wzrost log-szans w tej klatce


class Belief:

    def __init__(self, grid: Grid, prior: np.ndarray, params: BeliefParams,
                 model: DetectorModel):
        """Prior p_i⁰ na płaskiej siatce; komórki z prior = 0 są nieaktywne (bez próbek)."""
        if prior.shape != (grid.size,):
            raise ValueError(f'prior {prior.shape} != ({grid.size},)')
        self.grid = grid
        self.params = params
        self.model = model
        self.active = prior > 0
        self.ell = np.where(self.active, np.clip(logit(prior), params.ell_min, params.ell_max),
                            -np.inf)
        self.cx, self.cy = grid.centers()
        n_dir = params.gate_dir_buckets
        if not 1 <= n_dir <= 64:
            raise ValueError('gate_dir_buckets musi być w [1, 64]')
        self._n_range = 64 // n_dir
        self._gate = np.zeros(grid.size, np.uint64)

    # ------------------------------------------------------------ prior
    @staticmethod
    def uniform_prior(active: np.ndarray, n_samples: int, weights=None) -> np.ndarray:
        """p_i⁰ = M / N_d na komórkach dostępnych; z wagami - proporcjonalnie (wariant 6.4)."""
        w = active.astype(float) if weights is None else np.clip(weights, 0, None) * active
        total = w.sum()
        if total <= 0:
            raise ValueError('brak komórek dostępnych dla próbek')
        return np.clip(n_samples * w / total, 0.0, 0.5)

    # ------------------------------------------------------------ odczyt
    @property
    def p(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.ell))

    def set_p(self, ids, p):
        """Ustawia p w komórkach (testy, scenariusze startowe)."""
        self.ell[ids] = logit(p)

    def mass(self) -> float:
        """Σ p_i - oczekiwana liczba nieodnalezionych próbek."""
        return float(self.p[self.active].sum())

    def entropy(self) -> float:
        return float(bin_entropy(self.p[self.active]).sum())

    def copy(self) -> 'Belief':
        b = Belief.__new__(Belief)
        b.__dict__.update(self.__dict__)
        b.ell = self.ell.copy()
        b.active = self.active.copy()
        b._gate = self._gate.copy()
        return b

    # ------------------------------------------------------------ bramka
    def _gate_bits(self, ids, cam_xy, r):
        """Bit kubełka (kierunek × odległość) punktu widzenia; cam_xy - para lub tablice."""
        b = np.arctan2(np.asarray(cam_xy[1]) - self.cy[ids], np.asarray(cam_xy[0]) - self.cx[ids])
        n_dir = self.params.gate_dir_buckets
        # środki kubełków na wielokrotnościach 2π/n: kursy kandydackich póz są kwantowane
        # tak samo, więc kierunki obserwacji nie wypadają na granicach kubełków
        d = np.floor((b + math.pi) / (2 * math.pi) * n_dir + 0.5).astype(np.int64) % n_dir
        r0 = self.params.gate_range_min
        rb = np.floor(np.log(np.maximum(r, r0) / r0) / math.log(self.params.gate_range_ratio))
        rb = np.clip(rb.astype(np.int64), 0, self._n_range - 1)
        return np.left_shift(np.uint64(1), (d * self._n_range + rb).astype(np.uint64))

    # ------------------------------------------------------------ aktualizacja
    def update(self, vis: VisibleSet, detections_xy: np.ndarray, cam_xy,
               use_gate: bool = True, model: DetectorModel = None) -> UpdateResult:
        """
        Aktualizacja po jednej klatce.

        vis           - komórki objęte klatką (widoczne + otoczenie detekcji), z r i α,
        detections_xy - (K, 2) pozycje detekcji w ramce mapy (K może być 0),
        cam_xy        - położenie kamery (do bramki różnorodności),
        model         - model detektora tej kamery (domyślnie kamera przednia).
        """
        model = model or self.model
        det = np.asarray(detections_xy, float).reshape(-1, 2)
        keep = self.active[vis.ids]
        ids, r, alpha = vis.ids[keep], vis.r[keep], vis.alpha[keep]
        if len(ids) == 0:
            return UpdateResult(0, 0, len(det), 0.0)
        pd = model.pd(r, alpha)
        dens = None
        if len(det):
            dens = self._densities(ids, r, det, model)                  # (n, K)
            if self.params.explain_away:
                own = pd[:, None] * dens * self.p[ids, None]            # wkład komórki i
                other = np.maximum(own.sum(axis=0)[None, :] - own, 0.0)  # c_k bez komórki i
            else:
                other = np.zeros_like(dens)
        novel = np.ones(len(ids), bool)
        bits = None
        if use_gate:
            bits = self._gate_bits(ids, cam_xy, r)
            novel = (self._gate[ids] & bits) == 0
        n_gated = int((~novel).sum())
        ids, pd = ids[novel], pd[novel]
        if len(ids) == 0:
            return UpdateResult(0, n_gated, len(det), 0.0)

        lr = 1.0 - pd
        if dens is not None:
            lr = lr + pd * (dens[novel] / (model.lambda_fa + other[novel])).sum(axis=1)
        gain = np.log(np.maximum(lr, 1e-300))
        self.ell[ids] = np.clip(self.ell[ids] + gain, self.params.ell_min, self.params.ell_max)
        if use_gate:
            np.bitwise_or.at(self._gate, ids, bits[novel])
        return UpdateResult(len(ids), n_gated, len(det), float(gain.max()))

    def _densities(self, ids, r, det, model) -> np.ndarray:
        """
        𝒩(ẑ_k; c_i, σ_eff²) w oknie `detection_window_sigmas`·σ_eff, kształt (n, K).

        σ_eff² = σ(r)² + a²/12: gęstość uśredniona po powierzchni komórki (wariancja
        rozkładu jednostajnego w komórce o boku a). Bez tego przy σ ≪ a (kamera na
        nadgarstku, σ ≈ 2 cm, a = 20 cm) detekcja próbki leżącej z dala od środka komórki
        nie trafia w żadną komórkę i działa jak obserwacja negatywna.
        """
        sigma = model.sigma(r)
        d2 = ((self.cx[ids, None] - det[None, :, 0]) ** 2
              + (self.cy[ids, None] - det[None, :, 1]) ** 2)
        s2 = (sigma ** 2 + self.grid.res ** 2 / 12.0)[:, None]
        dens = np.exp(-d2 / (2 * s2)) / (2 * math.pi * s2)
        w = self.params.detection_window_sigmas
        return np.where(d2 <= (w * w) * s2, dens, 0.0)

    def clear(self, x: float, y: float, radius: float):
        """
        Po zgłoszeniu: w promieniu od ŝ nie ma już nieodnalezionej próbki.

        Komórki są trwale wyłączane - odpowiada to podjęciu próbki. W symulacji próbka
        zostaje na miejscu (chwytanie poza zakresem), a jej kolejne detekcje nie mogą
        prowadzić do ponownego zgłoszenia tej samej próbki.
        """
        m = self.active & ((self.cx - x) ** 2 + (self.cy - y) ** 2 <= radius * radius)
        self.ell[m] = -np.inf
        self.active[m] = False

    def reset_to_prior(self, ids, prior: np.ndarray):
        """Przywraca wartości a priori (np. resztki skupiska po zgłoszeniu próbki)."""
        ids = ids[self.active[ids]]
        self.ell[ids] = np.clip(logit(prior[ids]), self.params.ell_min, self.params.ell_max)

    def view_novelty(self, views) -> np.ndarray:
        """
        Dla każdego niezerowego (v, i) macierzy V: czy punkt widzenia v jest dla komórki i nowy.

        Planer musi liczyć zysk spójnie z bramką - obserwacja z kubełka, który komórka już
        przyjęła, nic nie wnosi (inaczej łazik stoi w miejscu, „obserwując” w kółko).
        """
        key = (self.params.gate_dir_buckets, self.params.gate_range_ratio,
               self.params.gate_range_min)
        if getattr(views, '_gate_key', None) != key:
            rows = np.repeat(np.arange(len(views)), np.diff(views.indptr))
            views._gate_bits = self._gate_bits(views.indices, (views.cam_x[rows],
                                                               views.cam_y[rows]), views.r)
            views._gate_key = key
        return (self._gate[views.indices] & views._gate_bits) == 0

    def observed(self) -> np.ndarray:
        """Maska komórek, które przyjęły co najmniej jedną obserwację."""
        return self._gate != 0

    def view_counts(self) -> np.ndarray:
        """Liczba różnych punktów widzenia (kubełków bramki), z których obejrzano komórkę."""
        return _POPCOUNT8[self._gate.view(np.uint8)].reshape(-1, 8).sum(axis=1)


_POPCOUNT8 = np.array([bin(i).count('1') for i in range(256)], np.uint8)


def no_detection_update(p: np.ndarray, pd: np.ndarray) -> np.ndarray:
    """Najbardziej prawdopodobny wynik obserwacji przy małych p: brak detekcji."""
    return p * (1.0 - pd) / (1.0 - p * pd)
