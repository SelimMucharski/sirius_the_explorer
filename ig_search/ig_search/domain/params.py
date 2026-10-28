"""
Parametry systemu jako zagnieżdżone dataclassy (bez ROS).

Jedna struktura `SearchParams`, ładowana z YAML (`from_dict`) albo z parametrów ROS
o nazwach z kropką (`flatten` / `from_flat`), np. `camera.hfov_deg`.
Notacja zgodna z docs/DESIGN.md §1.
"""
from dataclasses import dataclass, field, fields, is_dataclass, replace
import math
from typing import List


@dataclass(frozen=True)
class CameraParams:
    """Kamera przednia (OAK-D W) i jej zamocowanie w ramce base_link."""

    hfov_deg: float = 96.8          # 2·atan(W / (fx + fy)) z kalibracji, 1280x720
    vfov_deg: float = 64.7          # z hfov i proporcji 16:9
    r_min: float = 0.3              # [m] minimalny zasięg detekcji
    r_max: float = 6.0              # [m] powyżej P_d = 0
    mount_x: float = 0.223          # [m] przesunięcie kamery w przód od base_link (URDF)
    mount_z: float = 0.392          # [m] wysokość kamery nad terenem (base_link na gruncie)
    mount_pitch_deg: float = 0.0    # pochylenie montażu (+ w dół)
    frame_margin_deg: float = 5.0   # punkt widoczny tylko tyle od krawędzi kadru (próbka 20 cm
    #                                 musi być cała w obrazie; przy krawędzi detektor jej nie
    #                                 widzi, a obserwacja negatywna zbijała prawdziwą próbkę)
    target_height: float = 0.03     # δ: wysokość punktu próbki nad terenem [m]
    occlusion_margin: float = 0.02  # [m] promień musi przejść tyle nad terenem
    occlusion_step: float = 0.1     # [m] krok marszu po promieniu
    occlusion_skip_end: float = 0.25  # [m] końcówka promienia pomijana (przy celu)


@dataclass(frozen=True)
class DetectorParams:
    """
    Model detektora planera: P_d(r, α), σ(r), λ_fa (DESIGN §3.2).

    Domyślne wartości przed kalibracją (M6): recall 0,83 i 0,07 FP na klatkę
    z walidacji detektora front; λ_fa = 0,07 / ~20 m² widocznego terenu.
    """

    pd_max: float = 0.85
    r_full: float = 2.5             # [m] do tej odległości P_d = pd_max
    edge_falloff: float = 0.3       # spadek P_d na brzegu kadru (0 = brak)
    sigma0: float = 0.05            # σ(r) = sigma0 + sigma1·r  [m]
    sigma1: float = 0.04
    lambda_fa: float = 0.0035       # [1/(m²·klatka)]
    calibration_file: str = ''      # tablice z kalibracji (M6); '' = model parametryczny


@dataclass(frozen=True)
class BeliefParams:
    cell_size: float = 0.2          # a [m]
    n_samples: int = 3              # M
    ell_min: float = -12.0          # ograniczenie log-szans
    ell_max: float = 12.0
    gate_dir_buckets: int = 8       # bramka różnorodności: kubełki kierunku
    gate_range_ratio: float = 1.5   # kolejny kubełek odległości co ten współczynnik
    gate_range_min: float = 0.3     # [m] dolna granica pierwszego kubełka odległości
    prior_max_slope_deg: float = 35.0  # komórki bardziej strome nie są „dostępne” dla próbek
    detection_window_sigmas: float = 3.0
    explain_away: bool = True       # mianownik λ_fa + c_k bez komórki i (False = DESIGN §3.2)


@dataclass(frozen=True)
class TravelParams:
    v_max: float = 0.6              # [m/s] jak desired_linear_vel w Nav2
    w_max: float = 0.8              # [rad/s]
    t_overhead: float = 2.0         # [s] stały narzut na cel
    slope_max_deg: float = 20.0     # krawędzie bardziej strome usunięte
    uphill_cost: float = 2.0        # czas · (1 + c_up·s⁺ + c_down·s⁻), s = tg nachylenia
    downhill_cost: float = 0.5
    robot_radius: float = 0.55      # [m] odstęp od komórek nieprzejezdnych
    roughness_max: float = 0.12     # [m] max odch. std. wysokości w promieniu robota
    edge_margin: float = 1.5        # [m] odstęp od krawędzi mapy wysokości (poza nią teren
    #  nieznany: w Gazebo łazik wyjechał 0,25 m za ostatnią przejezdną komórkę przy krawędzi
    #  i spadł z uskoku - przeszkód ujemnych chmura Nav2 nie oznacza)
    bump_max: float = 0.15          # [m] max wysokość garbu ponad lokalną średnią w promieniu
    #  robota (prześwit pod platformą ~0,20 m: wheel_radius + suspension_height - suspension_z;
    #  w Gazebo łazik zawisł na skale 43 cm, choć chropowatość i nachylenie były w normie)


@dataclass(frozen=True)
class ViewParams:
    vp_step: float = 1.0            # [m] rozstaw kandydackich póz obserwacji
    n_yaw: int = 8
    edge_clearance: float = 0.0     # [m] pozy obserwacji i punkty meandra co najmniej tyle od
    #  terenu nieprzejezdnego i brzegu mapy (jazda bliżej dozwolona). 0 - wyłączone: odstęp,
    #  który omija miejsca utknięć Nav2 (2-2,6 m), zostawiał na erc2025 152-440 z 2808 póz
    #  i 47-81% widocznego priora (DESIGN §6)


@dataclass(frozen=True)
class CandidateParams:
    p_seed: float = 0.02            # komórki z p > p_seed tworzą skupiska
    approach_distance: float = 0.78  # [m] base_link -> ŝ: ŝ w środku obszaru chwytu (płasko)
    front_verify_distance: float = 1.15  # [m] weryfikacja kamerą przednią (bez nadgarstka;
    #                                      kamera nie widzi gruntu bliżej niż ~0,84 m)
    n_approach_dirs: int = 8
    t_verify: float = 15.0          # [s] czas weryfikacji na miejscu (skan nadgarstkiem)
    p_abandon: float = 0.05         # P_k poniżej -> porzuć kandydata
    p_approach_min: float = 0.1     # nowy dojazd tylko do kandydata z P_k ≥ tyle (w Gazebo dojazdy
    #                                 do P_k 0,04-0,09 kończyły się porzuceniem po 3 s)
    block_time: float = 90.0        # [s] porzucony kandydat ignorowany przez ten czas
    max_mass_under_rover: float = 0.1   # poza dojazdu: najwyżej taki ułamek P_k pod łazikiem
    #  (promień travel.robot_radius) - estymata ŝ z daleka myli się do ~0,7 m i poza 0,78 m od ŝ
    #  stawiała łazik na prawdziwej próbce (niewidoczna z kamery przedniej, potrącana)


@dataclass(frozen=True)
class ReportParams:
    thresholds: List[float] = field(default_factory=lambda: [0.80, 0.85, 0.95])  # τ_n
    epsilon: float = 0.15           # ε: promień kuli wokół ŝ [m]
    clear_radius: float = 0.4       # ε': promień wyłączenia beliefu po zgłoszeniu [m]
    # R(q): pierścień wokół podstawy ramienia, gdzie osiągalny jest cały chwyt pionowy (pozycja
    # przed chwytem i opuszczenie palców 12 mm poniżej osi próbki), ograniczony do kątów przed
    # łazikiem (z tyłu i z boku za podstawą - kadłub i koła). Zewnętrzny promień z IK zależy od
    # wysokości środka próbki względem base_link (dz): niżej położona próbka - krótszy zasięg.
    workspace_root_x: float = 0.1145  # [m] podstawa ramienia przed base_link
    workspace_r_min: float = 0.60   # [m] od podstawy ramienia
    workspace_dz: List[float] = field(
        default_factory=lambda: [-0.06, -0.03, 0.0, 0.015, 0.03, 0.06])   # [m]
    workspace_r_max_dz: List[float] = field(
        default_factory=lambda: [0.62, 0.68, 0.72, 0.74, 0.76, 0.78])     # [m]
    workspace_half_angle_deg: float = 75.0
    # dojazd kończy się, gdy ŝ jest w R(q) z tym zapasem (wybieg łazika; błąd ŝ z kamery
    # przedniej poprawia skan i pomiar z bliska przed chwytem)
    reach_margin: float = 0.035     # [m]
    reach_angle_margin_deg: float = 10.0
    max_verify_attempts: int = 2


@dataclass(frozen=True)
class PlannerParams:
    strategy: str = 'ig_greedy'     # ig_greedy | ig_horizon | meander | frontier | chase
    hysteresis: float = 0.25        # nowa akcja musi być o tyle lepsza, by wywłaszczyć
    replan_period: float = 3.0      # [s]
    min_gain: float = 1e-4          # poniżej -> brak sensownych akcji
    horizon_depth: int = 3          # D - liczba obserwacji w sekwencji (ig_horizon)
    beam_width: int = 10            # B - sekwencje zostające po każdym poziomie (ig_horizon)
    beam_candidates: int = 20       # N - rozszerzenia węzła wiązki (najlepsze zachłannie)
    approach_threshold: float = 0.5  # strategie odniesienia: dojedź, gdy P_k > próg
    meander_overlap: float = 0.2    # zakładka pasów meandra (ułamek szerokości)
    meander_waypoint_step: float = 2.0  # [m] rozstaw punktów na pasie
    waypoint_reached: float = 1.0   # [m] punkt meandra uznany za osiągnięty
    frontier_min_gain: float = 3.0  # min. Σ P_d po nieobejrzanych komórkach widoku
    # przejścia pokrycia meandra i frontiera (0 - bez limitu, do T_max; 1 - jedno, jak przed
    # 2026-10-27: po pokryciu terenu misja się kończyła)
    coverage_passes: int = 0


@dataclass(frozen=True)
class WristParams:
    """Skan weryfikujący kamerą OAK-D SR na nadgarstku i pozycja przed chwytem."""

    scan_height: float = 0.5        # [m] kamera -> ŝ wzdłuż osi optycznej (kadr ±0,35 × ±0,22 m)
    scan_timeout: float = 25.0      # [s] na ruch ramienia + zbieranie klatek
    pregrasp_timeout: float = 25.0  # [s]
    pregrasp_height: float = 0.05   # [m] końcówki palców nad osią próbki przed chwytem
    grasp_depth: float = 0.012      # [m] końcówki palców poniżej osi próbki przy zamykaniu
    #                                 (0,005 - styk 5 mm nad krawędzią szczęk, próbka wysuwała się)
    grasp: bool = True              # chwyt i odłożenie próbki po zgłoszeniu
    grasp_timeout: float = 40.0     # [s] opuszczenie + zamknięcie + podniesienie
    match_radius: float = 0.3       # [m] detekcja z nadgarstka należy do ŝ


@dataclass(frozen=True)
class MissionParams:
    t_max: float = 1800.0           # [s] T_max
    # f(t) z regulaminu ERC (7.3.4.4 a): ułamek zebranych punktów zależnie od czasu ukończenia
    # zadania - 100 % do 7:30, 90 % do 12:30, 80 % do 17:30, 70 % do 22:30, potem 60 %
    score_steps: List[float] = field(
        default_factory=lambda: [450.0, 750.0, 1050.0, 1350.0])     # [s] progi
    score_factors: List[float] = field(
        default_factory=lambda: [1.0, 0.9, 0.8, 0.7, 0.6])          # f przed progiem / po
    use_wrist: bool = True          # skan weryfikujący i pozycja przed chwytem

    def score_factor(self, t: float) -> float:
        """f(t) - schodkowa; równo na progu liczy się jeszcze wyższy stopień."""
        for step, f in zip(self.score_steps, self.score_factors):
            if t <= step:
                return f
        return self.score_factors[len(self.score_steps)]


@dataclass(frozen=True)
class SearchParams:
    camera: CameraParams = field(default_factory=CameraParams)
    detector: DetectorParams = field(default_factory=DetectorParams)
    belief: BeliefParams = field(default_factory=BeliefParams)
    travel: TravelParams = field(default_factory=TravelParams)
    views: ViewParams = field(default_factory=ViewParams)
    candidates: CandidateParams = field(default_factory=CandidateParams)
    report: ReportParams = field(default_factory=ReportParams)
    planner: PlannerParams = field(default_factory=PlannerParams)
    mission: MissionParams = field(default_factory=MissionParams)
    wrist: WristParams = field(default_factory=WristParams)
    # OAK-D SR na nadgarstku: krótki zasięg, poza kamery z TF (montaż nieużywany)
    wrist_camera: CameraParams = field(default_factory=lambda: CameraParams(
        hfov_deg=70.5, vfov_deg=47.7, r_min=0.1, r_max=1.2, mount_x=0.0, mount_z=0.0,
        frame_margin_deg=0.0,       # ramię celuje kamerą w ŝ; kadr z 0,3 m to ±0,13 m, a środek
        #                             komórki beliefu bywa 0,1 m od próbki
        occlusion_step=0.03, occlusion_skip_end=0.05))
    wrist_detector: DetectorParams = field(default_factory=lambda: DetectorParams(
        pd_max=0.95, r_full=0.7, edge_falloff=0.2, sigma0=0.01, sigma1=0.02, lambda_fa=0.08))

    @property
    def hfov(self) -> float:
        return math.radians(self.camera.hfov_deg)

    @property
    def vfov(self) -> float:
        return math.radians(self.camera.vfov_deg)

    @property
    def wrist_hfov(self) -> float:
        return math.radians(self.wrist_camera.hfov_deg)

    @property
    def wrist_vfov(self) -> float:
        return math.radians(self.wrist_camera.vfov_deg)


def _merge(base, data):
    """Kopia `base` z polami nadpisanymi wartościami z `data` (rekurencyjnie, z rzutowaniem)."""
    if not data:
        return base
    known = {f.name for f in fields(base)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f'{type(base).__name__}: nieznane parametry {sorted(unknown)}')
    kwargs = {}
    for name, value in data.items():
        default = getattr(base, name)
        if is_dataclass(default):
            kwargs[name] = _merge(default, value)
        elif isinstance(default, list):
            kwargs[name] = [type(default[0])(v) for v in value] if default else list(value)
        elif isinstance(default, bool):
            kwargs[name] = value if isinstance(value, bool) else str(value).lower() == 'true'
        else:
            kwargs[name] = type(default)(value)
    return replace(base, **kwargs)


def from_dict(data: dict) -> SearchParams:
    """Zbuduj SearchParams z zagnieżdżonego słownika (np. YAML); brakujące pola = domyślne."""
    return _merge(SearchParams(), data or {})


def flatten(params: SearchParams) -> dict:
    """{'camera.hfov_deg': 96.8, ...} - do deklarowania parametrów ROS."""
    out = {}
    for section in fields(params):
        sub = getattr(params, section.name)
        for f in fields(sub):
            out[f'{section.name}.{f.name}'] = getattr(sub, f.name)
    return out


def from_flat(flat: dict) -> SearchParams:
    nested = {}
    for key, value in flat.items():
        section, name = key.split('.', 1)
        nested.setdefault(section, {})[name] = value
    return from_dict(nested)


def with_section(params: SearchParams, section: str, **changes) -> SearchParams:
    """Kopia z podmienionymi polami jednej sekcji (wygodne w testach i ablacjach)."""
    return replace(params, **{section: replace(getattr(params, section), **changes)})
