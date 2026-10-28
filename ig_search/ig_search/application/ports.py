"""
Porty warstwy aplikacji: to, czego misja potrzebuje od świata zewnętrznego.

Implementacje ROS 2 (Nav2, MoveIt, publikacja zgłoszeń) są w warstwie infrastructure;
w testach - atrapy.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from ig_search.domain.params import ReportParams
from ig_search.domain.report_rule import DZ_FLAT, in_workspace, workspace_polar, workspace_r_max
from ig_search.domain.visibility import CameraPose
import numpy as np


class Status(Enum):
    IDLE = 0
    ACTIVE = 1
    SUCCEEDED = 2
    FAILED = 3


@dataclass(frozen=True)
class SamplePose:
    """
    Poza próbki z detektora: środek i oś (od stożka do głowicy) w ramce mapy.

    local_* - to samo w ramce lokalnej odometrii (bez korekt GNSS): do chwytu, gdy łazik stoi
    między pomiarem a ruchem ramienia (skok korekty GNSS przeniósłby się na pozycję palców).
    ground_dz - grunt pod próbką względem gruntu pod łazikiem (mapa wysokości): wysokość osi
    do chwytu jest ograniczana do pasa nad gruntem (głębia detektora z bliska bywa zawyżona).
    """

    position: np.ndarray
    axis: np.ndarray
    local_position: Optional[np.ndarray] = None
    local_axis: Optional[np.ndarray] = None
    ground_dz: Optional[float] = None


@dataclass
class Frame:
    """Jedna klatka detektora: czas, poza kamery z TF na ten czas, detekcje w ramce map."""

    stamp: float
    camera: CameraPose
    detections: np.ndarray                      # (K, 3); K może być 0
    poses: List[SamplePose] = field(default_factory=list)


@dataclass
class Report:
    index: int                                  # n = 1, 2, 3
    x: float
    y: float
    z: float
    probability: float
    threshold: float
    robot: tuple                                # q = (x, y, psi)
    stamp: float
    sample_pose: Optional[SamplePose] = None
    pregrasp_reached: bool = False
    grasped: bool = False


@dataclass(frozen=True)
class ReachGoal:
    """
    Dojazd w zasięg ramienia: punkt (x, y) ma się znaleźć w R(q) z zapasem.

    Kurs łazika jest dowolny - liczy się tylko położenie punktu względem podstawy ramienia,
    więc nie ma końcowego obrotu do zadanego kursu (obrót w miejscu przesuwa łazik).
    direct     - bez planera tras, od razu korekta z bieżącej pozy (punkt tuż obok łazika);
    stop_early - koniec, gdy tylko punkt jest w R(q), także w trakcie jazdy; bez tego jazda
                 do pozy celu (np. ponowne podejście - inny punkt widzenia), a korekta
                 końcowa jak wyżej, bez obrotu do kursu.
    """

    x: float
    y: float
    params: ReportParams
    direct: bool = False
    stop_early: bool = True
    dz: float = DZ_FLAT             # wysokość środka próbki względem base_link (zasięg ramienia)

    def contains(self, q) -> bool:
        return bool(in_workspace(q, [self.x], [self.y], self.params, margin=True,
                                 dz=self.dz)[0])

    @property
    def r_max(self) -> float:
        return float(workspace_r_max(self.params, self.dz))

    def polar(self, q):
        """(odległość od podstawy ramienia, kierunek względem kursu łazika) punktu."""
        r, b = workspace_polar(q, [self.x], [self.y], self.params)
        return float(r[0]), float(b[0])


class Navigator(ABC):

    @abstractmethod
    def go_to(self, x: float, y: float, yaw: float, precise: bool = False, via=(),
              reach: Optional[ReachGoal] = None):
        """
        Wysyła cel (wywłaszcza poprzedni); status od razu ACTIVE.

        precise - po dotarciu korekta pozycji do kilku cm (dojazd do próbki w zasięgu ramienia),
        via     - punkty pośrednie (x, y) trasy po terenie przejezdnym,
        reach   - koniec, gdy tylko reach.contains(q) (także w trakcie jazdy, bez dojazdu
                  do pozy); korekta końcowa tylko do wprowadzenia punktu w R(q).
        """

    @abstractmethod
    def cancel(self):
        ...

    @abstractmethod
    def back_up(self, distance: float):
        """
        Cofa łazik prosto o `distance` (wywłaszcza cel); status ACTIVE do końca cofania.

        Na czas, nie na pozę: przy utknięciu (koła buksują) poza z EKF+GNSS jest niewiarygodna.
        """

    @abstractmethod
    def status(self) -> Status:
        ...


class Manipulator(ABC):

    @abstractmethod
    def scan(self, x: float, y: float, dz: float):
        """
        Ustawia kamerę na nadgarstku tak, by patrzyła z góry na punkt (skan weryfikujący).

        dz - wysokość gruntu pod (x, y) względem gruntu pod łazikiem (z mapy wysokości).
        Względna, bo z ramki map zależy od lokalizacji: z GNSS+EKF (2D) base_link ma w niej
        zawsze z = 0, z prawdziwej pozy Gazebo - wysokość bezwzględną.
        """

    @abstractmethod
    def pregrasp(self, sample: SamplePose):
        """Ustawia chwytak w pozycji przed chwytem nad próbką."""

    @abstractmethod
    def grasp(self, sample: SamplePose):
        """
        Z pozycji przed chwytem: opuszczenie pionowo, zamknięcie chwytaka, podniesienie.

        SUCCEEDED, gdy po podniesieniu próbka jest w chwytaku. Następne stow() przenosi ją
        do pozycji transportowej (próbka zebrana).
        """

    @abstractmethod
    def stow(self):
        """Ramię w pozycji transportowej (nie zasłania kamery przedniej)."""

    @abstractmethod
    def status(self) -> Status:
        ...


class Reporter(ABC):

    @abstractmethod
    def report(self, report: Report):
        ...
