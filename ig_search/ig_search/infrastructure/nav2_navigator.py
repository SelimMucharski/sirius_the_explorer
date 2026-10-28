"""
Port Navigator: Nav2 `navigate_through_poses` + dokładny dojazd (fine positioning).

Nav2 kończy cel z tolerancją 0,4 m / 0,5 rad (nav2_params), a obszar chwytu ma ~0,2 m
szerokości. Przy `precise=True` po sukcesie Nav2 działa prosty regulator na /cmd_vel
z pozycją z TF. Nav2 potrafi też krążyć tuż przy celu bez sukcesu ani porażki - wtedy
(bliżej niż `handover_distance` i bez postępu przez `stall_time`) cel Nav2 jest anulowany,
a dojazd kończy regulator.

Z `reach` (dojazd do próbki) cel jest osiągnięty, gdy tylko próbka jest w obszarze chwytu -
także w trakcie jazdy Nav2 (`reach.stop_early`). Regulator końcowy nie ustawia łazika
centralnie przed próbką: obraca go tylko, gdy próbka jest poza zakresem kątów, i jedzie
prosto, by poprawić odległość (obrót w miejscu napędem burtowym przesuwa łazik, na zboczu
nawet o 0,3 m). Przy
`reach.direct` Nav2 jest pomijany. Bez `reach`: obrót do punktu -> jazda prosto -> obrót
do kursu.

Cel (bez `precise`) bliżej niż `near_goal` też omija Nav2: sam obrót w miejscu do kursu celu
(tolerancja `near_yaw_tol`), bez podjazdu - z tak bliska widok jest praktycznie ten sam.
Regulated Pure Pursuit nie radził sobie ze ścieżką krótszą niż odległość wyprzedzenia: łazik
0,35 m od pozy obserwacji o innym kursie drgał w miejscu ~70 s (Nav2: „Failed to make
progress” ×4), aż przerwało to wykrycie utknięcia. Regulator z podjazdem też zawodził: przy
0,5 m przełączał jazdę przód/tył (kierunek odwracał się o 180°) i kręcił się do limitu czasu.

Komendy prędkości mają zerowy znacznik czasu - sterownik napędu wstawia wtedy własny bieżący
czas. Znacznik z zegara węzła bywał o ~1 s starszy od zegara Gazebo (zajęty executor) i sterownik
odrzucał takie komendy jako przeterminowane.
"""
import math

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, TwistStamped
from ig_search.application.ports import Navigator, Status
from ig_search.infrastructure.ros_conversions import yaw_to_quat
from nav2_msgs.action import NavigateThroughPoses
from rclpy.action import ActionClient


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Nav2Navigator(Navigator):

    def __init__(self, node, pose_fn, map_frame='map', action='navigate_through_poses',
                 cmd_vel='/cmd_vel', xy_tol=0.05, yaw_tol=0.05, v_max=0.15, w_max=0.4,
                 fine_timeout=25.0, handover_distance=1.0, stall_time=8.0,
                 near_goal=0.6, near_yaw_tol=0.25, near_w=0.6):
        """pose_fn() -> (x, y, psi) w ramce map albo None."""
        self.node = node
        self.pose_fn = pose_fn
        self.map_frame = map_frame
        self.client = ActionClient(node, NavigateThroughPoses, action)
        self.cmd_pub = node.create_publisher(TwistStamped, cmd_vel, 10)
        self.xy_tol, self.yaw_tol = xy_tol, yaw_tol
        self.v_max, self.w_max = v_max, w_max
        self.fine_timeout = fine_timeout
        self.handover_distance, self.stall_time = handover_distance, stall_time
        self.near_goal, self.near_w = near_goal, near_w
        self._near_tol = (near_goal, near_yaw_tol)  # w promieniu near_goal - tylko obrót
        self._tol = (xy_tol, yaw_tol)               # tolerancje regulatora dla bieżącego celu
        self._reach = None
        self._best = (float('inf'), 0.0)            # (najmniejsza odległość, kiedy)
        self._st = Status.IDLE
        self._seq = 0
        self._goal = None
        self._precise = False
        self._handle = None
        self._fine_t0 = None
        self._yaw_phase = False
        self._backup_until = None                   # koniec cofania po utknięciu [s]
        self.ticks = 0                              # wywołania tick() (diagnostyka)

    # ------------------------------------------------------------ port
    def go_to(self, x, y, yaw, precise=False, via=(), reach=None):
        self._seq += 1
        seq = self._seq
        self._goal, self._precise, self._reach = (x, y, yaw), precise, reach
        self._fine_t0 = None
        self._tol = (self.xy_tol, self.yaw_tol)
        self._best = (float('inf'), self._now())
        self._st = Status.ACTIVE
        if not precise and reach is None:
            q = self.pose_fn()
            if q is not None and math.hypot(x - q[0], y - q[1]) < self.near_goal:
                if self._handle is not None:        # wywłaszczony cel Nav2
                    self._seq += 1
                    self._handle.cancel_goal_async()
                    self._handle = None
                self._tol = self._near_tol
                self._fine_t0 = self._now()
                self._yaw_phase = True
                self.node.get_logger().info(
                    f'cel {math.hypot(x - q[0], y - q[1]):.2f} m od łazika - bez Nav2')
                return
        q = self.pose_fn() if reach is not None else None
        if q is not None and ((reach.stop_early and reach.contains(q)) or reach.direct):
            if self._handle is not None:            # wywłaszczony cel Nav2
                self._handle.cancel_goal_async()
                self._handle = None
            if reach.stop_early and reach.contains(q):
                self._finish()
                return
            self._fine_t0 = self._now()             # blisko - bez Nav2
            self.node.get_logger().info('dojazd w zasięg bez Nav2')
            return
        if not self.client.server_is_ready():
            self.node.get_logger().warn('navigate_through_poses niedostępne',
                                        throttle_duration_sec=5.0)
            self._st = Status.FAILED
            return
        # Zawsze navigate_through_poses (także bez punktów pośrednich): bt_navigator odrzuca
        # cel jednego nawigatora, gdy drugi jest aktywny, więc mieszanie navigate_to_pose
        # i navigate_through_poses dawało serie odrzuconych celów.
        pts = list(via) + [(x, y)]
        poses = [self._pose(px, py, math.atan2(pts[i + 1][1] - py, pts[i + 1][0] - px))
                 for i, (px, py) in enumerate(pts[:-1])] + [self._pose(x, y, yaw)]
        fut = self.client.send_goal_async(NavigateThroughPoses.Goal(poses=poses))
        fut.add_done_callback(lambda f: self._on_accepted(f, seq))   # nowy cel wywłaszcza

    def _pose(self, x, y, yaw) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = self.map_frame
        ps.header.stamp = self.node.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = float(x), float(y)
        (ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z,
         ps.pose.orientation.w) = yaw_to_quat(yaw)
        return ps

    def cancel(self):
        self._seq += 1
        if self._handle is not None:
            self._handle.cancel_goal_async()
            self._handle = None
        if self._fine_t0 is not None or self._backup_until is not None:
            self._stop()
        self._fine_t0 = None
        self._backup_until = None
        self._goal = None
        self._reach = None
        self._st = Status.IDLE

    def back_up(self, distance: float):
        self.cancel()
        self._backup_until = self._now() + distance / self.v_max
        self._st = Status.ACTIVE
        self.node.get_logger().warn(f'cofanie {distance:.1f} m (utknięcie)')

    def status(self) -> Status:
        return self._st

    # ------------------------------------------------------------ Nav2
    def _on_accepted(self, fut, seq):
        if seq != self._seq:
            return
        handle = fut.result()
        if not handle.accepted:
            self.node.get_logger().warn('cel odrzucony przez Nav2')
            self._st = Status.FAILED
            return
        self._handle = handle
        handle.get_result_async().add_done_callback(lambda f: self._on_result(f, seq))

    def _on_result(self, fut, seq):
        status = fut.result().status
        self.node.get_logger().debug(f'Nav2 wynik {status} (cel #{seq}, bieżący #{self._seq})')
        if seq != self._seq:
            return                                  # wynik wywłaszczonego celu
        self._handle = None
        if status != GoalStatus.STATUS_SUCCEEDED:
            self._st = Status.FAILED
        elif self._precise:
            self._fine_t0 = self._now()             # dalej: dokładny dojazd
            self._yaw_phase = False
            self.node.get_logger().info('Nav2 u celu - dokładny dojazd')
        else:
            self._st = Status.SUCCEEDED

    # ------------------------------------------------------------ dokładny dojazd
    def _now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _send(self, v, w):
        m = TwistStamped()                          # znacznik 0 - czas wstawia sterownik napędu
        m.header.frame_id = 'base_link'
        m.twist.linear.x, m.twist.angular.z = float(v), float(w)
        self.cmd_pub.publish(m)

    def _stop(self):
        self._send(0.0, 0.0)

    def _finish(self, ok_msg=None):
        self._stop()
        self._fine_t0 = None
        if ok_msg:
            self.node.get_logger().warn(ok_msg)
        self._st = Status.SUCCEEDED

    def _check_stall(self):
        """Nav2 blisko celu bez postępu -> przejęcie przez regulator dokładnego dojazdu."""
        if not self._precise or self._handle is None or self._goal is None:
            return
        q = self.pose_fn()
        if q is None:
            return
        d = math.hypot(self._goal[0] - q[0], self._goal[1] - q[1])
        now = self._now()
        if d < self._best[0] - 0.05:
            self._best = (d, now)
        elif d < self.handover_distance and now - self._best[1] > self.stall_time:
            self.node.get_logger().info(
                f'Nav2 bez postępu {d:.2f} m od celu - dokładny dojazd przejmuje')
            self._seq += 1                          # ignoruj wynik anulowanego celu
            self._handle.cancel_goal_async()
            self._handle = None
            self._fine_t0 = now
            self._yaw_phase = False

    def tick(self):
        """
        Krok regulatora dokładnego dojazdu; wołany przez węzeł (timer węzła, ~20 Hz).

        Adapter nie ma własnego timera: timery tworzone przez adaptery w węźle lifecycle
        z czasem symulacji nie odpalały (0 wywołań), timery samego węzła działały.
        """
        self.ticks += 1
        if self._backup_until is not None:
            if self._now() < self._backup_until:
                self._send(-self.v_max, 0.0)
                return
            self._backup_until = None
            self._finish()
            return
        self._check_stall()
        if (self._reach is not None and self._st == Status.ACTIVE
                and (self._reach.stop_early or self._fine_t0 is not None)):
            q = self.pose_fn()
            if q is not None and self._reach.contains(q):
                if self._handle is not None:
                    self._seq += 1                  # ignoruj wynik anulowanego celu
                    self._handle.cancel_goal_async()
                    self._handle = None
                r, b = self._reach.polar(q)
                self.node.get_logger().info(
                    f'próbka w zasięgu ramienia: {r:.2f} m, {math.degrees(b):.0f}° - stop')
                self._finish()
                return
        if self._fine_t0 is None or self._goal is None:
            return
        q = self.pose_fn()
        if q is None:
            return
        if self._reach is not None:
            self._reach_step(q)
            return
        gx, gy, gyaw = self._goal
        dx, dy = gx - q[0], gy - q[1]
        dist = math.hypot(dx, dy)
        self.node.get_logger().info(
            f'dokładny dojazd: {dist:.2f} m, kurs {math.degrees(_wrap(gyaw - q[2])):.0f}°',
            throttle_duration_sec=2.0)
        if self._now() - self._fine_t0 > self.fine_timeout:
            self._finish(f'dokładny dojazd: limit czasu, błąd {dist:.2f} m')
            return
        # histereza: obrót w miejscu przy napędzie burtowym przesuwa łazik (na zboczu
        # nawet o ~0,1 m); pas osiągalności ramienia ma ±0,15 m, więc dopiero 0,25 m wraca
        xy_tol, yaw_tol = self._tol
        near = self._tol == self._near_tol
        if self._yaw_phase and not near and dist > max(3 * xy_tol, 0.25):
            self._yaw_phase = False
        if not self._yaw_phase and dist <= xy_tol:
            self._yaw_phase = True
        if not self._yaw_phase:
            bearing = math.atan2(dy, dx)
            reverse = abs(_wrap(bearing - q[2])) > math.pi / 2 and dist < 0.5
            heading = _wrap(bearing + math.pi) if reverse else bearing
            e = _wrap(heading - q[2])
            if abs(e) > 0.1:                         # najpierw obrót do punktu
                self._send(0.0, max(-self.w_max, min(self.w_max, 1.5 * e)))
            else:
                v = max(0.04, min(self.v_max, 0.8 * dist)) * (-1.0 if reverse else 1.0)
                self._send(v, max(-self.w_max, min(self.w_max, 1.5 * e)))
            return
        e = _wrap(gyaw - q[2])
        if abs(e) > yaw_tol:                        # wolniej: mniejszy poślizg przy obrocie
            w_lim = self.near_w if near else 0.5 * self.w_max
            w = max(-w_lim, min(w_lim, 1.5 * e))
            self._send(0.0, w if abs(w) > 0.08 else math.copysign(0.08, e))
            return
        self._finish()

    def _reach_step(self, q):
        """Regulator „w zasięg”: najmniejszy obrót, potem jazda prosto (bez kursu końcowego)."""
        rp = self._reach.params
        r, b = self._reach.polar(q)
        self.node.get_logger().info(
            f'dojazd w zasięg: {r:.2f} m od podstawy ramienia, {math.degrees(b):.0f}°',
            throttle_duration_sec=2.0)
        if self._now() - self._fine_t0 > self.fine_timeout:
            self._finish(f'dojazd w zasięg: limit czasu, {r:.2f} m, {math.degrees(b):.0f}°')
            return
        lo = rp.workspace_r_min + rp.reach_margin
        hi = self._reach.r_max - rp.reach_margin
        a_max = math.radians(rp.workspace_half_angle_deg - rp.reach_angle_margin_deg)
        # odległość poprawia się jazdą prosto tylko, gdy próbka jest dość blisko osi łazika
        limit = a_max if lo <= r <= hi else min(a_max, math.radians(45.0))
        if abs(b) > limit:                          # obrót tylko do granicy zakresu kątów
            e = b - math.copysign(0.8 * limit, b)
            w = max(-0.5 * self.w_max, min(0.5 * self.w_max, 1.5 * e))
            self._send(0.0, w if abs(w) > 0.08 else math.copysign(0.08, e))
            return
        mid = 0.5 * (lo + hi)
        v = 0.8 * (r - mid) / max(math.cos(b), 0.5)  # do przodu, gdy próbka za daleko
        v = math.copysign(max(0.04, min(self.v_max, abs(v))), v)
        self._send(v, 0.0)
