"""
Port Manipulator: skan kamerą na nadgarstku i pozycja przed chwytem przez move_group.

Cel w przestrzeni stawów z numerycznego IK (arm_kinematics, w wątku - trwa 1-2 s),
trajektorię bezkolizyjną (z gruntem ze sceny planowania) planuje i wykonuje move_group.

Przed pozycją przed chwytem próbka z pomiaru nadgarstka trafia do sceny planowania jako walec
(`probe_object`). Scena ma ramkę łazika (bez stawu wirtualnego), więc obiekt jest ważny tylko
do odjazdu - usuwany przy następnym skanie.

Chwyt (grasp): pionowe opuszczenie i podniesienie wysyłane wprost do manip_controller (krótki
odcinek o znanym przebiegu; MoveIt odrzuciłby go jako kolizję palców z próbką i gruntem),
między nimi zamknięcie chwytaka. Próbka jest w chwytaku, gdy szczęki zatrzymały się na niej
(rozstaw szczęk ≈ średnica próbki). Trzymana próbka jest w scenie dołączona do chwytaka; po dojściu
do pozycji transportowej - komunikat ~/sample_stored (położenie chwytaka w ramce map),
na który symulacja usuwa próbkę (ig_search_experiments/sample_store).

Po zamknięciu na próbce szczęki są przytrzymywane w osiągniętym rozwarciu, a ~/sample_grasped
(punkt między szczękami) pozwala symulacji przyspawać próbkę do chwytaka; podniesienie czeka
na ~/sample_attached (najwyżej ATTACH_TIMEOUT - bez symulacji tylko tyle opóźnienia).
Otwarcie chwytaka z próbką - ~/sample_released.
"""
import threading

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped, Pose
from ig_search.application.ports import Manipulator, SamplePose, Status
from ig_search.infrastructure.arm_kinematics import ARM_JOINTS, ArmIK, FINGERTIP_OFFSET, STOW
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (AttachedCollisionObject, CollisionObject, Constraints,
                             JointConstraint, MoveItErrorCodes, PlanningScene, RobotState)
from moveit_msgs.srv import GetStateValidity
import numpy as np
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

GROUND_PLANE_Z = 0.01               # [m] wierzch gruntu w scenie MoveIt (ramka base_link)
ARM_ROOT_X = 0.1145                 # [m] podstawa ramienia przed base_link (ReportParams)
GRIPPER_JOINTS = ['joint_gripper_left', 'joint_gripper_right']
GRIPPER_OPEN = 0.0326               # [m] rozwarcie szczęki (stan `open` w SRDF)
GRIPPER_CLOSED = 0.0
# [rad] tolerancja celu w stawach; 0,01 rad dawało do ~1 cm i ~1° odchyłki chwytaka od celu IK
JOINT_TOL = 0.002
PROBE_ID = 'target_probe'
GRASP = 'chwyt'
GRASP_STEPS = 4                     # punkty pionowego odcinka (opuszczenie / podniesienie)
CLOSE_TIME = 2.5                    # [s] zamykanie chwytaka na próbce
ATTACH_TIMEOUT = 5.0                # [s] czekanie na potwierdzenie spawu próbki (symulacja)
# [m] szczęki przytrzymane tyle ciaśniej niż rozwarcie na próbce: docisk, ale bez dojścia do
# zamknięcia, gdy symulacja na chwilę usuwa próbkę (podmiana na przyspawaną kopię)
HOLD_SQUEEZE = 0.001
# [m] punkt chwytu: tyle nad końcówkami palców (na osi tcp), w obrębie szczęk
GRIP_POINT_DEPTH = 0.02
# [m] rozstaw szczęk (suma rozwarć) z próbką ⌀0,03 m; szczęki są sterowane osobno, więc para
# przesuwa się w bok razem z próbką (np. 0,0297 + 0,0004) - liczy się tylko suma
HOLD_GAP = (0.02, 0.045)
# próbka (z zapasem 1 cm, długość 22 cm) sięga w modelu do kamery na nadgarstku
TOUCH_LINKS = ['gripper_left', 'gripper_right', 'tcp_connector', 'wrist_camera']
PROBE_RADIUS = 0.015 + 0.01         # [m] walec próbki (probe.sdf) + zapas na błąd pomiaru
PROBE_LENGTH = 0.20 + 0.02
# [m] pas wysokości osi próbki nad gruntem pod nią do chwytu (promień 0,015 m; mapa wysokości
# p95 4 cm): z pomiaru z bliska bywało 8 cm za wysoko - palce zamykały się nad próbką
AXIS_ABOVE_GROUND = (0.0, 0.045)


def _quat_z_to(axis):
    """Kwaternion (x, y, z, w) obracający oś z na `axis`."""
    a = np.asarray(axis, float) / max(np.linalg.norm(axis), 1e-9)
    c = float(a[2])
    if c < -0.999999:
        return 1.0, 0.0, 0.0, 0.0
    v = np.cross([0.0, 0.0, 1.0], a)
    q = np.r_[v, 1.0 + c]
    q /= np.linalg.norm(q)
    return tuple(float(x) for x in q)


class MoveItManipulator(Manipulator):

    def __init__(self, node, base_pose_fn, scan_height=0.6, pregrasp_height=0.12,
                 target_height=0.03, group='arm', action='move_action',
                 planning_time=5.0, velocity_scaling=0.5, probe_object=True,
                 base_frame='base_link', grasp_depth=0.005, map_frame='map',
                 carry_scaling=0.2, grasp_step_time=1.2, local_base_fn=None):
        """base_pose_fn() -> macierz 4x4 base_link w ramce map albo None."""
        self.node = node
        self.base_pose_fn = base_pose_fn
        self.scan_height = scan_height
        self.pregrasp_height = pregrasp_height
        self.target_height = target_height
        self.group = group
        self.planning_time = planning_time
        self.velocity_scaling = velocity_scaling
        self.client = ActionClient(node, MoveGroup, action)
        self._ik = None
        self._q = np.array(STOW, float)
        self._st = Status.IDLE
        self._seq = 0
        self._lock = threading.Lock()
        self._ready_goal = None                     # (seq, stawy, opis) z wątku IK
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String, '/robot_description', self._on_urdf, latched)
        node.create_subscription(JointState, '/joint_states', self._on_joints, 10)
        # akcja, nie topik ~/joint_trajectory: kontroler chwytaka w symulacji ignorował
        # trajektorie z topiku (szczęki zostawały zamknięte), akcję wykonuje
        self._gripper_client = ActionClient(node, FollowJointTrajectory,
                                            '/gripper_controller/follow_joint_trajectory')
        self.probe_object = probe_object
        self.base_frame = base_frame
        self._scene_pub = node.create_publisher(PlanningScene, '/planning_scene', 10)
        self._has_probe = False
        self.grasp_depth = grasp_depth
        self.local_base_fn = local_base_fn          # base_link w ramce odometrii (do chwytu)
        self.carry_scaling = carry_scaling          # skalowanie prędkości z próbką w chwytaku
        self.grasp_step_time = grasp_step_time
        self.map_frame = map_frame
        self._jaws = np.zeros(2)
        self._pregrasp_ctx = None                   # (p, a, down, wysokość) w base_link
        self._grasp = None                          # stan sekwencji chwytu
        self._holding = False
        self._traj = ActionClient(node, FollowJointTrajectory,
                                  '/manip_controller/follow_joint_trajectory')
        self._stored_pub = node.create_publisher(PointStamped, '~/sample_stored', 10)
        self._grasped_pub = node.create_publisher(PointStamped, '~/sample_grasped', 10)
        self._released_pub = node.create_publisher(PointStamped, '~/sample_released', 10)
        node.create_subscription(Bool, '~/sample_attached', self._on_attached, 10)
        self._attach_ack = None                     # odpowiedź symulacji na ~/sample_grasped
        self._sim_held = False                      # ~/sample_grasped wysłane, nie zebrana
        # diagnostyka nieudanego planowania: czy sam cel jest poprawny w scenie move_group
        self._validity = node.create_client(GetStateValidity, '/check_state_validity')
        self._joint_state = None
        self._goal_q = {}                           # seq -> stawy celu wysłanego do move_group
        self.ticks = 0

    # ------------------------------------------------------------ stan robota
    def _on_urdf(self, msg):
        if self._ik is None:
            self._ik = ArmIK(msg.data)

    def _on_joints(self, msg):
        self._joint_state = msg
        idx = {n: i for i, n in enumerate(msg.name)}
        if all(j in idx for j in ARM_JOINTS):
            self._q = np.array([msg.position[idx[j]] for j in ARM_JOINTS])
        if all(j in idx for j in GRIPPER_JOINTS):
            self._jaws = np.array([msg.position[idx[j]] for j in GRIPPER_JOINTS])

    def _to_base(self, p_map, v_map=None, with_down=False):
        T = self.base_pose_fn()
        if T is None:
            return (None, None, None) if with_down else (None, None)
        Ti = np.linalg.inv(T)
        p = (Ti @ np.r_[p_map, 1.0])[:3]
        v = None if v_map is None else Ti[:3, :3] @ np.asarray(v_map, float)
        if with_down:
            return p, v, Ti[:3, :3] @ np.array([0.0, 0.0, -1.0])
        return p, v

    def _sample_to_base(self, sample: SamplePose):
        """
        Próbka w base_link (+ pion w dół): z ramki odometrii, jeśli pomiar ją ma.

        Z `ground_dz` wysokość osi przycięta do pasa nad gruntem pod próbką (base_link leży na
        gruncie pod łazikiem). Tylko z: x, y z nadgarstka są trafne (błąd zgłoszeń 0,2-2,4 cm
        przy błędzie z do 15 cm), przesuwanie wzdłuż promienia kamery psuło je (DESIGN §6).
        """
        T = self.local_base_fn() if (self.local_base_fn is not None
                                     and sample.local_position is not None) else None
        if T is None:
            p, a, down = self._to_base(sample.position, sample.axis, with_down=True)
        else:
            Ti = np.linalg.inv(T)
            p, a, down = ((Ti @ np.r_[sample.local_position, 1.0])[:3],
                          Ti[:3, :3] @ sample.local_axis, Ti[:3, :3] @ np.array([0.0, 0.0, -1.0]))
        if p is not None and sample.ground_dz is not None:
            lo, hi = (sample.ground_dz + d for d in AXIS_ABOVE_GROUND)
            if not lo <= p[2] <= hi:
                self.node.get_logger().info(
                    f'chwyt: oś próbki {100 * p[2]:+.1f} cm (pomiar), grunt pod nią '
                    f'{100 * sample.ground_dz:+.1f} cm - przycięta do pasa nad gruntem')
                p = np.r_[p[:2], np.clip(p[2], lo, hi)]
        return p, a, down

    # ------------------------------------------------------------ port
    def _probe_to_scene(self, p, a):
        """Walec próbki (środek p, oś a w base_link) do sceny planowania; p None - usuń."""
        obj = CollisionObject(id=PROBE_ID)
        obj.header.frame_id = self.base_frame
        if p is None:
            obj.operation = CollisionObject.REMOVE
        else:
            obj.operation = CollisionObject.ADD     # ADD istniejącego id zastępuje obiekt
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (float(v) for v in p)
            (pose.orientation.x, pose.orientation.y, pose.orientation.z,
             pose.orientation.w) = _quat_z_to(a)
            obj.primitives = [SolidPrimitive(type=SolidPrimitive.CYLINDER,
                                             dimensions=[PROBE_LENGTH, PROBE_RADIUS])]
            obj.primitive_poses = [pose]
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects = [obj]
        self._scene_pub.publish(scene)
        self._has_probe = p is not None

    def scan(self, x, y, dz):
        if self._has_probe:                         # łazik odjechał - obiekt nieaktualny
            self._probe_to_scene(None, None)

        def solve():
            T = self.base_pose_fn()
            if T is None:
                return None
            z = T[2, 3] + dz                        # grunt pod (x, y) w ramce map
            target, _ = self._to_base(np.array([x, y, z + self.target_height]))
            ground, _ = self._to_base(np.array([x, y, z]))
            if target is None or ground is None:
                return None
            return self._ik.look_at(target, self.scan_height, q0=self._q, ground_z=ground[2])
        self._start(solve, f'skan ({x:.2f}, {y:.2f}, dz {100 * dz:+.0f} cm)')

    def _gripper(self, opening, sec: float = 1.5):
        """
        Otwiera / zamyka chwytak (trajektoria do gripper_controller, równolegle z ramieniem).

        `opening`: rozwarcie obu szczęk albo para (lewa, prawa).
        """
        msg = JointTrajectory(joint_names=GRIPPER_JOINTS)
        msg.points.append(JointTrajectoryPoint(
            positions=[float(v) for v in np.broadcast_to(opening, 2)],
            time_from_start=Duration(seconds=sec).to_msg()))
        if not self._gripper_client.server_is_ready():
            self.node.get_logger().warn('gripper_controller niedostępny')
            return
        self._gripper_client.send_goal_async(FollowJointTrajectory.Goal(trajectory=msg))

    def pregrasp(self, sample: SamplePose):
        self._release()
        self._gripper(GRIPPER_OPEN)

        def solve():
            p, a, down = self._sample_to_base(sample)
            if p is None:
                return None
            if self.probe_object:
                self._probe_to_scene(p, a)
            # scena MoveIt ma płaski grunt na poziomie base_link (add_ground.py, margines
            # 0,01 m); na zboczu próbka bywa niżej - końcówki palców ≥ 4 cm nad tą płaszczyzną
            height = max(self.pregrasp_height,
                         (GROUND_PLANE_Z + 0.04 - p[2]) / max(-down[2], 0.5))
            self._pregrasp_ctx = (p, a, down, height)
            return self._ik.pregrasp(p, a, height, down=down, q0=self._q)
        self._start(solve, 'pozycja przed chwytem')

    def grasp(self, sample: SamplePose):
        ctx = self._pregrasp_ctx

        def solve():
            if ctx is None:
                return None
            p, a, down, height = ctx
            q_pre = self._q.copy()
            heights = np.linspace(height, -self.grasp_depth, GRASP_STEPS + 1)[1:]
            # położenie z końcowego celu (mediana pomiarów z bliska), gdy po wyczerpaniu korekt
            # różni się od pozycji przed chwytem; oś szczęk i prześwit - z pozycji przed chwytem
            # (inna oś z pomiaru to obrót nadgarstka, którego pionowy odcinek nie robi)
            targets = [p]
            ps, _, _ = self._sample_to_base(sample) if sample is not None else (None,) * 3
            if ps is not None:
                targets.insert(0, ps)
            for target in targets:
                down_path = self._ik.vertical_path(target, a, heights, down, q_pre)
                if down_path is not None:
                    return {'down': down_path, 'up': down_path[-2::-1] + [q_pre]}
            rho = float(np.hypot(p[0] - ARM_ROOT_X, p[1]))
            beta = float(np.degrees(np.arctan2(p[1], p[0] - ARM_ROOT_X)))
            self.node.get_logger().warn(
                f'chwyt: brak IK pionowego odcinka - cel w base_link '
                f'({p[0]:.3f}, {p[1]:.3f}, {p[2]:+.3f}), ρ={rho:.3f} m, β={beta:.0f}°, '
                f'oś ({a[0]:+.2f}, {a[1]:+.2f}, {a[2]:+.2f}), prześwit {100 * height:.1f} cm')
            return None
        if self._has_probe:                         # palce obejmą próbkę - nie przeszkoda
            self._probe_to_scene(None, None)
        self._start(solve, GRASP)

    def stow(self):
        self._gripper(GRIPPER_CLOSED)
        self._start(lambda: np.array(STOW, float), 'pozycja transportowa')

    def status(self) -> Status:
        return self._st

    # ------------------------------------------------------------ wykonanie
    def _start(self, solve, what):
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._ready_goal = None
        self._grasp = None
        self._st = Status.ACTIVE
        if self._ik is None:
            self.node.get_logger().error('brak /robot_description - ramię niedostępne')
            self._st = Status.FAILED
            return

        def work():
            q = solve()
            with self._lock:
                if seq == self._seq:
                    self._ready_goal = (seq, q, what)
        threading.Thread(target=work, daemon=True).start()

    def tick(self):
        """Wysyła cel z wątku IK, gdy gotowy; wołany przez węzeł (jak Nav2Navigator.tick)."""
        self.ticks += 1
        self._grasp_tick()
        with self._lock:
            goal, self._ready_goal = self._ready_goal, None
        if goal is None:
            return
        seq, q, what = goal
        if q is None:
            self.node.get_logger().warn(f'{what}: brak rozwiązania IK')
            self._st = Status.FAILED
            return
        if what == GRASP:
            self._grasp = {'seq': seq, 'phase': 'down', 'up': q['up'], 't0': None}
            self.node.get_logger().info('ramię: chwyt - opuszczenie')
            self._send_path(q['down'], seq)
            return
        if not self.client.server_is_ready():
            self.node.get_logger().warn('move_group niedostępny')
            self._st = Status.FAILED
            return
        g = MoveGroup.Goal()
        r = g.request
        r.group_name = self.group
        r.num_planning_attempts = 3
        r.allowed_planning_time = self.planning_time
        scaling = self.carry_scaling if self._holding else self.velocity_scaling
        r.max_velocity_scaling_factor = scaling
        r.max_acceleration_scaling_factor = scaling
        r.goal_constraints = [Constraints(joint_constraints=[
            JointConstraint(joint_name=j, position=float(v), tolerance_above=JOINT_TOL,
                            tolerance_below=JOINT_TOL, weight=1.0)
            for j, v in zip(ARM_JOINTS, q)])]
        g.planning_options.plan_only = False
        g.planning_options.replan = True
        g.planning_options.replan_attempts = 2
        self.node.get_logger().info(f'ramię: {what}')
        self._goal_q = {seq: np.asarray(q, float)}
        fut = self.client.send_goal_async(g)
        fut.add_done_callback(lambda f: self._on_accepted(f, seq, what))

    def _on_accepted(self, fut, seq, what):
        h = fut.result()
        if seq != self._seq:
            return
        if not h.accepted:
            self.node.get_logger().warn(f'{what}: cel odrzucony przez move_group')
            self._st = Status.FAILED
            return
        h.get_result_async().add_done_callback(lambda f: self._on_result(f, seq, what))

    def _on_result(self, fut, seq, what):
        if seq != self._seq:
            return
        code = fut.result().result.error_code.val
        if code == MoveItErrorCodes.SUCCESS and self._holding and what == 'pozycja transportowa':
            self._store()
        if code == MoveItErrorCodes.SUCCESS:
            self._st = Status.SUCCEEDED
        else:
            self.node.get_logger().warn(f'{what}: move_group zwrócił kod {code}')
            self._st = Status.FAILED
            self._diagnose(seq, what)

    def _diagnose(self, seq, what):
        """Po nieudanym planowaniu: stan celu w scenie move_group (kolizje, limity) do logu."""
        q = self._goal_q.get(seq)
        if q is None or self._joint_state is None or not self._validity.service_is_ready():
            return
        js = self._joint_state
        pos = dict(zip(js.name, js.position))
        pos.update({j: float(v) for j, v in zip(ARM_JOINTS, q)})
        rs = RobotState()
        rs.joint_state.name = list(pos)
        rs.joint_state.position = [float(v) for v in pos.values()]
        joints = ', '.join(f'{v:+.3f}' for v in q)
        lim = self._ik.lim
        at_limit = [f'{j} ({v:+.3f} ∈ [{lo:+.2f}, {hi:+.2f}])'
                    for j, v, (lo, hi) in zip(ARM_JOINTS, q, lim)
                    if min(v - lo, hi - v) < 2 * JOINT_TOL]

        def done(fut):
            r = fut.result()
            pairs = sorted({' / '.join(sorted((c.contact_body_1, c.contact_body_2)))
                            for c in r.contacts})
            limits = [c.constraint.joint_constraints[0].joint_name if
                      c.constraint.joint_constraints else '?'
                      for c in r.constraint_result if not c.result]
            self.node.get_logger().warn(
                f'{what}: cel {"poprawny" if r.valid else "NIEPOPRAWNY"} w scenie move_group '
                f'(stawy [{joints}]; kolizje: {", ".join(pairs) or "brak"}'
                f'{"; ograniczenia: " + ", ".join(limits) if limits else ""}'
                f'{"; przy limicie: " + ", ".join(at_limit) if at_limit else ""})')
        self._validity.call_async(GetStateValidity.Request(
            robot_state=rs, group_name=self.group)).add_done_callback(done)

    # ------------------------------------------------------------ chwyt
    def _now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _send_path(self, path, seq):
        """Punkty stawów wprost do manip_controller (FollowJointTrajectory)."""
        if not self._traj.server_is_ready():
            self.node.get_logger().warn('manip_controller niedostępny')
            self._grasp_done(seq, False)
            return
        traj = JointTrajectory(joint_names=ARM_JOINTS)
        for i, q in enumerate(path, 1):
            traj.points.append(JointTrajectoryPoint(
                positions=[float(v) for v in q],
                time_from_start=Duration(seconds=i * self.grasp_step_time).to_msg()))
        fut = self._traj.send_goal_async(FollowJointTrajectory.Goal(trajectory=traj))
        fut.add_done_callback(lambda f: self._on_path_accepted(f, seq))

    def _on_path_accepted(self, fut, seq):
        h = fut.result()
        if seq != self._seq or self._grasp is None:
            return
        if not h.accepted:
            self.node.get_logger().warn('chwyt: trajektoria odrzucona przez manip_controller')
            self._grasp_done(seq, False)
            return
        h.get_result_async().add_done_callback(lambda f: self._on_path_result(f, seq))

    def _on_path_result(self, fut, seq):
        g = self._grasp
        if seq != self._seq or g is None:
            return
        code = fut.result().result.error_code
        if code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.node.get_logger().warn(f'chwyt ({g["phase"]}): manip_controller kod {code}')
        if g['phase'] == 'down':                    # także po przerwaniu - zamknij, co jest
            self.node.get_logger().info(
                f'chwyt: na dole, rozwarcie {np.round(self._jaws, 4).tolist()} - zamykanie')
            self._gripper(GRIPPER_CLOSED)
            g['phase'], g['t0'] = 'close', self._now()
        elif g['phase'] == 'up':
            held = self._jaws_hold()
            self.node.get_logger().info(
                f'chwyt: rozwarcie szczęk {np.round(self._jaws, 4).tolist()} - '
                f'{"próbka w chwytaku" if held else "brak próbki"}')
            self._grasp_done(seq, held)

    def _grasp_tick(self):
        g = self._grasp
        if g is None:
            return
        if g['phase'] == 'close' and self._now() - g['t0'] >= CLOSE_TIME:
            if self._jaws_hold():                   # szczęki zatrzymały się na próbce
                self._gripper(np.maximum(self._jaws - HOLD_SQUEEZE, GRIPPER_CLOSED), sec=0.5)
                self._attach_ack = None
                tip = self._grip_point(GRIP_POINT_DEPTH)
                if tip is not None:
                    self._publish_point(self._grasped_pub, tip)
                    self._sim_held = True
                    g['phase'], g['t0'] = 'attach', self._now()
                    self.node.get_logger().info(
                        f'chwyt: szczęki na próbce, rozwarcie {np.round(self._jaws, 4).tolist()}'
                        ' - przytrzymane, czekam na spaw')
                    return
            self._lift(g)
        elif g['phase'] == 'attach' and (self._attach_ack is not None
                                         or self._now() - g['t0'] >= ATTACH_TIMEOUT):
            self.node.get_logger().info(
                f'chwyt: spaw {"potwierdzony" if self._attach_ack else "niepotwierdzony"}')
            self._lift(g)

    def _lift(self, g):
        g['phase'] = 'up'
        self.node.get_logger().info(
            f'chwyt: zamknięty, rozwarcie {np.round(self._jaws, 4).tolist()} - podniesienie')
        self._send_path(g['up'], g['seq'])

    def _on_attached(self, msg):
        self._attach_ack = bool(msg.data)

    def _jaws_hold(self) -> bool:
        return bool(HOLD_GAP[0] < float(np.sum(self._jaws)) < HOLD_GAP[1])

    def _grasp_done(self, seq, held):
        if seq != self._seq:
            return
        self._grasp = None
        self._holding = held
        if held:
            self._attach_probe(True)
        else:
            self._release()
        self._st = Status.SUCCEEDED if held else Status.FAILED

    def _release(self):
        """Chwytak otwierany z próbką: symulacja ją odłącza (próbka spada)."""
        if self._holding:
            self._holding = False
            self._attach_probe(False)
        if self._sim_held:
            self._sim_held = False
            tip = self._grip_point()
            if tip is not None:
                self._publish_point(self._released_pub, tip)

    def _attach_probe(self, attach):
        """Próbka w chwytaku jako obiekt dołączony do tcp_connector (MoveIt planuje z nią)."""
        aco = AttachedCollisionObject(link_name='tcp_connector', touch_links=TOUCH_LINKS)
        aco.object.id = PROBE_ID
        aco.object.header.frame_id = 'tcp_connector'
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        if attach:
            aco.object.operation = CollisionObject.ADD
            pose = Pose()                           # oś próbki ⟂ szczękom (y) i podejściu (z)
            pose.position.z = FINGERTIP_OFFSET - self.grasp_depth
            pose.orientation.y = pose.orientation.w = 0.5 ** 0.5
            aco.object.primitives = [SolidPrimitive(type=SolidPrimitive.CYLINDER,
                                                    dimensions=[PROBE_LENGTH, PROBE_RADIUS])]
            aco.object.primitive_poses = [pose]
        else:                                       # odłączenie przenosi obiekt do świata
            aco.object.operation = CollisionObject.REMOVE
            gone = CollisionObject(id=PROBE_ID, operation=CollisionObject.REMOVE)
            gone.header.frame_id = self.base_frame
            scene.world.collision_objects = [gone]
        scene.robot_state.attached_collision_objects = [aco]
        self._scene_pub.publish(scene)

    def _grip_point(self, depth=0.0):
        """Punkt na osi chwytaka `depth` nad końcówkami palców w ramce map albo None."""
        T = self.base_pose_fn()
        if T is None or self._ik is None:
            return None
        tcp = self._ik.tcp.fk(dict(zip(ARM_JOINTS, self._q)))
        return (T @ np.r_[tcp[:3, 3] + tcp[:3, 2] * (FINGERTIP_OFFSET - depth), 1.0])[:3]

    def _publish_point(self, pub, p):
        msg = PointStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.point.x, msg.point.y, msg.point.z = (float(v) for v in p)
        pub.publish(msg)

    def _store(self):
        """Próbka w pozycji transportowej: zebrana - komunikat dla symulacji, scena bez niej."""
        self._holding = False
        self._sim_held = False                      # symulacja usuwa przyspawaną próbkę
        self._attach_probe(False)
        tip = self._grip_point()
        if tip is None:
            return
        self._publish_point(self._stored_pub, tip)
        self.node.get_logger().info(
            f'próbka zebrana (chwytak w ({tip[0]:.2f}, {tip[1]:.2f}, {tip[2]:.2f}))')
