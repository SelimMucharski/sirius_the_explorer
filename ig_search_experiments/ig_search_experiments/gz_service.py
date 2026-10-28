"""
Wywołania usług Gazebo w osobnym procesie (`gz service`).

Zapytanie gz.transport z Pythona w procesie, który subskrybuje też /world/<w>/pose/info
(GzWorldPoses), po pierwszym wywołaniu kończyło się limitem czasu (callback subskrypcji
i oczekiwanie na odpowiedź blokowały się nawzajem); CLI w osobnym procesie działa.
"""
import math
import subprocess


def _call(service, reqtype, req, timeout_ms=5000) -> bool:
    try:
        res = subprocess.run(['gz', 'service', '-s', service, '--reqtype', reqtype,
                              '--reptype', 'gz.msgs.Boolean', '--timeout', str(timeout_ms),
                              '--req', req],
                             capture_output=True, text=True, timeout=timeout_ms / 1000 + 5)
    except subprocess.TimeoutExpired:
        return False
    return res.returncode == 0 and 'data: true' in res.stdout


def set_pose(world, name, x, y, z, yaw=0.0) -> bool:
    req = (f'name: "{name}" position {{x: {x} y: {y} z: {z}}} '
           f'orientation {{z: {math.sin(yaw / 2)} w: {math.cos(yaw / 2)}}}')
    return _call(f'/world/{world}/set_pose', 'gz.msgs.Pose', req)


def remove_model(world, name) -> bool:
    return _call(f'/world/{world}/remove', 'gz.msgs.Entity', f'name: "{name}" type: MODEL')


def create_model(world, sdf_file, name, position, quaternion) -> bool:
    """Model z pliku SDF w pozie (x, y, z) + kwaternion (x, y, z, w); bez zmiany nazwy."""
    (x, y, z), (qx, qy, qz, qw) = position, quaternion
    req = (f'sdf_filename: "{sdf_file}" name: "{name}" allow_renaming: false '
           f'pose {{position {{x: {x} y: {y} z: {z}}} '
           f'orientation {{x: {qx} y: {qy} z: {qz} w: {qw}}}}}')
    return _call(f'/world/{world}/create', 'gz.msgs.EntityFactory', req)
