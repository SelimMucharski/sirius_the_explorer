#!/usr/bin/env python3
"""
Rendery świata Gazebo z zadanych póz kamery (bez GUI): porównanie wariantów świata, podgląd.

  python3 render_views.py --world worlds/erc2025_real.sdf --out <katalog> [--probes]

Do kopii świata dokłada statyczne kamery (po jednej na widok, temat /render/<nazwa>)
i opcjonalnie próbki (wbite pionowo - statyczne, leżące - dynamiczne), uruchamia
`gz sim -s -r --headless-rendering`, zapisuje po jednej klatce z każdej kamery (PNG) i kończy.
"""
import argparse
import math
import os
import signal
import subprocess
import tempfile
import threading
import time

from gz.msgs10.image_pb2 import Image as GzImage
from gz.transport13 import Node
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, '..', '..'))
PROBE_SDF = os.path.normpath(os.path.join(PKG, '..', 'probe_supervisor', 'objects', 'probe',
                                          'probe.sdf'))

# nazwa: (x, y, z, roll, pitch, yaw, hfov [rad])
VIEWS = {
    'overview': (-2.5, -13.0, 5.5, 0.0, 0.42, 0.62, 1.3),
    'garden_a': (1.6, -1.5, 1.3, 0.0, 0.32, -0.45, 1.2),
    'garden_b': (20.5, -0.5, 1.2, 0.0, 0.3, -2.6, 1.2),
    'rover_eye': (3.0, 1.8, 0.45, 0.0, 0.22, -0.2, 1.69),
    'probe_upright': (4.4, 1.25, 0.5, 0.0, 0.42, -0.25, 1.0),
    'probe_lying': (11.6, -8.4, 0.55, 0.0, 0.5, 0.6, 1.0),
    'mound': (6.8, 3.2, 1.1, 0.0, 0.35, -0.5, 1.2),
}
# próbki do podglądu: (x, y, upright)
PROBES = [(5.6, 0.95, True), (5.9, 1.25, False), (12.7, -7.6, False), (12.4, -7.9, True),
          (8.6, 0.6, True), (17.0, -3.0, True), (5.2, -3.6, False)]


def camera_model(name, pose, hfov, w, h):
    x, y, z, r, p, yaw, _ = pose
    return f"""
    <model name="render_{name}">
      <static>true</static>
      <pose>{x} {y} {z} {r} {p} {yaw}</pose>
      <link name="link">
        <sensor name="cam" type="camera">
          <always_on>1</always_on>
          <update_rate>2</update_rate>
          <topic>/render/{name}</topic>
          <camera>
            <horizontal_fov>{hfov}</horizontal_fov>
            <image><width>{w}</width><height>{h}</height><format>R8G8B8</format></image>
            <clip><near>0.05</near><far>120</far></clip>
          </camera>
        </sensor>
      </link>
    </model>"""


def probe_models(heightmap):
    import sys
    sys.path.insert(0, os.path.normpath(os.path.join(PKG, '..', 'probe_supervisor')))
    from probe_supervisor.domain.upright import Footprints, HeightGrid, PROBE_LENGTH
    g = HeightGrid.load(heightmap)
    fp = Footprints.load(os.path.join(PKG, 'models', 'erc2025_dressing', 'footprints.yaml'))
    body = open(PROBE_SDF).read()
    body = body[body.index('<model'):body.rindex('</model>') + len('</model>')]
    out = []
    for i, pr in enumerate(PROBES):
        x, y, up = pr[:3]
        yaw = pr[3] if len(pr) > 3 else 0.4 + i
        for k in range(60):                     # obok kamienia, nie w nim (spirala co 5 cm)
            dx, dy = 0.05 * k ** 0.5 * math.cos(2.4 * k), 0.05 * k ** 0.5 * math.sin(2.4 * k)
            if not fp.blocked(x + dx, y + dy, 0.05):
                x, y = x + dx, y + dy
                break
        zg = g.at(x, y)
        if up:
            pose = f'{x} {y} {zg - 0.35 * PROBE_LENGTH + 0.5 * PROBE_LENGTH} 0.08 -0.05 {yaw}'
        else:                                       # leży na gruncie (statycznie - bez upadku)
            pose = f'{x} {y} {zg + 0.016} 1.5708 0 {yaw}'
        m = body.replace('<model name="probe">',
                         f'<model name="probe_{i}">\n    <static>true</static>', 1)
        m = m.replace('meshes/probe.obj', 'file://' + os.path.join(
            os.path.dirname(PROBE_SDF), 'meshes', 'probe.obj'))
        out.append(m.replace('<link name="link">', f'<pose>{pose}</pose>\n    <link name="link">',
                             1))
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--world', default=os.path.join(PKG, 'worlds', 'erc2025_real.sdf'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--prefix', default='')
    ap.add_argument('--views', nargs='*', default=list(VIEWS))
    ap.add_argument('--probes', action='store_true')
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--settle', type=float, default=8.0, help='[s] na ułożenie się próbek')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    w = open(a.world).read()
    extra = ''.join(camera_model(n, VIEWS[n], VIEWS[n][6], a.width, a.height) for n in a.views)
    if a.probes:
        extra += probe_models(os.path.normpath(os.path.join(
            PKG, '..', 'ig_search', 'config', 'maps', 'erc2025_heightmap.yaml')))
    w = w.replace('</world>', extra + '\n  </world>')
    tmp = tempfile.NamedTemporaryFile('w', suffix='.sdf', delete=False)
    tmp.write(w)
    tmp.close()
    env = dict(os.environ)
    env['GZ_SIM_RESOURCE_PATH'] = os.path.join(PKG, 'models') + ':' + env.get(
        'GZ_SIM_RESOURCE_PATH', '')
    proc = subprocess.Popen(['gz', 'sim', '-s', '-r', '--headless-rendering', tmp.name],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    got, lock, t0 = {}, threading.Lock(), time.monotonic()
    node = Node()

    def cb_for(name):
        def cb(msg):
            if time.monotonic() - t0 < a.settle or name in got:
                return
            img = Image.frombytes('RGB', (msg.width, msg.height), msg.data)
            with lock:
                got[name] = img
        return cb
    for n in a.views:
        node.subscribe(GzImage, f'/render/{n}', cb_for(n))
    try:
        while len(got) < len(a.views) and time.monotonic() - t0 < 240:
            time.sleep(0.5)
    finally:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        os.unlink(tmp.name)
    for n, img in got.items():
        path = os.path.join(a.out, f'{a.prefix}{n}.png')
        img.save(path)
        print(path)
    missing = set(a.views) - set(got)
    if missing:
        print('brak klatek:', sorted(missing))


if __name__ == '__main__':
    main()
