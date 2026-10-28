"""
CLI: mapa wysokości (.npy + .yaml) z mesha terenu Gazebo.

  ros2 run ig_search mesh_to_heightmap <teren.obj> <wyjście.yaml> [--res 0.1]
"""
import argparse
import time

from ig_search.domain.mesh_heightmap import rasterize, read_obj
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('obj', help='mesh terenu (OBJ, oś z w górę, ramka świata Gazebo = map)')
    ap.add_argument('out', help='plik wyjściowy .yaml (obok powstanie .npy)')
    ap.add_argument('--res', type=float, default=0.1, help='rozdzielczość [m]')
    args = ap.parse_args(argv)
    t0 = time.monotonic()
    verts, tris = read_obj(args.obj)
    hm = rasterize(verts, tris, args.res)
    hm.save(args.out)
    g = hm.grid
    print(f'{len(tris)} trójkątów -> {g.width}x{g.height} komórek po {g.res} m, '
          f'origin ({g.origin_x:.2f}, {g.origin_y:.2f}), pokrycie {hm.valid.mean():.1%}, '
          f'h ∈ [{np.nanmin(hm.h):.2f}, {np.nanmax(hm.h):.2f}] m, {time.monotonic() - t0:.1f} s')


if __name__ == '__main__':
    main()
