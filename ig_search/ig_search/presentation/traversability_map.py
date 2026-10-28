"""
CLI: mapa przejezdności (PNG + .yaml, format nav2_map_server / kalman_nav2) z mapy wysokości.

  ros2 run ig_search traversability_map <mapa_wysokości.yaml> <wyjście.yaml> [--start 0 0]
      [--exclude <strefy.yaml>] [--collision <mapa_kolizyjna.yaml> --spawn-out <próbki.yaml>]

Z --collision powstaje też mapa miejsc na próbki: mapa przejezdności bez miejsc, w których
próbka zapada się pod widoczny teren (mesh kolizyjny niżej niż wizualny).

Kryteria terenu jak w TravelModel (parametry travel z domain/params.py), bez promienia łazika,
oraz wycięte wzniesienia ponad otoczenie (kopce).
"""
import argparse
import time

from ig_search.domain.params import TravelParams
from ig_search.domain.terrain import HeightMap
from ig_search.domain.traversability import load_exclusions, TraversabilityMap


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('heightmap', help='mapa wysokości .yaml (mesh_to_heightmap)')
    ap.add_argument('out', help='plik wyjściowy .yaml (obok powstanie .png)')
    ap.add_argument('--start', type=float, nargs=2, default=(0.0, 0.0), metavar=('X', 'Y'),
                    help='punkt startu łazika: zostaje spójna strefa z tym punktem')
    ap.add_argument('--radius', type=float, default=0.3,
                    help='promień wygładzenia (domknięcie i otwarcie) [m]')
    ap.add_argument('--min-hole', type=float, default=2.0,
                    help='dziury mniejsze niż tyle [m²] są zalewane')
    ap.add_argument('--upsample', type=int, default=2, help='zagęszczenie siatki obrazu')
    ap.add_argument('--hill-height', type=float, default=0.25,
                    help='wzniesienia wyżej niż tyle ponad medianę otoczenia [m] są wycinane '
                         '(0 - bez wycinania)')
    ap.add_argument('--hill-window', type=float, default=6.0,
                    help='bok kwadratu otoczenia dla wzniesień [m]')
    ap.add_argument('--exclude', action='append', default=[],
                    help='strefy wyłączone: .yaml {zones: [{name, polygon}]} (można kilka razy)')
    ap.add_argument('--collision', default='',
                    help='mapa wysokości z mesha kolizyjnego (mesh_to_heightmap)')
    ap.add_argument('--spawn-out', default='', help='wyjście: mapa miejsc na próbki (.yaml)')
    ap.add_argument('--burial-depth', type=float, default=0.015,
                    help='widoczny teren wyżej niż kolizyjny o więcej [m] - próbka znika')
    ap.add_argument('--burial-window', type=float, default=0.5,
                    help='bok kwadratu, w którym liczona jest ta różnica [m]')
    args = ap.parse_args(argv)
    t0 = time.monotonic()
    hm = HeightMap.load(args.heightmap)
    tm = TraversabilityMap.from_heightmap(hm, TravelParams(), tuple(args.start), args.radius,
                                          args.min_hole, args.upsample, args.hill_height,
                                          args.hill_window,
                                          [z for f in args.exclude for z in load_exclusions(f)])
    tm.save(args.out)
    g = tm.grid
    print(f'{g.width}x{g.height} px po {g.res} m, origin ({g.origin_x:.2f}, {g.origin_y:.2f}), '
          f'przejezdne {tm.area():.0f} m² ({tm.mask.mean():.0%}), '
          f'{time.monotonic() - t0:.1f} s')
    if args.collision and args.spawn_out:
        sm = tm.spawn_map(hm, HeightMap.load(args.collision), args.burial_depth,
                          args.burial_window)
        sm.save(args.spawn_out)
        print(f'miejsca na próbki: {sm.area():.0f} m² ({sm.area() / tm.area():.0%} mapy)')


if __name__ == '__main__':
    main()
