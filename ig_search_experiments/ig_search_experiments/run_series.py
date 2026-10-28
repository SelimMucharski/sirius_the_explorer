r"""
Seria epizodów Gazebo bez GUI (M7): ziarna × warianty, przeplatane - niepełna seria daje pary.

  ros2 run ig_search_experiments run_series --out <katalog> --seeds 1 2 3 \
      --variant ig_greedy=strategy:=ig_greedy \
      --variant ig_horizon_d5="strategy:=ig_horizon horizon_depth:=5"

Każdy epizod: `sim_episode.launch.py seed:=<s> headless:=true watch_output:=<katalog>
variant:=<etykieta> <argumenty wariantu>` w <out>/<etykieta>/seed_<s>. Epizod z istniejącym
summary.json jest pomijany (wznowienie serii). Przerwanie: koniec misji (summary.json), padnięcie
kluczowego procesu, brak zdarzeń misji przez --stall s, Nav2 nieaktywny po --nav2-timeout s
(menedżer cyklu życia potrafi zawisnąć na konfiguracji - epizod jest wtedy powtarzany, najwyżej
--retries razy), limit czasu ściennego. Wynik każdego epizodu (także przerwanego) - wiersz
w <out>/results.jsonl.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import time

KEY_PROCESSES = ('search_node', 'gz sim', 'move_group', 'controller_server', 'bt_navigator')
EVENT = re.compile(r'ig_search\]: \[ *[0-9.]+ s\]')
NAV2_ACTIVE = re.compile(r'lifecycle_manager_navigation\].*Managed nodes are active')
PATTERNS = ('sirius_ws/install/', '/opt/ros/jazzy/lib/', 'gz sim', 'ros2 launch', 'gz_tools')


def _ps():
    out = subprocess.run(['ps', '-eo', 'pid=,pgid=,args='], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3:
            yield int(parts[0]), int(parts[1]), parts[2]


def cleanup(own_pgid: int):
    """Zabija grupy procesów symulacji (resztki epizodu, np. zamykający się move_group)."""
    groups = {pgid for pid, pgid, args in _ps()
              if pgid != own_pgid and any(p in args for p in PATTERNS)
              and 'ros2-daemon' not in args and 'run_series' not in args}
    for g in groups:
        try:
            os.killpg(g, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(3)


def run_episode(seed, label, args, out, a):
    os.makedirs(out, exist_ok=True)
    log = os.path.join(out, 'launch.log')
    cmd = (['ros2', 'launch', 'ig_search', 'sim_episode.launch.py', f'seed:={seed}',
            'headless:=true', f'watch_output:={out}', f'variant:={label}'] + args.split())
    t0 = time.monotonic()
    with open(log, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    n_ev, last_ev, reason = 0, t0, ''
    while True:
        time.sleep(15)
        now = time.monotonic()
        text = open(log, errors='replace').read()
        if os.path.exists(os.path.join(out, 'summary.json')):
            reason = 'koniec misji'
        elif proc.poll() is not None:
            reason = 'launch zakończony bez summary'
        elif any(re.search(rf'process has died.*{p}', text) for p in KEY_PROCESSES):
            reason = 'padł kluczowy proces'
        elif now - t0 > a.nav2_timeout and not NAV2_ACTIVE.search(text):
            reason = 'nav2'
        elif now - t0 > a.wall_limit:
            reason = 'limit czasu ściennego'
        else:
            n = len(EVENT.findall(text))
            if n > n_ev:
                n_ev, last_ev = n, now
            if now - last_ev > a.stall:
                reason = f'brak zdarzeń misji przez {a.stall:.0f} s'
        if reason:
            break
    time.sleep(5)
    for sig, wait in ((signal.SIGINT, 15), (signal.SIGKILL, 0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        time.sleep(wait)
    cleanup(os.getpgid(0))
    return reason, round(time.monotonic() - t0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--variant', action='append', required=True,
                    help='etykieta=argumenty launch (np. ig_horizon="strategy:=ig_horizon")')
    ap.add_argument('--stall', type=float, default=900.0, help='[s ściennych] bez zdarzeń')
    ap.add_argument('--nav2-timeout', type=float, default=300.0)
    ap.add_argument('--wall-limit', type=float, default=7500.0)
    ap.add_argument('--retries', type=int, default=3, help='powtórzenia, gdy Nav2 nie wstał')
    a = ap.parse_args(argv)
    variants = [v.split('=', 1) for v in a.variant]
    os.makedirs(a.out, exist_ok=True)
    results = os.path.join(a.out, 'results.jsonl')
    cleanup(os.getpgid(0))
    for seed in a.seeds:
        for label, args in variants:
            out = os.path.join(a.out, label, f'seed_{seed}')
            if os.path.exists(os.path.join(out, 'summary.json')):
                print(f'{label} seed {seed}: już jest', flush=True)
                continue
            for attempt in range(1, a.retries + 1):
                print(f'{time.strftime("%T")} start {label} seed {seed} (próba {attempt})',
                      flush=True)
                reason, wall = run_episode(seed, label, args, out, a)
                if reason != 'nav2':
                    break
                os.rename(out, f'{out}_nav2fail_{time.strftime("%H%M%S")}')
            row = {'variant': label, 'seed': seed, 'args': args, 'reason': reason,
                   'wall': wall, 'attempts': attempt}
            sp = os.path.join(out, 'summary.json')
            if os.path.exists(sp):
                s = json.load(open(sp))
                row.update(state=s['state'], elapsed=s['elapsed'], correct=s['correct_0.15'],
                           reports=len(s['reports']), distance=s['distance'])
            with open(results, 'a') as f:
                f.write(json.dumps(row) + '\n')
            print(f'{time.strftime("%T")} koniec {label} seed {seed}: {json.dumps(row)}',
                  flush=True)
    print(f'{time.strftime("%T")} seria zakończona', flush=True)


if __name__ == '__main__':
    main()
