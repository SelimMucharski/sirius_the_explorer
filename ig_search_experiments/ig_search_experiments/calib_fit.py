"""
M6: dopasowanie tablic modelu detektora z danych calib_collect (DESIGN §3.2).

Wynik w formacie DetectorModel.load_table:
  r_edges, alpha_edges, pd[r][α]  - P_d w przedziałach (częstość wykryć widocznych próbek),
  sigma_r_edges, sigma            - σ(r) błędu pozycji 2D (na oś: sqrt(mean(e_x² + e_y²) / 2)),
  lambda_fa                       - fałszywe alarmy w V(q) na m² widocznego terenu na klatkę.
Przedziały z małą liczbą próbek dostają wartość z przedziału zbiorczego (wszystkie α) albo
sąsiedniego r. Raport (liczności, przedziały ufności Wilsona, obciążenie błędu wzdłuż promienia)
- <out>.txt obok YAML.

  ros2 run ig_search_experiments calib_fit <katalog danych> <faza> <out.yaml> [--cell 0.2]
"""
import argparse
import csv
import math
import os

import numpy as np
import yaml

EDGES = {
    # kamera przednia: jeden przedział α - przy ~10 niezależnych pozach na przedział r zależność
    # od α nie jest wiarygodna (pominięcia przychodzą całymi pozami: próbka widziana końcem,
    # częściowo zasłonięta), a dzielenie na α przypisywało je przypadkowym przedziałom
    'front': {'r': [0.3, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
              'alpha': [0.0, 0.85]},
    'wrist': {'r': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2],
              'alpha': [0.0, 0.3, 0.62]},
}
MIN_N = 20                  # próbek na przedział P_d
MIN_N_SIGMA = 10


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fit(data_dir, phase, cell, still_only=True):
    probes = read(os.path.join(data_dir, f'{phase}_probes.csv'))
    frames = read(os.path.join(data_dir, f'{phase}_frames.csv'))
    if still_only:
        probes = [p for p in probes if p['moving'] == '0']
        frames = [f for f in frames if f['moving'] == '0']
    r = np.array([float(p['r']) for p in probes])
    a = np.array([float(p['alpha']) for p in probes])
    det = np.array([int(p['detected']) for p in probes])
    ex = np.array([float(p['err_x']) if p['err_x'] else np.nan for p in probes])
    ey = np.array([float(p['err_y']) if p['err_y'] else np.nan for p in probes])
    er = np.array([float(p['err_range']) if p['err_range'] else np.nan for p in probes])
    ec = np.array([float(p['err_cross']) if p['err_cross'] else np.nan for p in probes])
    re, ae = EDGES[phase]['r'], EDGES[phase]['alpha']
    ri = np.clip(np.searchsorted(re, r, side='right') - 1, 0, len(re) - 2)
    ai = np.clip(np.searchsorted(ae, a, side='right') - 1, 0, len(ae) - 2)
    inside = (r >= re[0]) & (r <= re[-1])
    nr, na = len(re) - 1, len(ae) - 1
    pd = np.zeros((nr, na))
    pair = np.array([f"{p['pose']}/{p['probe']}" for p in probes])
    lines = [f'faza {phase}: {len(probes)} obserwacji widocznych próbek, {len(frames)} klatek, '
             f'{len(set(pair))} par poza-próbka',
             '', 'P_d (k/n klatek = P_d, przedział Wilsona 95% dla klatek; pary poza-próbka):']
    for i in range(nr):
        row_sel = inside & (ri == i)
        k_all, n_all = int(det[row_sel].sum()), int(row_sel.sum())
        pooled = (k_all + 0.5) / (n_all + 1) if n_all else None
        cells = []
        for j in range(na):
            sel = row_sel & (ai == j)
            k, n = int(det[sel].sum()), int(sel.sum())
            if n >= MIN_N:
                pd[i, j] = (k + 0.5) / (n + 1)
                src = ''
            elif pooled is not None and n_all >= MIN_N:
                pd[i, j] = pooled
                src = '*'
            else:
                pd[i, j] = pd[i - 1, j] if i > 0 else 0.0
                src = '^'
            lo, hi = wilson(k, n)
            n_pairs = len(set(pair[sel]))
            cells.append(f'{k}/{n}={pd[i, j]:.2f}{src} [{lo:.2f}, {hi:.2f}] ({n_pairs} par)')
        lines.append(f'  r {re[i]:.1f}-{re[i + 1]:.1f} m: ' + ' | '.join(cells))
    lines.append('  (* - wartość zbiorcza dla wszystkich α, ^ - z poprzedniego r)')
    sig = np.full(nr, np.nan)
    lines += ['', 'σ(r) [m] (na oś), obciążenie wzdłuż promienia / w poprzek [m]:']
    ok = det.astype(bool) & np.isfinite(ex)
    for i in range(nr):
        sel = ok & inside & (ri == i)
        n = int(sel.sum())
        if n >= MIN_N_SIGMA:
            sig[i] = math.sqrt(np.mean(ex[sel] ** 2 + ey[sel] ** 2) / 2)
            lines.append(f'  r {re[i]:.1f}-{re[i + 1]:.1f}: σ = {sig[i]:.3f} (n {n}), '
                         f'wzdłuż {np.mean(er[sel]):+.3f} ± {np.std(er[sel]):.3f}, '
                         f'w poprzek {np.mean(ec[sel]):+.3f} ± {np.std(ec[sel]):.3f}')
        else:
            lines.append(f'  r {re[i]:.1f}-{re[i + 1]:.1f}: za mało ({n}) - z sąsiedniego')
    have = np.flatnonzero(np.isfinite(sig))     # puste przedziały - najbliższy z danymi
    for i in np.flatnonzero(~np.isfinite(sig)):
        sig[i] = sig[have[np.argmin(np.abs(have - i))]] if len(have) else 0.05
    # σ niemalejąca z r (przedziały z kilku póz bywają zaszumione, a błąd nie maleje z odległością)
    sig_raw, sig = sig, np.maximum.accumulate(sig)
    if np.any(sig != sig_raw):
        lines.append('  wygładzone (niemalejąca): ' + ', '.join(f'{v:.3f}' for v in sig))
    n_fa = sum(int(f['n_fa_visible']) for f in frames)
    area = sum(int(f['n_visible_cells']) for f in frames) * cell * cell
    # bez fałszywych alarmów - górna granica 95% (reguła trzech): λ = 0 dzieliłoby przez zero
    # w aktualizacji beliefu, a brak FA w symulacji nie znaczy, że ich nie ma
    lam = n_fa / area if n_fa else (3.0 / area if area else 1e-4)
    n_fa_out = sum(int(f['n_fa_other']) for f in frames)
    n_dup = sum(int(f.get('n_fa_near') or 0) for f in frames)
    lines += ['', f'λ_fa = {n_fa} FA w V(q) / {area:.0f} m²·klatka = {lam:.6f} [1/(m²·klatka)]'
              + (' (0 FA: górna granica 95%, 3 / powierzchnia)' if not n_fa else ''),
              f'(FA poza V(q): {n_fa_out}; średnio {n_fa / max(len(frames), 1):.3f} FA w V na '
              f'klatkę; podwójnych detekcji próbki < 0,15 m, niewliczonych: {n_dup})']
    table = {'r_edges': [float(v) for v in re], 'alpha_edges': [float(v) for v in ae],
             'pd': np.round(pd, 3).tolist(), 'sigma_r_edges': [float(v) for v in re],
             'sigma': np.round(sig, 4).tolist(), 'lambda_fa': float(f'{lam:.3g}')}
    return table, '\n'.join(lines)


def main(args=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('data_dir')
    ap.add_argument('phase', choices=['front', 'wrist'])
    ap.add_argument('out')
    ap.add_argument('--cell', type=float, default=0.2)
    ap.add_argument('--include-moving', action='store_true')
    a = ap.parse_args(args)
    table, report = fit(a.data_dir, a.phase, a.cell, still_only=not a.include_moving)
    with open(a.out, 'w') as f:
        f.write(f'# Kalibracja modelu detektora ({a.phase}) z Gazebo - calib_fit (M6).\n')
        yaml.safe_dump(table, f, sort_keys=False, default_flow_style=None)
    with open(os.path.splitext(a.out)[0] + '.txt', 'w') as f:
        f.write(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
