"""
Analiza serii epizodów (M9): tabela miar, krzywe Kaplana-Meiera, entropia beliefu, pary ziaren.

  ros2 run ig_search_experiments analyze_series <katalog serii> [--t-max 1800]

Czyta <katalog>/<wariant>/seed_<n>/summary.json (+ status.csv) i results.jsonl (epizody
przerwane bez summary.json - cenzurowane). Zapisuje do <katalog>/analysis/: summary.md (tabela
miar z punktacją wg f(t) regulaminu i czasy t₃ w parach ziaren), km_t1.png, km_t2.png,
km_t3.png, entropy.png, score.png.
"""
import argparse
import csv
import glob
import json
import os

from ig_search.domain.params import MissionParams
from ig_search_experiments.metrics import aborted, episode_from_summary, kaplan_meier, M, \
    summarize, time_to_k
import numpy as np


def load(series: str):
    episodes = {}
    for path in sorted(glob.glob(os.path.join(series, '*', 'seed_*', 'summary.json'))):
        variant = path.split(os.sep)[-3]
        seed = int(path.split(os.sep)[-2].split('_')[1])
        with open(path) as f:
            episodes[(variant, seed)] = episode_from_summary(json.load(f), variant, seed)
    results = os.path.join(series, 'results.jsonl')
    if os.path.exists(results):
        for line in open(results):
            r = json.loads(line)
            key = (r['variant'], r['seed'])
            if key not in episodes:
                episodes[key] = aborted(*key)
    return episodes


def status_series(series: str, variant: str):
    """Lista (t, entropia) dla epizodów wariantu."""
    out = []
    for path in glob.glob(os.path.join(series, variant, 'seed_*', 'status.csv')):
        rows = list(csv.DictReader(open(path)))
        if rows:
            out.append((np.array([float(r['t']) for r in rows]),
                        np.array([float(r['belief_entropy']) for r in rows])))
    return out


def fmt(v, unit=''):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '-'
    return f'{v:.0f}{unit}' if unit == ' s' else f'{v:.2f}{unit}'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('series')
    ap.add_argument('--t-max', type=float, default=1800.0)
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    eps = load(a.series)
    variants = sorted({v for v, _ in eps})
    seeds = sorted({s for _, s in eps})
    out = os.path.join(a.series, 'analysis')
    os.makedirs(out, exist_ok=True)

    lines = ['| wariant | epizody | 3/3 | η | π | fałszywe | t₁ med. | t₂ med. | t₃ med. '
             '| droga śr. | błąd śr. | punktacja śr. | przerwane |', '|' + '---|' * 13]
    for v in variants:
        s = summarize([e for (vv, _), e in eps.items() if vv == v], a.t_max)
        lines.append(f'| {v} | {s["episodes"]} | {s["complete"]} | {fmt(s["eta"])} | '
                     f'{fmt(s["precision"])} | {s["false_reports"]} | '
                     + ' | '.join(fmt(s[f't{k}_median'], ' s') for k in range(1, M + 1))
                     + f' | {s["distance_mean"]:.0f} m | {fmt(100 * s["error_mean"], " cm")} | '
                     f'{fmt(s["score_mean"])} | {s["aborted"]} |')
    lines += ['', 'Czas t₃ [s] w parach ziaren (- = mniej niż 3 poprawne zgłoszenia, '
              'w nawiasie liczba poprawnych):', '',
              '| ziarno | ' + ' | '.join(variants) + ' |', '|' + '---|' * (len(variants) + 1)]
    for sd in seeds:
        cells = []
        for v in variants:
            e = eps.get((v, sd))
            if e is None:
                cells.append(' ')
            elif len(e.times) >= M:
                cells.append(f'{e.times[M - 1]:.0f}')
            else:
                cells.append(f'- ({len(e.times)}, {e.state})')
        lines.append(f'| {sd} | ' + ' | '.join(cells) + ' |')
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(out, 'summary.md'), 'w') as f:
        f.write(text)
    print(text)

    for k in range(1, M + 1):
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
        for v in variants:
            d, e = time_to_k([ep for (vv, _), ep in eps.items() if vv == v], k, a.t_max)
            t, s = kaplan_meier(d, e)
            ax.step(np.r_[0.0, t, a.t_max], 1.0 - np.r_[1.0, s, s[-1] if len(s) else 1.0],
                    where='post', label=f'{v} (n={len(d)})')
        ax.set_xlim(0, a.t_max)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('czas misji [s]')
        ax.set_ylabel(f'P(odnaleziono ≥ {k} próbk{"ę" if k == 1 else "i"})')
        ax.set_title(f'Kaplan-Meier: czas t{k} ({k}. poprawne zgłoszenie)')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'km_t{k}.png'))
        plt.close(fig)

    # punktacja: chwile końca misji na tle stopni f(t) regulaminu
    mp = MissionParams()
    edges = [0.0] + list(mp.score_steps) + [a.t_max]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(variants) + 1.8), dpi=110)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        f = mp.score_factors[i]
        ax.axvspan(lo / 60, hi / 60, color='#2a78d6', alpha=0.04 + 0.16 * (1 - f) / 0.4, lw=0)
        ax.text((lo + hi) / 120, len(variants) - 0.35, f'f = {f:.0%}', ha='center',
                va='bottom', fontsize=9, color='#555')
    for r, v in enumerate(variants):
        y = len(variants) - 1 - r
        vep = [e for (vv, _), e in eps.items() if vv == v and e.state != 'ABORTED']
        dy = {sd: 0.3 * (0.5 - k / max(len(seeds) - 1, 1)) for k, sd in enumerate(seeds)}
        full = [e for e in vep if len(e.times) >= M]
        part = [e for e in vep if len(e.times) < M]
        ax.scatter([e.elapsed / 60 for e in full], [y + dy[e.seed] for e in full], s=60,
                   color='#2a78d6', edgecolor='white', lw=1.5, zorder=3)
        for e in part:
            ax.scatter([e.elapsed / 60], [y + dy[e.seed]], s=60, facecolor='white',
                       edgecolor='#2a78d6', lw=1.5, zorder=3)
            ax.annotate(f'{len(e.times)}/{M}', (e.elapsed / 60, y + dy[e.seed]), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=8, color='#555')
        sc = summarize(vep, a.t_max)['score_mean']
        ax.text(a.t_max / 60 + 0.4, y, f'{sc:.2f}', va='center', fontsize=10)
    ax.text(a.t_max / 60 + 0.4, len(variants) - 0.35, 'śr.', va='bottom', fontsize=9,
            color='#555')
    ax.set_yticks(range(len(variants)), list(reversed(variants)))
    ax.set_ylim(-0.6, len(variants) - 0.1)
    ax.set_xlim(0, a.t_max / 60)
    ax.set_xticks([0.0] + [t / 60 for t in mp.score_steps] + [a.t_max / 60])
    ax.xaxis.set_major_formatter(lambda x, _: f'{int(x)}:{int(round(x % 1 * 60)):02d}')
    ax.set_xlabel('koniec misji [mm:ss]\n● 3/3 próbki, ○ mniej (liczba nad punktem); '
                  'ziarna w wierszu kolejno z góry na dół', fontsize=9)
    ax.set_title('Punktacja wg regulaminu ERC: (n_pop / 3) · f(t_końca)', loc='left')
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'score.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
    grid = np.linspace(0, a.t_max, 361)
    for i, v in enumerate(variants):
        runs = status_series(a.series, v)
        if not runs:
            continue
        ys = np.array([np.interp(grid, t, h, right=np.nan) for t, h in runs])
        with np.errstate(all='ignore'):
            ax.plot(grid, np.nanmedian(ys, axis=0), label=f'{v} (mediana, n={len(runs)})',
                    color=f'C{i}')
    ax.set_xlabel('czas misji [s]')
    ax.set_ylabel('entropia beliefu [bit]')
    ax.set_title('Entropia beliefu w czasie')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'entropy.png'))
    plt.close(fig)
    print(f'wykresy: {out}')


if __name__ == '__main__':
    main()
