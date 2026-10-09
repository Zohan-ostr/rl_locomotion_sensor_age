"""Build publication assets from preserved measurements; no simulations are run."""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'
SOURCES = {}
TABLE_SECTIONS = []
METHODS = ('B1', 'M', 'M2')
COLORS = {'B1': '#777777', 'M': '#d97706', 'M2': '#2563eb'}


def load(name):
    path = ROOT / name
    SOURCES[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return center-half, center+half


def table(name, rows):
    keys = list(rows[0])
    with (OUT/'tables'/f'{name}.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| '+' | '.join(keys)+' |', '| '+' | '.join(['---']*len(keys))+' |']
    for row in rows:
        lines.append('| '+' | '.join(f'{row[k]:.6f}' if isinstance(row[k], float) else str(row[k]) for k in keys)+' |')
    TABLE_SECTIONS.append(f'<a id="{name}"></a>\n\n## {name}\n\n'+'\n'.join(lines)+'\n')


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/'figures'/f'{name}.{ext}', dpi=170, bbox_inches='tight')
    plt.close(fig)


def main():
    TABLE_SECTIONS.clear()
    (OUT/'figures').mkdir(parents=True, exist_ok=True)
    (OUT/'tables').mkdir(parents=True, exist_ok=True)
    development = load('results/m2_hybrid_h1/comparison.json')['groups']
    validation = load('results/m2_validation_h1/validation_summary.json')['groups']
    cohorts = [('development_0_9', 'Подбор: seed 0–9', development),
               ('validation_10_19', 'Проверка: seed 10–19', validation)]
    rows = []
    for cohort, _, groups in cohorts:
        for condition, methods in groups.items():
            for method, values in methods.items():
                lo, hi = wilson(values['falls'], values['n'])
                rows.append({'cohort': cohort, 'condition': condition, 'method': method,
                             **values, 'fall_rate': values['falls']/values['n'],
                             'wilson95_low': lo, 'wilson95_high': hi})
    table('comparison', rows)
    for metric, ylabel in [('fall_rate', 'Доля падений'), ('velocity_rmse', 'Среднее RMSE скорости, м/с'),
                           ('mean_alive_s', 'Среднее T_alive, с')]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for row, (_, title, groups) in enumerate(cohorts):
            for col, (prefix, stream) in enumerate([('J', 'Суставы'), ('I', 'IMU')]):
                ax = axes[row, col]
                for index, method in enumerate(METHODS):
                    values = [groups[f'{prefix}{delay}'][method] for delay in (20, 40, 80)]
                    x = np.array([20, 40, 80]) + (index-1)*1.2
                    if metric == 'fall_rate':
                        y = np.array([v['falls']/v['n'] for v in values])
                        bounds = np.array([wilson(v['falls'], v['n']) for v in values])
                        error = np.maximum(0, np.array([y-bounds[:, 0], bounds[:, 1]-y]))
                        ax.errorbar(x, y, yerr=error, marker='o', capsize=3, label=method, color=COLORS[method])
                        ax.set_ylim(-.04, 1.04)
                    else:
                        ax.plot(x, [v[metric] for v in values], 'o-', label=method, color=COLORS[method])
                    assert all(v['n'] == 10 for v in values)
                ax.set(title=f'{title} — {stream}', xlabel='Задержка потока, мс', ylabel=ylabel,
                       xticks=[20, 40, 80])
                ax.grid(alpha=.2)
                ax.legend()
        note = 'Wilson 95%; N=10 на точку. Сдвиг ±1,2 мс только для читаемости.' if metric == 'fall_rate' else \
               'N=10 на точку. RMSE усечён до падения; T_alive ограничено 20 с.'
        fig.suptitle(note, fontsize=11)
        save(fig, metric)
    pilot_path = ROOT/'results/pilot_h1/groups.csv'
    SOURCES[str(pilot_path.relative_to(ROOT))] = hashlib.sha256(pilot_path.read_bytes()).hexdigest()
    with pilot_path.open() as file:
        pilot = list(csv.DictReader(file))
    extra = load('results/m2_validation_h1/extra_summary.json')
    extension_rows = []
    wanted = ['clean', 'jitter'] + [f'equal{d}' for d in (20, 40, 80, 120)] + \
             [f'drop_{s}_{n}' for s in ('joints', 'imu') for n in (1, 2, 4, 6)]
    for condition in wanted:
        for method in METHODS:
            if method == 'M2':
                values = [v for v in extra if v['condition']['id'] == condition]
                if not values:
                    continue  # Equal-delay M2 measurements do not exist.
                n, falls = len(values), sum(v['fallen'] for v in values)
                rmse, alive = np.mean([v['rmse'] for v in values]), np.mean([v['alive_s'] for v in values])
            else:
                values = [v for v in pilot if v['condition'] == condition and v['method'] == method]
                assert len(values) == 1, (condition, method)
                value = values[0]
                n, falls = int(value['n']), int(value['falls'])
                rmse, alive = float(value['mean_velocity_rmse_m_s']), float(value['mean_alive_s'])
            lo, hi = wilson(falls, n)
            extension_rows.append({'cohort': 'development_0_9', 'condition': condition, 'method': method,
                                   'n': n, 'falls': falls, 'fall_rate': falls/n,
                                   'wilson95_low': lo, 'wilson95_high': hi,
                                   'velocity_rmse': float(rmse), 'mean_alive_s': float(alive)})
    table('extensions', extension_rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for col, stream in enumerate(('joints', 'imu')):
        for method in METHODS:
            values = [next(r for r in extension_rows if r['method'] == method and
                           r['condition'] == f'drop_{stream}_{n}') for n in (1, 2, 4, 6)]
            axes[0, col].plot([1, 2, 4, 6], [v['velocity_rmse'] for v in values], 'o-', label=method, color=COLORS[method])
        axes[0, col].set(title=f'Одна серия потерь: {stream}; падений 0/10 во всех точках',
                        xlabel='Число потерянных пакетов', ylabel='RMSE скорости, м/с', xticks=[1, 2, 4, 6])
    for method in ('B1', 'M'):
        values = [next(r for r in extension_rows if r['method'] == method and r['condition'] == f'equal{d}') for d in (20, 40, 80, 120)]
        axes[1, 0].plot([20, 40, 80, 120], [v['fall_rate'] for v in values], 'o-', label=method, color=COLORS[method])
    axes[1, 0].set(title='Одинаковые задержки IMU и суставов; M2 не проверен', xlabel='Задержка, мс', ylabel='Доля падений')
    for index, method in enumerate(METHODS):
        values = [next(r for r in extension_rows if r['method'] == method and r['condition'] == c) for c in ('clean', 'jitter')]
        axes[1, 1].bar(np.arange(2)+(index-1)*.24, [v['velocity_rmse'] for v in values], .24, label=method, color=COLORS[method])
    axes[1, 1].set(title='Свежие пакеты / независимый jitter 0–40 мс', ylabel='RMSE скорости, м/с', xticks=[0, 1], xticklabels=['clean', 'jitter'])
    for ax in axes.flat:
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle('Дополнительные режимы: seed 0–9, N=10 на условие/метод. Одна потеря ≠ постоянная задержка.')
    save(fig, 'fault_modes')
    candidate_rows = []
    for folder in ('m2_h1', 'm2_periodic_h1', 'm2_followup_h1', 'm2_hybrid_h1'):
        selection = load(f'results/{folder}/selection.json')
        candidate_rows.extend({'candidate': name, 'source': folder, **values} for name, values in selection['candidates'].items())
    table('candidates', candidate_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    labels = [r['candidate'] for r in candidate_rows]
    axes[0].barh(labels, [r['falls']/r['n'] for r in candidate_rows])
    axes[0].set(xlabel='Доля падений', title='Все кандидаты: подбор seed 0–9, 60 эпизодов каждый')
    axes[1].barh(labels, [r['velocity_rmse'] for r in candidate_rows])
    axes[1].set(xlabel='Среднее RMSE скорости, м/с', title='Ошибки усечены до падения')
    save(fig, 'candidates')
    timings = load('results/m2_validation_h1/statistics.json')['timing_ms']
    timing_rows = [{'method': method, 'component': component, **values}
                   for method, components in timings.items() for component, values in components.items()]
    table('cpu_timing', timing_rows)
    table('run_modes', [
        {'mode': 'baseline', 'duration': '20 s default', 'algorithms': 'fresh', 'controls': 'none', 'realtime': 'no'},
        {'mode': 'sensor episode', 'duration': '20 s protocol', 'algorithms': 'B0/B1/B2/M/AJ/AI/M2', 'controls': 'saved schedule', 'realtime': 'no'},
        {'mode': 'interactive', 'duration': '20 s default', 'algorithms': 'B1/M', 'controls': 'I/J, mouse, Esc', 'realtime': 'yes'},
        {'mode': 'sandbox', 'duration': 'until Esc', 'algorithms': 'B1/M2 (or legacy M)', 'controls': 'W/C/R/I/J, mouse, Esc', 'realtime': 'yes'},
    ])
    (OUT/'tables'/'README.md').write_text('# Сводные таблицы\n\n'
        'Числа из сохранённых измерений. CSV доступны рядом с этим документом.\n'
        'Подбор и дополнительная проверка разделены; RMSE усечён до падения.\n\n'
        +'\n\n---\n\n'.join(TABLE_SECTIONS))
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
    for ax, component, title in zip(axes, ('adapter_ms', 'inference_ms', 'control_cycle_ms'), ('Адаптер', 'Политика', 'Полный цикл')):
        x = np.arange(3)
        for offset, metric in [(-.18, 'median'), (.18, 'p95')]:
            ax.bar(x+offset, [timings[m][component][metric] for m in METHODS], .36, label=metric)
        ax.set(title=title, ylabel='мс', xticks=x, xticklabels=METHODS)
        ax.legend()
    fig.suptitle('Intel i9-12900H, один поток; реальные такты 60 проверочных эпизодов на метод')
    save(fig, 'cpu_timing')
    source = ROOT/'scripts/build_report_assets.py'
    SOURCES[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {'source_sha256': SOURCES, 'new_simulations_run': False,
                'closed_test_seeds_used': False, 'missing_measurements': ['M2 equal-delay conditions'],
                'notes': ['RMSE truncated before fall', 'Development and validation are separate',
                          'ALL comprises six repeated conditions per seed, not 60 independent seeds']}
    (OUT/'assets_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Built 6 figures (PNG/PDF), 5 tables (CSV/Markdown), {len(SOURCES)} source hashes.')


if __name__ == '__main__':
    main()
