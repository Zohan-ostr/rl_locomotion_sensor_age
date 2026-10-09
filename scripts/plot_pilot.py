"""Publication-exportable pilot figures using only saved real measurements."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(fig, root, name):
    fig.savefig(root/(name+'.png'), dpi=180, bbox_inches='tight')
    fig.savefig(root/(name+'.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('results/pilot_h1'))
    args = p.parse_args()
    root = args.output
    summary = json.loads((root/'summary.json').read_text())
    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'B1': '#3b618a', 'M': '#bd5b33'}
    def group(condition, method):
        return next(g for g in summary['groups'] if g['condition'] == condition and g['method'] == method)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharey=True)
    for ax, prefix, label in zip(axes, ('J', 'I'), ('Joint delay; IMU fresh', 'IMU delay; joints fresh')):
        for method in ('B1', 'M'):
            groups = [group(f'{prefix}{d}', method) for d in (20, 40, 80)]
            rates = np.array([g['fall_rate'] for g in groups])
            intervals = np.array([g['wilson95'] for g in groups])
            ax.errorbar(np.array([20, 40, 80])+(-1 if method == 'B1' else 1), rates,
                        yerr=np.maximum(0, np.array([rates-intervals[:, 0], intervals[:, 1]-rates])),
                        marker='o', capsize=3, label=method, color=colors[method])
            for delay, g in zip((20, 40, 80), groups):
                ax.annotate(f'{g["falls"]}/10', (delay, g['fall_rate']), xytext=(0, 8 if method == 'B1' else -14),
                            textcoords='offset points', ha='center', fontsize=8, color=colors[method])
        ax.set(title=label, xlabel='Delay (ms)', ylim=(-.08, 1.12), xticks=[20, 40, 80])
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Fall fraction (Wilson 95% interval)')
    axes[1].legend()
    fig.suptitle('Exploratory pilot: seed 0–9; horizons tuned on these same seeds', fontsize=11)
    fig.tight_layout()
    save(fig, figures, 'E2_falls')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, stream in zip(axes, ('joints', 'imu')):
        for method in ('B1', 'M'):
            groups = [group(f'drop_{stream}_{n}', method) for n in (1, 2, 4, 6)]
            ax.plot([1, 2, 4, 6], [g['fall_rate'] for g in groups], 'o-', label=method, color=colors[method])
        ax.set(title=f'Lost {stream} packets', xlabel='Consecutive missing packets', xticks=[1, 2, 4, 6], ylim=(-.05, 1.05))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Fall fraction (10 episodes per point)')
    axes[1].legend()
    fig.suptitle('Burst-loss pilot: selected horizons; no closed test', fontsize=11)
    fig.tight_layout()
    save(fig, figures, 'E3_falls')
    # Predeclared seed 0 example, not chosen by best outcome.
    h = summary['horizon_selection']['selected']['joints_ms']
    folders = {'B1': root/'episodes/seed_0/J80/B1_H0_0', 'M': root/f'episodes/seed_0/J80/M_H{h}_{h}'}
    fig, axes = plt.subplots(4, 2, figsize=(11, 9), sharex='col')
    for column, method in enumerate(('B1', 'M')):
        folder = folders[method]
        s = json.loads((folder/'summary.json').read_text())
        with np.load(folder/'signals.npz') as data:
            t = data['decision_steps']*.002
            axes[0, column].plot(t, data['ages_s'][:, 0]*1000, label='joints')
            axes[0, column].plot(t, data['ages_s'][:, 1]*1000, label='IMU')
            axes[1, column].plot(t, data['q_true'][1:, 2], label='true')
            axes[1, column].plot(t, data['q_delivered'][:, 2], label='delivered', alpha=.8)
            axes[1, column].plot(t, data['q_predicted'][:, 2], label='policy input', linestyle='--')
        trajectory = np.genfromtxt(folder/'trajectory.csv', delimiter=',', names=True)
        axes[2, column].plot(trajectory['time_s'], np.rad2deg(trajectory['roll_rad']), label='roll')
        axes[2, column].plot(trajectory['time_s'], np.rad2deg(trajectory['pitch_rad']), label='pitch')
        axes[3, column].plot(trajectory['time_s'], trajectory['vx_yaw_m_s'], label='forward speed')
        axes[3, column].axhline(.5, color='gray', linestyle='--', label='command')
        for ax in axes[:, column]:
            if s['fall_time_s'] is not None:
                ax.axvline(s['fall_time_s'], color='black', linestyle=':', label='confirmed fall')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, loc='best')
        axes[0, column].set_title(f'{method}: seed 0, J80; T_alive={s["time_alive_s"]:.3f} s')
        axes[3, column].set_xlabel('Simulation time (s)')
    for row, ylabel in enumerate(('Data age (ms)', 'Left hip pitch (rad)', 'Body angle (deg)', 'Speed (m/s)')):
        axes[row, 0].set_ylabel(ylabel)
    fig.suptitle('Paired pilot example: each method has its own physical trajectory', fontsize=11)
    fig.tight_layout()
    save(fig, figures, 'seed0_J80')
    fig, ax = plt.subplots(figsize=(7, 3.5))
    methods = ['B0', 'B1', 'M', 'B2', 'AJ', 'AI']
    values = [summary['timing'][m]['measurements_ms']['adapter_ms']['p95'] for m in methods]
    ax.bar(methods, values, color=['#777777', '#3b618a', '#bd5b33', '#777777', '#777777', '#777777'])
    ax.set(ylabel='Adapter p95 (ms)', title='CPU pilot timing; threshold 20 ms (outside chart scale)')
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    save(fig, figures, 'adapter_timing')
    print(json.dumps({'figures': sorted(str(f) for f in figures.iterdir()), 'source': str(root/'summary.json')}))


if __name__ == '__main__':
    main()
