"""Standalone figures from real development and additional-validation logs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path('results/m2_validation_h1')
    profile = json.loads((root/'validated_profile.json').read_text())
    old = json.loads((Path(profile['selection']['root'])/'comparison.json').read_text())['groups']
    new = json.loads((root/'validation_summary.json').read_text())['groups']
    folder = root/'figures'
    folder.mkdir(exist_ok=True)
    conditions = ['J20', 'J40', 'J80', 'I20', 'I40', 'I80']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(6)
    for row, (title, groups) in enumerate([('Подбор: seed 0–9', old), ('Проверка: seed 10–19', new)]):
        for i, method in enumerate(('B1', 'M', 'M2')):
            offset = (i-1)*.24
            falls = [groups[c][method]['falls']/groups[c][method]['n'] for c in conditions]
            axes[row, 0].bar(x+offset, falls, .24, label=method)
            rmse = [groups[c][method]['velocity_rmse'] for c in conditions]
            axes[row, 1].bar(x+offset, rmse, .24, label=method)
        axes[row, 0].set(title=title+' — падения', ylabel='Доля падений', ylim=(0, 1.1))
        axes[row, 1].set(title=title+' — скорость', ylabel='Среднее эпизодальное RMSE, м/с')
        for ax in axes[row]:
            ax.set_xticks(x, conditions)
            ax.legend()
            ax.grid(axis='y', alpha=.2)
    fig.suptitle('N=10 на условие и метод. RMSE аварийных эпизодов усечён до падения.')
    for extension in ('png', 'pdf'):
        fig.savefig(folder/('comparison.'+extension), dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
