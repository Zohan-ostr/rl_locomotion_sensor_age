"""Generate a source-grounded report only after all planned runs complete."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np


def main():
    root = Path('results/m2_validation_h1')
    profile = json.loads((root/'validated_profile.json').read_text())
    development = json.loads((Path(profile['selection']['root'])/'comparison.json').read_text())
    validation = json.loads((root/'validation_summary.json').read_text())
    extra = json.loads((root/'extra_summary.json').read_text())
    audits = json.loads((root/'audit.json').read_text())
    assert len(audits) == 280 and len(extra) == 100
    candidates = []
    for path in ('results/m2_h1', 'results/m2_periodic_h1', 'results/m2_followup_h1', 'results/m2_hybrid_h1'):
        selection = json.loads((Path(path)/'selection.json').read_text())
        for name, values in selection['candidates'].items():
            candidates.append((path, name, values))
    assert len(candidates) == 13
    records = [json.loads(p.read_text()) for p in sorted((root/'episodes').glob('seed_*/*/*/summary.json'))]
    assert len(records) == 180
    differences, timings = {}, {}
    for reference in ('B1', 'M'):
        delta = []
        for seed in range(10, 20):
            old = [s for s in records if s['seed'] == seed and s['method'] == reference]
            new = [s for s in records if s['seed'] == seed and s['method'] == 'M2']
            assert len(old) == len(new) == 6
            delta.append((sum(s['fallen'] for s in old)-sum(s['fallen'] for s in new))/6)
        rng = np.random.default_rng(20261009)
        samples = np.mean(rng.choice(delta, size=(5000, 10), replace=True), axis=1)
        differences[reference+'-M2'] = {'mean': float(np.mean(delta)),
                                      'descriptive_95_interval': np.quantile(samples, [.025, .975]).tolist()}
    for method in ('B1', 'M', 'M2'):
        folders = sorted((root/'episodes').glob(f'seed_*/*/{method}'))
        vectors = {k: [] for k in ('adapter_ms', 'inference_ms', 'control_cycle_ms')}
        for folder in folders:
            with np.load(folder/'signals.npz') as x:
                for k in vectors:
                    vectors[k].append(x[k])
        timings[method] = {k: {'median': float(np.median(np.concatenate(v))),
                               'p95': float(np.percentile(np.concatenate(v), 95))} for k, v in vectors.items()}
    (root/'statistics.json').write_text(json.dumps({'paired_fall_differences': differences,
        'timing_ms': timings, 'bootstrap_resamples': 5000, 'bootstrap_seed': 20261009}, indent=2)+'\n')
    lines = ['# M2: история суставов и раздельный выбор моделей', '',
        '9 октября 2026. Реализован составной компенсатор: суставы прогнозируются по доставленной периодической истории, IMU сохраняет прежний M20. Готовые веса, PD, команда, формат и нормализация входа не менялись.', '',
        'Проведены 780 эпизодов подбора/составной проверки на seed 0–9, 180 дополнительных парных эпизодов на seed 10–19, 100 регрессионных эпизодов выбранного профиля и 6 начальных проверочных эпизодов. Основная численная работа — 1066 запусков; повторные кандидаты не увеличивают независимый N. Отдельные проверки CLI и интерактивного frontend перечислены ниже и не увеличивают N сравнения. Закрытые seed 100–139 не использовались.', '',
        '## Алгоритм', '',
        '- Суставы: до 60 доставленных измерений, Fourier fit с периодом 0,8 с, 3 гармониками, ridge=0,001. После накопления полного периода к последним q и dq прибавляется половина модельного изменения до sample_time + min(age, 0,12 с). Поправки ограничены 0,35 рад и 6 рад/с, положения ограничены XML. Поправка dq берётся из производной той же периодической модели и привязывается к измеренному dq; это две якорные поправки, а не точное интегрирование предсказанного dq.',
        '- IMU: прежнее вращение направления гравитации с отрицательным знаком omega, максимум 20 мс; omega удерживается. Более сложные IMU-прогнозы не дали выигрыша по выбранному критерию.',
        '- При нулевом возрасте — точное копирование, без клиппинга/перенормировки. Повторные удержания и старые приходы не расширяют историю. Истинное состояние, будущие пакеты, команды и действия адаптеру недоступны.', '',
        '## Все кандидаты на пилоте', '',
        '| Вариант | Падения / N | Среднее T_alive, с | Среднее RMSE, м/с |', '|---|---:|---:|---:|']
    for path, name, values in candidates:
        lines.append(f"| {name} | {values['falls']}/{values['n']} | {values['mean_alive_s']:.3f} | {values['velocity_rmse']:.3f} |")
    lines += ['', 'Вначале проверялись ускорительные варианты, затем Fourier, перенос изменений предыдущего цикла и затухание поправки с возрастом. Неудачные варианты сохранены. H80/H120 с одинаковым ускорительным окном эквивалентны на E2, где возраст не превышает 80 мс; их нельзя считать независимыми подтверждениями.', '',
              'Для составного профиля каждый поток выбирался отдельно по своим трём асинхронным E2-условиям: меньше падений, больше T_alive, меньше RMSE. Затем профиль повторён во всех 60 условиях. Наблюдения, действия и физика побитово совпали с источником выбранного компонента. Этот способ выбора добавлен до проверки seed 10–19 и отражён в docs/README.md#m2_development. Исходный M включён в сравнение моделей, чтобы сохранять удачные части.', '',
              '## Подбор и дополнительная проверка', '']
    for title, groups in [('Пилот seed 0–9 (использован для настройки)', development['groups']),
                           ('Seed 10–19 (профиль зафиксирован до запуска, без повторного подбора)', validation['groups'])]:
        lines += ['### '+title, '', '| Условие | B1 падения | M падения | M2 падения | B1/M/M2 RMSE, м/с | B1/M/M2 T_alive, с |',
                  '|---|---:|---:|---:|---|---|']
        for condition, values in groups.items():
            lines.append('| '+condition+' | '+' | '.join(f"{values[m]['falls']}/{values[m]['n']}" for m in ('B1', 'M', 'M2'))+
                         ' | '+' / '.join(f"{values[m]['velocity_rmse']:.3f}" for m in ('B1', 'M', 'M2'))+
                         ' | '+' / '.join(f"{values[m]['mean_alive_s']:.3f}" for m in ('B1', 'M', 'M2'))+' |')
        lines += ['']
    lines += ['Парная разность частот падения на seed 10–19 (положительная означает меньше падений у M2):', '']
    for label, values in differences.items():
        lo, hi = values['descriptive_95_interval']
        lines.append(f"- {label}: {values['mean']*100:.2f} п.п.; описательный 95% bootstrap-интервал [{lo*100:.2f}; {hi*100:.2f}] п.п.")
    lines += ['', 'Повторная выборка seed целиком, 5000 повторов, RNG 20261009. N=10 на условие. Это небольшая дополнительная проверка, а не исходная закрытая серия 40 seed. При совпадении всех пар bootstrap разности вырождается в [0;0]; это свойство наблюдаемой выборки, не доказательство эквивалентности методов. При 0/10 падений верхняя граница Wilson 95% около 27,75%; ноль не означает отсутствия риска. RMSE усечён до падения, поэтому уменьшение ошибки аварийных эпизодов нельзя считать улучшением устойчивости. Сравниваются собственные траектории методов.', '',
              '## Регрессии и CPU', '',
              f"Выбранный M2: clean/E3/jitter — {sum(s['fallen'] for s in extra)}/100 падений. Clean (10 seed) побитово совпал с сохранённым B1 по наблюдениям, действиям и физическим измерениям. Все 280 дополнительных эпизодов прошли аудит причинности/пакетов/нулевого возраста. Состояния и расписания трёх методов в E2 проверены по хешам.", '',
              '30 unit-проверок: 14 прежних, 7 ускорительных, 3 Fourier, 2 циклических, 2 затухания, 2 составных. Все прошли; логи — ../m2_h1_setup/.', '',
              '| Метод | Adapter median / p95, мс | Inference median / p95, мс | Cycle median / p95, мс |', '|---|---|---|---|']
    for method, values in timings.items():
        lines.append('| '+method+' | '+' | '.join(f"{values[k]['median']:.4f} / {values[k]['p95']:.4f}" for k in ('adapter_ms', 'inference_ms', 'control_cycle_ms'))+' |')
    example = records[0]
    lines += ['', 'В таблице объединены реальные такты 60 новых E2-эпизодов каждого метода. У M2 adapter_ms включает update истории/fit и predict. В первой ускорительной серии update входил только в cycle; её adapter_ms не используется для этого сравнения. Cycle исключает загрузку и финальную запись. Графическое окно измеряется отдельно.', '',
              f"Условия: CPU Intel i9-12900H, Python {example['python']}, Torch {example['torch']} (CUDA build={example['cuda_build']}), MuJoCo {example['mujoco']}, NumPy {example['numpy']}. Torch intra-op 1, inter-op 1; OMP/OpenBLAS/MKL 1. Проектная .venv, новых зависимостей нет. dt=2 мс, политика 50 Гц, команда [0,5; 0; 0], 20 с, разгон 2 с, возмущения суставов ±0,003 рад.", '',
              '## Ограничения и воспроизведение', '',
              '- Самый сложный I80 остаётся неустойчивым. Длинные ручные отключения, переходы идти/стоять и реальные датчики не покрываются этой численной проверкой. Периодическая модель предполагает похожие циклы шага, а не гарантирует восстановление после произвольного движения.',
              '- Сбои относятся только к входу политики; PD получает свежую локальную обратную связь. Перенос на реального робота не проверялся, веса H1 не являются воспроизведением литературных систем.',
              '- Старые результаты сохранены; закрытая серия не выполнена, hypothesis_supported оставлен null. Все отрицательные результаты учитываются. Профиль seed 10–19 не подстраивался по их исходам.', '',
              'Реальные команды: [commands.md](../m2_h1_setup/commands.md). Зависимости: [requirements.lock.txt](../m2_h1_setup/requirements.lock.txt). Исходники каждой завершённой стадии — source_snapshot/ в её каталоге; каждый summary.json фиксирует SHA-256 runner/адаптеров/весов/XML/мешей, upstream commit и условия запуска.', '',
              'Профиль до проверки: [frozen_profile.json](frozen_profile.json), после завершения: [validated_profile.json](validated_profile.json). Полные аудиты: [audit.json](audit.json), метрики: [validation_summary.json](validation_summary.json), статистика и CPU: [statistics.json](statistics.json), дополнительная серия: [extra_summary.json](extra_summary.json).', '',
              'Интерактивная инструкция: [docs/README.md#interactive_demo](../../docs/README.md#interactive_demo). Новый профиль выбирается параметром --compensator m2 и включён по умолчанию в launch_h1_sandbox.sh; прежний алгоритм доступен через --compensator legacy. Переключатель C по-прежнему выбирает B1 / выбранный компенсатор. Начальное состояние — B1.', '',
              'Отдельные проверки frontend: 20-секундный повтор CLI seed 10/J40, 30-секундный headless с C/W/R и короткой блокировкой, 16-секундное окно с теми же событиями. Окно: 16 с симуляции за 16,300 с реального времени, без падения; C/W/R работают. Полная 16-секундная часть наблюдений, действий, qpos/qvel и возрастов побитово совпала с headless. CLI M2 по умолчанию загрузил профиль 120/20 мс и побитово воспроизвёл проверочную траекторию. Лог — ../m2_h1_setup/frontend_checks.log. Эти три запуска — технические повторы, не новые статистические пары.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'validation': validation['groups']['ALL'], 'paired_differences': differences,
                      'timing_ms': timings, 'extra_falls': sum(s['fallen'] for s in extra)}, indent=2))


if __name__ == '__main__':
    main()
