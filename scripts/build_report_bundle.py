"""Create a compact, self-contained evidence bundle for writing a report."""
import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('baseline_h1', 'pilot_h1', 'm2_h1', 'm2_periodic_h1',
          'm2_followup_h1', 'm2_hybrid_h1', 'm2_validation_h1',
          'm2_h1_setup', 'interactive_h1_checks', 'sandbox_h1_checks', 'm3_h1')
EXAMPLES = (
    'results/baseline_h1/headless20',
    'results/pilot_h1/episodes/seed_0/J80/B1_H0_0',
    'results/pilot_h1/episodes/seed_0/J80/M_H20_20',
    'results/m2_validation_h1/episodes/seed_10/J40/B1',
    'results/m2_validation_h1/episodes/seed_10/J40/M',
    'results/m2_validation_h1/episodes/seed_10/J40/M2',
    'results/m3_h1/closed_primary/seed_100/J80/B1',
    'results/m3_h1/closed_primary/seed_100/J80/M2',
    'results/m3_h1/closed_primary/seed_100/J80/M3',
    'results/m3_h1/closed_primary/seed_100/I80/B1',
    'results/m3_h1/closed_primary/seed_100/I80/M3',
    'results/m3_h1/closed_extensions/seed_100/J60/B1',
    'results/m3_h1/closed_extensions/seed_100/J60/M3',
    'results/m3_h1/closed_joint/seed_100/J40I60/B1',
    'results/m3_h1/closed_joint/seed_100/J40I60/M3',
    'results/m3_h1/diagnostics/holdout_26/J80/B1',
    'results/m3_h1/diagnostics/holdout_26/I80/B1',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(destination):
    if destination.exists():
        raise FileExistsError(f'{destination} exists; use a new output directory')
    destination.mkdir(parents=True)
    sources = {}
    missing_links = []

    def copy(path):
        relative = path.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[str(relative)] = digest(path)

    # Preserve readable documentation, code, profiles and all aggregate results.
    for base in ('docs', 'reports', 'configs', 'scripts'):
        for path in sorted((ROOT/base).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                copy(path)
    for name in ('requirements.txt', 'references.bib', 'results/README.md'):
        copy(ROOT/name)
    for stage in STAGES:
        folder = ROOT/'results'/stage
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix in ('.md', '.json', '.jsonl', '.csv', '.txt', '.log'):
                copy(path)
        for subfolder in ('source_snapshot', 'figures', 'validation'):
            for path in sorted((folder/subfolder).rglob('*')):
                if path.is_file() and path.suffix in ('.py', '.png', '.pdf', '.json', '.log'):
                    copy(path)
        if stage == 'm3_h1':
            for subfolder in ('tables', 'frozen', 'models', 'logs', 'ages'):
                for path in sorted((folder/subfolder).rglob('*')):
                    if path.is_file():copy(path)
            for path in sorted((folder/'diagnostics').rglob('*.json')):copy(path)
            for path in sorted((folder/'checks').rglob('*')):
                if path.is_file() and path.suffix in ('.json','.log'):copy(path)
    copy(ROOT/'results/baseline_h1/UNITREE_LICENSE')

    # Every episode is represented, including failed candidates and technical repeats.
    metrics = []
    fields = ['stage', 'role', 'source_summary', 'source_sha256', 'candidate', 'seed', 'condition',
              'method', 'horizon_joints_ms', 'horizon_imu_ms', 'fallen', 'time_alive_s',
              'fall_time_s', 'censored', 'walking_success', 'velocity_rmse_m_s',
              'mean_forward_speed_m_s', 'joint_prediction_rmse_rad',
              'gravity_prediction_mean_angle_rad', 'wall_seconds_loop',
              'episode_wall_seconds_including_setup_and_output', 'simulation_steps',
              'simulation_seconds', 'fall_history_json',
              'adapter_median_ms', 'adapter_p95_ms', 'inference_median_ms',
              'inference_p95_ms', 'cycle_median_ms', 'cycle_p95_ms',
              'cohort', 'oracle_components_json', 'record_diagnostics', 'adapter_model_sha256',
              'adapter_config_json', 'ages_s_json']
    for stage in STAGES:
        for path in sorted((ROOT/'results'/stage).rglob('summary.json')):
            data = json.loads(path.read_text())
            # Top-level aggregate summaries do not describe an episode.
            if 'simulation_steps' not in data and 'simulation_seconds' not in data:
                continue
            relative = str(path.relative_to(ROOT))
            sources[relative] = digest(path)
            condition = data.get('condition', {})
            row = {key: data.get(key, '') for key in fields}
            role = 'technical_repeat'
            if 'probe_contaminated' in path.parts:
                role = 'excluded_probe_memory'
            elif 'episodes' in path.parts:
                role = 'additional_validation' if stage == 'm2_validation_h1' else \
                       'pilot' if stage == 'pilot_h1' else 'development'
            elif 'pilot_checks' in path.parts:
                role = 'regression'
            if stage == 'm3_h1':
                cohort=path.relative_to(ROOT/'results'/stage).parts[0]
                role={'training':'training', 'development':'tuning', 'development_joint':'development_joint',
                      'closed_primary':'closed_primary', 'closed_extensions':'closed_single_extension',
                      'closed_joint':'closed_joint_extension', 'diagnostics':'diagnostic',
                      'checks':'technical_repeat', 'smoke':'baseline_smoke'}.get(cohort,'technical_repeat')
                if data.get('oracle_components'):role='oracle_diagnostic'
                row['cohort']=cohort
            row.update(stage=stage, source_summary=relative, source_sha256=sources[relative],
                       oracle_components_json=json.dumps(data.get('oracle_components',[])),
                       adapter_config_json=json.dumps(data.get('adapter_config')),
                       ages_s_json=json.dumps(data.get('ages_s',{})),
                       role=role, fall_history_json=json.dumps(data.get('fall_history_s', [])),
                       condition=condition.get('id', '') if isinstance(condition, dict) else condition,
                       candidate=path.parent.name if stage == 'm3_h1' else path.parts[path.parts.index('episodes')+1]
                       if 'episodes' in path.parts and stage.startswith('m2_') and stage != 'm2_validation_h1' else '',
                       velocity_rmse_m_s=data.get('velocity_rmse_after_warmup_before_fall_m_s',
                                                 data.get('velocity_xy_rmse_after_2s_m_s', '')),
                       mean_forward_speed_m_s=data.get('mean_forward_speed_after_warmup_before_fall_m_s',
                           data.get('mean_vx_yaw_after_warmup_before_fall_m_s',
                                    data.get('mean_vx_yaw_after_2s_m_s', ''))))
            for prefix, key in [('adapter', 'adapter_ms'), ('inference', 'inference_ms'), ('cycle', 'control_cycle_ms')]:
                baseline_key = {'inference': 'policy_inference',
                                'cycle': 'control_cycle_including_metrics'}.get(prefix, key)
                timing = data.get('timing_ms', {}).get(key, data.get(key, data.get(baseline_key, {})))
                if isinstance(timing, dict):
                    row[prefix+'_median_ms'] = timing.get('median', timing.get('median_ms', ''))
                    row[prefix+'_p95_ms'] = timing.get('p95', timing.get('p95_ms', ''))
            metrics.append(row)
            if stage in ('baseline_h1', 'interactive_h1_checks', 'sandbox_h1_checks', 'm3_h1'):
                copy(path)
                if stage=='m3_h1' and (path.parent/'audit.json').exists():copy(path.parent/'audit.json')
        for path in (ROOT/'results'/stage).rglob('EXCLUDED.md'):
            copy(path)
    metrics_dir = destination/'metrics'
    metrics_dir.mkdir()
    with (metrics_dir/'all_episode_metrics.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    counts = {stage: sum(row['stage'] == stage for row in metrics) for stage in STAGES}
    (metrics_dir/'dataset_counts.json').write_text(json.dumps(counts, indent=2)+'\n')

    # Selected, explicitly named examples are illustrative, not additional independent evidence.
    for example in EXAMPLES:
        folder = ROOT/example
        for name in ('summary.json', 'trajectory.csv', 'signals.npz', 'diagnostics.npz'):
            if (folder/name).exists():
                copy(folder/name)
        summary = json.loads((folder/'summary.json').read_text())
        for name in ('schedule_path', 'initial_state_path'):
            if name in summary:
                copy(Path(summary[name]))
        if not (folder/'signals.npz').exists():
            continue
        signals = np.load(folder/'signals.npz')
        indices = np.searchsorted(signals['sample_steps'], signals['decision_steps'])
        assert np.array_equal(signals['sample_steps'][indices], signals['decision_steps'])
        columns = ['time_s', 'age_joints_ms', 'age_imu_ms', 'q_true_rad', 'q_delivered_rad',
                   'q_predicted_rad', 'dq_true_rad_s', 'dq_delivered_rad_s', 'dq_predicted_rad_s',
                   'g_true_x', 'g_true_y', 'g_true_z', 'g_delivered_x', 'g_delivered_y',
                   'g_delivered_z', 'g_predicted_x', 'g_predicted_y', 'g_predicted_z', 'action_hip_pitch',
                   'omega_true_x_rad_s', 'omega_true_y_rad_s', 'omega_true_z_rad_s',
                   'omega_delivered_x_rad_s', 'omega_delivered_y_rad_s', 'omega_delivered_z_rad_s',
                   'omega_predicted_x_rad_s', 'omega_predicted_y_rad_s', 'omega_predicted_z_rad_s']
        values = np.column_stack([
            signals['decision_steps'] * summary['upstream_config']['simulation_dt'], signals['ages_s']*1000,
            signals['q_true'][indices, 2], signals['q_delivered'][:, 2], signals['q_predicted'][:, 2],
            signals['dq_true'][indices, 2], signals['dq_delivered'][:, 2], signals['dq_predicted'][:, 2],
            signals['g_true'][indices], signals['g_delivered'], signals['g_predicted'], signals['actions'][:, 2],
            signals['omega_true'][indices], signals['omega_delivered'], signals['omega_predicted']])
        np.savetxt(destination/example/'diagnostic_signals.csv', values, delimiter=',',
                   header=','.join(columns), comments='')

    # Keep the original installation guide as plain text; the package has one primary README.
    shutil.copy2(ROOT/'README.md', destination/'project_guide.txt')
    sources['README.md'] = digest(ROOT/'README.md')
    guide = '''# Материалы для подготовки исследовательского отчёта

Актуальный самостоятельный комплект: baseline, M, M2 и завершённое исследование
M3 на Unitree H1 в MuJoCo на CPU. Начните с M3; старые отчёты описывают
исторические этапы. Их фразы «закрытая серия не выполнена» относятся к
состоянию до M3. Сборка этого комплекта не запускает новых экспериментов.

## Порядок чтения

1. [Отчёт M3: результаты, диагностика, ограничения и перспективы](results/m3_h1/REPORT.md).
2. [Основной критерий и интервал](results/m3_h1/primary_conclusion.json),
   [все сравнительные таблицы](results/m3_h1/tables/groups.csv),
   [парные разности](results/m3_h1/tables/paired_differences.csv).
3. [Падения](results/m3_h1/figures/falls.png),
   [парные разности](results/m3_h1/figures/paired_differences.png),
   [совместные задержки](results/m3_h1/figures/joint_delays.png).
   Все семь рисунков PNG/PDF — в `results/m3_h1/figures/`.
4. [История baseline/M/M2](reports/REPORT.md), её таблицы и графики.
5. [Теория и протокол](docs/README.md), [библиография](references.bib).
6. [Метрики каждого запуска всех этапов](metrics/all_episode_metrics.csv).

## Задание для отдельного чата

Подготовь подробный исследовательский отчёт на русском языке по материалам
архива. Опиши постановку задачи, готовую рекуррентную политику H1, CPU-окружение,
модель асинхронных пакетов, методы B1/M/M2/M3, oracle-диагностику, обучение
ridge-прогнозатора, абляцию действий, подбор и проверку замороженной модели.
Приведи реальные результаты с графиками и таблицами, указывая файлы-источники,
N, единицы, интервалы и правила усечения. Падения — основной показатель;
точность восстановления, действия и отслеживание скорости оценивай отдельно.
Объясни, где M3 помогает и где ухудшает результат. Заверши ограничениями
и проверяемыми перспективами дальнейшего исследования.

Разделяй факт литературы, собственную интерпретацию, гипотезу, настройку
и измеренный результат. Не выдумывай данные и недоступные детали статей.
Не называй готовый Unitree checkpoint воспроизведением работ на других роботах.
Не объявляй экстраполяцию или работу с памятью впервые предложенными.
Исторические отрицательные результаты и неудачные кандидаты не скрывай.
Если материала недостаточно, явно перечисли пробелы.

## Ключевые результаты и границы выводов

- Linux, Intel i9-12900H, проектное CPU-окружение; веса RL-политики,
  физика, PD, нормализация, команда и правило падения между методами сохранены.
- Задерживаются только наблюдения политики; локальный PD использует свежие
  показания. Поток суставов — q/dq десяти суставов; IMU — omega и направление g,
  а не сырые ускорения. Политика 50 Гц, физика/PD 500 Гц.
- Исторический M2: seed 10–19, у B1/M/M2 по 20/60 падений. Улучшение RMSE
  при J20/J40 не подтвердило снижение падений.
- M3: диагностика и обучение на seed 20–25, подбор на seed 26–29.
  Четыре заранее записанных кандидата: действия включены/выключены × вес 0,5/1.
  Выбран вариант без истории действий, вес 0,5, ridge=1, горизонт 80/80 мс.
- Профиль, модель и код зафиксированы до открытия seed 100–139.
  Выполнены 960 основных закрытых эпизодов, 320 отдельных J60/I60 и
  120 совместных на seed 100–109. После открытия перенастройки не было.
- Исходные шесть E2: B1/M/M2/M3 — 80/79/79/40 падений из 240 на метод.
  Основной N=40 seed; 960 запусков не равны 960 независимым состояниям.
  Снижение B1−M3 16,67 п.п. Формальный критерий ≥10 п.п. и нижняя граница
  парного 95% интервала >0 выполнен. Bootstrap выбирает seed целиком.
- Интервал [16,67;16,67] п.п. вырожден: разность исходов одинакова у всех
  40 seed. Он не означает нулевой неопределённости вне малых начальных
  возмущений ±0,003 рад, одной скорости, пола и checkpoint.
- I80: M3 0/40 падений, RMSE 0,174 м/с; B1 40/40. J80: M3 и B1 40/40
  падений, среднее время M3 6,693 с против B1 7,845 с. При J20/J40
  M3 точнее восстанавливает q/dq, но хуже отслеживает скорость.
- J60: B1 14/40 падений, M3 0/40; RMSE M3 0,212 м/с, поэтому успешной
  ходьбы по порогу 0,2 нет. I60: падений нет у обоих, M3 улучшает RMSE
  0,237 → 0,142 м/с. Это расширения, не исходная проверка гипотезы.
- Совместные J40I40/J60I40/J40I60: N=10 seed на условие, M3 без падений.
  При J40I40 RMSE M3 хуже B1. Все условия анализируются отдельно.
- 50/60 мс дают одинаковый возраст 60 мс на входе политики; 70/80 — 80 мс.
  Номинально разные задержки с одинаковыми входами не независимы.
- Oracle — только диагностика. Рабочий M3 не получает текущую истину.
  Общая траектория и одинаковая копия LSTM оценивают локальное влияние входа,
  а замкнутые эпизоды методов имеют разные физические траектории.
- RMSE усечён до падения после 2 с; T_alive ограничено 20 с. Меньшая ошибка
  аварийного эпизода не доказывает устойчивости. Вторичный анализ одинаковых
  окон скорости отмечен отдельно и не участвовал в подборе.
- M3: adapter median/p95 0,436/0,659 мс; полный цикл 1,796/2,682 мс.
  Это измерения данного CPU, не гарантия времени в иной среде.
- Seed 0–29 и 100–139 просмотрены. Следующему варианту нужны новые данные
  разработки и новая независимая проверка. Перенос на реального робота не проверен.

## Где искать доказательства

`results/m3_h1/REPORT.md` содержит итог и исходную регистрацию протокола.
`plan.json`, `selection.json` и `models/training.json` фиксируют условия,
все кандидаты и обучающие строки. `frozen/` содержит профиль, ridge-модель,
план проверки, код, SHA-256 и отметку открытия серии. Поле `test_opened:false`
сохраняет статус на момент заморозки; фактическое открытие — `opened.json`.

`tables/episodes.csv` содержит 1672 строки разработки и закрытой проверки;
`tables/groups.csv` — сводные метрики; `paired_differences.csv` — парные 95%
интервалы; `ages.csv` — фактические возраста; `cpu_timing.csv` — времена;
`common_trajectory.csv` и `common_action_timeline.csv` — диагностика действий;
`matched_velocity_*` — вторичный анализ одинаковых окон.
Данные обучения, oracle и технические повторы не входят в основную таблицу
1672 эпизодов; они представлены в общей таблице `metrics/` и своих manifests.

Включены все индивидуальные `summary.json` и аудиты M3: обучение, кандидаты,
oracle, закрытые серии и технические повторы. История других этапов представлена
отчётами, агрегатами и метриками каждого запуска. `metrics/dataset_counts.json`
считает manifests, включая повторы, а не независимые испытания. Пустые значения
в CSV означают отсутствующее измерение, не ноль.

Полные `trajectory.csv`/`signals.npz` включены для явно выбранных примеров:
baseline20; seed 0/J80 B1/M; seed 10/J40 B1/M/M2; seed 100/J80 B1/M2/M3,
I80 B1/M3, J60 B1/M3, J40I60 B1/M3; общие диагностические траектории seed 26
J80/I80 B1 с сохранённой LSTM. Это иллюстрации, не дополнительная выборка.
Рядом `diagnostic_signals.csv`: q/dq левого hip-pitch, omega/g,
возраста и действие; истина сопоставлена по decision_steps.

`commands.json`, `runs.jsonl` и `logs/` сохраняют реальные вызовы и логи,
включая прерванную подготовку данных и исправленный сбой скрипта анализа.
`requirements.lock.txt` — фактические зависимости. `final_audit.json` и
`artifact_audit.json` — проверки. `scripts/` позволяет проверить формулы.

Не включены окружение, upstream, веса RL-политики, видео, полные массивы
всех тысяч эпизодов и большая часть расписаний. Обе небольшие ridge-модели
и замороженная модель M3 включены. Это комплект для отчёта, а не полностью
автономный установочный архив симулятора. `project_guide.txt` содержит
исходную инструкцию установки. `manifest.json` фиксирует SHA-256 файлов
и использованных источников; отсутствующие компактные ссылки обозначены явно.
'''
    original = (ROOT/'README.md').read_text()
    guide += '\n\n' + original[original.index('<a id="third_party"></a>'):]
    (destination/'README.md').write_text(guide)

    # Resolve all Markdown references within the compact package without inventing artifacts.
    for file in destination.rglob('*.md'):
        def resolve(match):
            raw = match.group(2)
            if '://' in raw or raw.startswith('mailto:'):
                return match.group(0)
            target = raw.split('#')[0]
            if target and not (file.parent/target).exists():
                missing_links.append({'document': str(file.relative_to(destination)), 'omitted_target': raw})
                return match.group(1) + f' (в исходном проекте: `{raw}`)'
            return match.group(0)
        file.write_text(re.sub(r'\[([^\]]+)\]\(([^)]+)\)', resolve, file.read_text()))
    manifest = {'purpose': 'report evidence bundle', 'new_experiments_run': False,
                'closed_test_seeds_used': True, 'closed_primary_seed_count': 40,
                'closed_primary_episode_count': 960, 'closed_extension_episode_count': 440,
                'source_sha256': sources,
                'episode_manifest_counts': counts, 'omitted_link_targets': missing_links,
                'illustrative_examples': EXAMPLES,
                'files': [{'path': str(p.relative_to(destination)), 'bytes': p.stat().st_size,
                           'sha256': digest(p)} for p in sorted(destination.rglob('*')) if p.is_file()]}
    (destination/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output': str(destination), 'files': len(manifest['files'])+1,
                      'episode_rows': len(metrics), 'bytes': sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'for_gpt')
    parser.add_argument('--zip', action='store_true', help='Also create a ZIP beside the folder')
    parser.add_argument('--replace', action='store_true', help='Replace an existing generated bundle after building a new one')
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.is_symlink():raise ValueError('Bundle destination must not be a symlink')
    if destination.exists() and not args.replace:raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix='.report_bundle_',dir=destination.parent) as temporary:
        staging=Path(temporary)/destination.name
        build(staging)
        archive=None
        if args.zip:
            archive=Path(shutil.make_archive(str(Path(temporary)/'archive'),'zip',
                         root_dir=staging.parent,base_dir=staging.name))
        previous=Path(temporary)/'previous_bundle'
        if destination.exists():destination.rename(previous)
        try:staging.rename(destination)
        except BaseException:
            if previous.exists():previous.rename(destination)
            raise
        if archive is not None:
            target=destination.with_suffix('.zip')
            archive.replace(target);print(target)


if __name__ == '__main__':
    main()
