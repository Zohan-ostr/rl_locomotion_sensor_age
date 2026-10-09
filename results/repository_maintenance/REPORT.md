# Оформление и очистка репозитория

9 октября 2026. Исследовательские серии не перезапускались; новые данные
получены только в технических smoke/reproduction-проверках.

## Что изменено

- `reports/REPORT.md`: общий отчёт, измеренные результаты, ограничения,
  перспективы и рекомендуемый следующий маршрут.
- `reports/figures/`: 6 сводных рисунков в PNG/PDF;
  `reports/tables/`: 5 таблиц в CSV/Markdown, включая режимы запуска.
- `scripts/build_report_assets.py`: построение из реальных агрегатов
  с SHA-256 источников; пропущенные измерения не заменены вымыслом.
- `requirements.txt`: точные версии существующего рабочего окружения,
  явный CPU build Torch.
- `scripts/setup.sh`, `verify_setup.py`: отдельная `.venv`, установка,
  закреплённый upstream, проверка зависимостей/26 ассетов и ходьбы.
- README: системные предпосылки, установка одной командой, интерактивное
  управление, baseline, парное сравнение, новая пилотная серия и диагностика.
- Удалены 19 файлов Python-кэша (148997 байт); `.cache` и новые пользовательские
  запуски исключены из Git. Перечень — cleanup.json.

Сохранены все исследовательские данные, реальные логи, отрицательные
результаты, альтернативы алгоритмов, лицензия и snapshots. Они нужны
для воспроизводимости и используют действующие пути. Большой локальный
объём остаётся у `.venv`, upstream и записанных сигналов; это не мусор.
Публикация, коммит и отправка не выполнялись.

## Проверки

- `setup.sh --offline` прошёл дважды; pip check не нашёл конфликтов;
  версии всех 30 Python-пакетов совпали с requirements.
- CPU Torch: CUDA build=None; 26 ассетов совпали по SHA-256.
- 5-секундный baseline: без падения, dx=1,875 м, RMSE после разгона≈0,150 м/с.
- Повторно прошли 30 unit-проверок пакетов/компенсаторов.
- Команды README J40/B1, J40/M20 и J40/M2 реально выполнены.
  Физические и policy-сигналы побитово совпали с исходными проверочными
  эпизодами seed 10; временные затраты ожидаемо отличаются.
- Проверены 42 строки comparison.csv и SHA-256 всех 10 источников manifest.
- PNG визуально просмотрены: падения и дополнительные режимы сбоев.
- Requirements успешно разобран через pip --dry-run --no-index на
  существующем окружении. Shell-синтаксис установщика/launcher проверен.

Полная установка в пустое окружение с загрузкой всех пакетов отдельно
не выполнена. GUI повторно в этом сеансе не открывался; ранее проверенные
реальные оконные логи сохранены. Закрытые seed не использованы.

Команды — раздел ниже; фактические результаты проверок — соседние *.log.

Финальная проверка: 84 локальные ссылки без ошибок, 33 Python-файла
успешно разобраны; по 6 PNG/PDF и по 5 CSV/Markdown. Результат —
final_checks.json. Повторная проверка таблиц и J40-сигналов также прошла.


---

<a id="commands"></a>

# Команды оформления и технической проверки

Выполнено из корня проекта, в существующей `.venv`. Сами записи файлов
отчёта/README и исходники генератора находятся в репозитории. Здесь команды
проверок, не вымышленные примеры запусков.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I scripts/build_report_assets.py > results/repository_maintenance/build_assets.log 2>&1
./scripts/setup.sh --offline > results/repository_maintenance/setup_offline.log 2>&1
bash -n scripts/setup.sh scripts/launch_h1_sandbox.sh scripts/launch_interactive_h1.sh

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I scripts/check_sensor_age.py > results/repository_maintenance/check_sensor_age.log 2>&1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I scripts/check_history_compensator.py > results/repository_maintenance/check_history.log 2>&1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I scripts/check_periodic_compensator.py > results/repository_maintenance/check_periodic.log 2>&1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I scripts/check_cycle_compensator.py > results/repository_maintenance/check_cycle.log 2>&1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I scripts/check_conservative_compensator.py > results/repository_maintenance/check_conservative.log 2>&1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I scripts/check_hybrid_compensator.py > results/repository_maintenance/check_hybrid.log 2>&1

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I scripts/run_sensor_episode.py --initial results/m2_validation_h1/inputs/seed_10/initial.json --schedule results/m2_validation_h1/inputs/seed_10/J40.json --method B1 --horizon-joints-ms 20 --horizon-imu-ms 20 --output results/repository_maintenance/J40_B1 > results/repository_maintenance/J40_B1.log 2>&1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I scripts/run_sensor_episode.py --initial results/m2_validation_h1/inputs/seed_10/initial.json --schedule results/m2_validation_h1/inputs/seed_10/J40.json --method M --horizon-joints-ms 20 --horizon-imu-ms 20 --output results/repository_maintenance/J40_M > results/repository_maintenance/J40_M.log 2>&1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I scripts/run_sensor_episode.py --initial results/m2_validation_h1/inputs/seed_10/initial.json --schedule results/m2_validation_h1/inputs/seed_10/J40.json --method M2 --output results/repository_maintenance/J40_M2 > results/repository_maintenance/J40_M2.log 2>&1

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I scripts/check_report_assets.py > results/repository_maintenance/check_assets.log 2>&1
.venv/bin/python -I -m pip install --dry-run --no-index -r requirements.txt > results/repository_maintenance/requirements_dry_run.log 2>&1
```

Первый `setup.sh --offline` также выполнен без внешнего перенаправления;
оба запуска имеют собственные логи в `results/setup/` и каждый выполнил
реальный пятисекундный smoke-run. Это повторы, не новые статистические пары.

Кэши удалены Python-скриптом: рекурсивный поиск `__pycache__` только в
scripts/docs/configs/results, `shutil.rmtree` найденных кэшей; поиск
`.DS_Store` вне `.venv` и vendor. Точный список файлов и объём — cleanup.json.
Данные эпизодов и исторические source_snapshot не удалялись.

Финально выполнены `bash -n scripts/setup.sh scripts/launch_h1_sandbox.sh
scripts/launch_interactive_h1.sh`, `git diff --check` и повтор
`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I scripts/check_report_assets.py`.
Локальные ссылки проверены Python-скриптом через регулярное извлечение
Markdown-целей и Path.exists; все scripts/*.py разобраны ast.parse.
Результат записан в final_checks.json.

## Упрощение документации

Документы объединены: по одному Markdown-файлу в каждом каталоге проекта,
18 файлов вместо 38 до удаления старого задания. В корне остался README,
в docs — единый README с якорями, в reports — общий REPORT, в reports/tables —
единый README. Лицензии включены в корневой README. Дублирующие исторические
HTML/PDF-экспорты удалены; исследовательский текст сохранён в docs/README.

Генератор сводных таблиц больше не создаёт отдельные Markdown-файлы.
Пилотный анализ обновляет раздел таблиц внутри REPORT, сохраняя команды.
Проверены замена раздела без дублирования, сохранение команд, 122 локальные
ссылки и якоря, синтаксис 34 Python-файлов и значения сводных таблиц.
Численные данные, расписания, логи измерений и snapshots не изменены.
