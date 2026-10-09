# Реальные команды разработки M2

Рабочий каталог `/home/zohan/projects/rl_locomotion_sensor_age`, проектная `.venv`.
Новые зависимости не устанавливались. Численные серии последовательные,
без sleep и рендеринга, CPU-only; OMP/OpenBLAS/MKL/Torch по одному потоку.

```bash
mkdir -p results/m2_h1_setup
.venv/bin/python -I scripts/check_history_compensator.py > results/m2_h1_setup/unit_checks.log 2>&1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I scripts/tune_history_compensator.py > results/m2_h1_setup/tuning.log 2>&1
.venv/bin/python -I scripts/check_sensor_age.py > results/m2_h1_setup/legacy_checks.log 2>&1
.venv/bin/python -I -m pip freeze > results/m2_h1_setup/requirements.lock.txt
.venv/bin/python -I scripts/check_periodic_compensator.py > results/m2_h1_setup/periodic_unit_checks.log 2>&1
.venv/bin/python -I scripts/check_cycle_compensator.py > results/m2_h1_setup/cycle_unit_checks.log 2>&1
.venv/bin/python -I scripts/check_conservative_compensator.py > results/m2_h1_setup/confidence_unit_checks.log 2>&1
.venv/bin/python -I scripts/analyze_m2.py --output results/m2_h1 > results/m2_h1_setup/acceleration_analysis.log 2>&1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I scripts/tune_history_compensator.py --family periodic --output results/m2_periodic_h1 > results/m2_h1_setup/periodic_tuning.log 2>&1
.venv/bin/python -I scripts/analyze_m2.py --output results/m2_periodic_h1 > results/m2_h1_setup/periodic_analysis.log 2>&1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I scripts/tune_m2_followup.py > results/m2_h1_setup/followup_tuning.log 2>&1
.venv/bin/python -I scripts/analyze_m2.py --output results/m2_followup_h1 > results/m2_h1_setup/followup_analysis.log 2>&1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I scripts/tune_hybrid_compensator.py > results/m2_h1_setup/hybrid_tuning.log 2>&1
.venv/bin/python -I scripts/check_hybrid_compensator.py > results/m2_h1_setup/hybrid_unit_checks.log 2>&1
.venv/bin/python -I scripts/analyze_m2.py --output results/m2_hybrid_h1 > results/m2_h1_setup/hybrid_analysis.log 2>&1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I scripts/validate_m2.py > results/m2_h1_setup/validation.log 2>&1
```

Между сериями изменялась реализация runner для подключения новых семейств.
Перед изменением исходники завершённой серии копировались в её
`source_snapshot/`. SHA-256 каждого реально использованного runner и
адаптера, входов, весов и ассетов сохранён в summary.json каждого эпизода.
Старые baseline/pilot эпизоды не перезаписывались.

Первый ускорительный пилот измерял в adapter_ms только predict, а обновление
истории входило в полный cycle. Последующие серии включают update + predict
в adapter_ms; первое значение нельзя сравнивать с ними как полные расходы.

После завершения численной проверки:

```bash
cp results/m2_validation_h1/validated_profile.json configs/m2_profile.json
.venv/bin/python -I scripts/report_m2.py > results/m2_h1_setup/report.log 2>&1
MPLCONFIGDIR=/tmp/h1-pilot-mpl .venv/bin/python -I scripts/plot_m2_comparison.py > results/m2_h1_setup/plotting.log 2>&1
scripts/launch_h1_sandbox.sh --headless --seconds 30 --event 3:c --event 5:w --event 6:r --event 8:w --block joints:10:10.12 --output results/m2_validation_h1/interactive_checks > results/m2_h1_setup/interactive_m2.log 2>&1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I scripts/run_sensor_episode.py --method M2 --initial results/m2_validation_h1/inputs/seed_10/initial.json --schedule results/m2_validation_h1/inputs/seed_10/J40.json --output results/m2_validation_h1/cli_check > results/m2_h1_setup/cli_check.log 2>&1
scripts/launch_h1_sandbox.sh --seconds 16 --event 3:c --event 5:w --event 6:r --event 8:w --block joints:10:10.12 --output results/m2_validation_h1/window_check > results/m2_h1_setup/window_check.log 2>&1
.venv/bin/python -I scripts/check_interactive_h1.py > results/m2_h1_setup/interactive_regression.log 2>&1
.venv/bin/python -I scripts/check_h1_sandbox.py > results/m2_h1_setup/sandbox_regression.log 2>&1
.venv/bin/python -I scripts/check_m2_frontend.py > results/m2_h1_setup/frontend_checks.log 2>&1
.venv/bin/python -I scripts/report_m2.py > results/m2_h1_setup/report.log 2>&1
bash -n scripts/launch_h1_sandbox.sh
git diff --check
```

Оконный запуск получил разрешение графического сеанса. Параметры адаптера
извлечены из profile JSON в configs/m2_adapter.json через стандартные
json/Path проектного Python. Для обычного M2 CLI этот файл не обязателен:
профиль загружается автоматически при --method M2.
