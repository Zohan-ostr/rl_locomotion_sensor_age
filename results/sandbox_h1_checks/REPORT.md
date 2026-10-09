# Свободный режим H1: W/C/R, без лимита времени

Запуск: `./scripts/launch_h1_sandbox.sh`. Launcher включает `--sandbox`;
без явно заданного `--seconds` длительность не ограничена. Выход — Esc
или закрытие окна. Исходный launcher сохраняет 20-секундную демонстрацию.

W переключает команду скорости 0,5/0 м/с; C — B1/M без сброса памяти
политики; R восстанавливает вертикальную позу в текущем XY и включает
нулевую команду. При R очищаются скорости, действие, память LSTM,
пакеты, текущий признак падения и фаза; общее время, выбранная компенсация
и история падений сохраняются. Новые пакеты после сброса получаются из
реального восстановленного состояния. I/J и камера работают как раньше.

## Проверки

- [checks.log](checks.log): три содержательные проверки прошли — клавиши
  с игнорированием REPEAT/RELEASE, реальный сброс MuJoCo/LSTM, аудит
  30-секундного запуска со сменой команды/метода и подъёмом после падения.
- [previous_checks.log](previous_checks.log): четыре прежние проверки
  демонстрации прошли. Чистый новый bounded-run воспроизводит исходное
  смещение baseline побитово: 8,343844 / 0,147613 м.
- [final_controls](final_controls/summary.json): 30 с, 1500 решений,
  блокировка IMU с 6 до 8 с, падение на 7,578 с, сброс на 8 с,
  команда стоять до 12 с, затем ходьба. После сброса нового падения нет.
  Память/предыдущее действие и фаза сброшены; возраст свежих показаний 0.
  Данные записаны двумя порциями 1000/500 решений, время строго возрастает.
- [window](window/summary.json): настоящее окно, 24 с симуляции за
  24,284 с реального времени. Помимо заданных событий были ручные W/R/I.
  Три сброса, падение на 22,992 с. Реальные события — window/events.jsonl.
  Обычный запуск не содержит проверочного 24-секундного ограничения.
- Bash launcher проверен `bash -n`; Python-файлы проверены `py_compile`.

Существующая `.venv`, CPU-only Torch, один вычислительный поток;
новые зависимости не устанавливались. [Версии](requirements.lock.txt).
В каждом summary.json записаны реальная команда, версии, CPU, commit
upstream и хеши исходников/моделей/весов.

## Ограничение стояния и сохранённые отрицательные результаты

Готовая политика при нулевой команде не фиксирует мировую позицию.
В stand за 30 с без сбоев не упала, но сместилась на −2,259 м по X
и +0,604 м по Y. Пользовательский режим STAND означает нулевую команду.

Проверены альтернативы статического PD-удержания: nominal-поза
(stand_pd) упала на 3,628 с, выпрямленная поза (stand_upright) — на
2,110 с. В controls_pd падения на 3,628 и 10,174 с. Эти варианты
отклонены и не включены в итоговый режим. Данные/логи сохранены;
исходная политика и штатный PD используются и при нулевой команде.
Точное стояние требует отдельного проверенного контроллера балансировки.

Сброс R — служебная телепортация в симуляции, не навык самостоятельного
подъёма. Ручные вмешательства и сбросы не являются независимой тестовой
серией. PD получает свежую локальную обратную связь; компенсация действует
только на вход политики. Закрытые seed не запускались.

## Команды, выполненные из корня проекта

```bash
chmod +x scripts/launch_h1_sandbox.sh
mkdir -p results/sandbox_h1_checks
scripts/launch_interactive_h1.sh --headless --auto-close --output results/sandbox_h1_checks/bounded > results/sandbox_h1_checks/bounded.log 2>&1
scripts/launch_h1_sandbox.sh --headless --seconds 30 --event 0:w --event 3:c --event 5:w --event 8:r --event 12:w --block imu:6:8 --output results/sandbox_h1_checks/controls > results/sandbox_h1_checks/controls.log 2>&1
scripts/launch_h1_sandbox.sh --headless --seconds 30 --event 0:w --output results/sandbox_h1_checks/stand > results/sandbox_h1_checks/stand.log 2>&1
scripts/launch_h1_sandbox.sh --seconds 24 --event 0:w --event 3:w --event 5:c --event 7:r --event 10:w --output results/sandbox_h1_checks/window > results/sandbox_h1_checks/window.log 2>&1
scripts/launch_h1_sandbox.sh --headless --seconds 30 --event 0:w --output results/sandbox_h1_checks/stand_pd > results/sandbox_h1_checks/stand_pd.log 2>&1
scripts/launch_h1_sandbox.sh --headless --seconds 30 --event 0:w --event 3:c --event 5:w --event 8:r --event 12:w --block imu:6:8 --output results/sandbox_h1_checks/controls_pd > results/sandbox_h1_checks/controls_pd.log 2>&1
scripts/launch_h1_sandbox.sh --headless --seconds 30 --event 0:w --output results/sandbox_h1_checks/stand_upright > results/sandbox_h1_checks/stand_upright.log 2>&1
scripts/launch_h1_sandbox.sh --headless --seconds 30 --event 0:w --event 3:c --event 5:w --event 8:r --event 12:w --block imu:6:8 --output results/sandbox_h1_checks/final_controls > results/sandbox_h1_checks/final_controls.log 2>&1
.venv/bin/python -I scripts/check_interactive_h1.py > results/sandbox_h1_checks/previous_checks.log 2>&1
.venv/bin/python -I scripts/check_h1_sandbox.py > results/sandbox_h1_checks/checks.log 2>&1
bash -n scripts/launch_h1_sandbox.sh
.venv/bin/python -I -m pip freeze > results/sandbox_h1_checks/requirements.lock.txt
.venv/bin/python -I -m py_compile scripts/interactive_h1.py scripts/check_h1_sandbox.py
```

Оконный запуск получил разрешение графического сеанса. Эксперименты с
альтернативным стоянием выполнялись на промежуточных версиях кода;
их хеши сохранены в соответствующих summary.json. Каталоги не перезаписывались.
