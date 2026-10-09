# Проверка интерактивной демонстрации

Реализованы `scripts/interactive_h1.py` и `scripts/launch_interactive_h1.sh`.
Инструкция: [docs/README.md#interactive_demo](../../docs/README.md#interactive_demo).
Существующие baseline/pilot runner и результаты не изменялись.

Два потока с настоящими начальными пакетами, 50 Гц. Удержание I/J теряет
новые пакеты IMU/суставов, сохраняет последнее доставленное показание;
после отпускания свежий пакет поступает на следующем такте. Обе клавиши
работают одновременно. Потеря фокуса снимает блокировку. По умолчанию B1;
опциональный M использует ранее выбранные горизонты 20/20 мс.
PD использует свежую обратную связь. Команда [0,5; 0; 0], физика 500 Гц,
готовая CPU-политика, неизменные нормализации/действия/PD.

Камера следует за корпусом на расстоянии 3,5 м; ЛКМ меняет только азимут
и угол возвышения. Эпизод длится 20 секунд симуляции, включая время после
зарегистрированного падения. Итоговый кадр остаётся до закрытия окна,
если не указан --auto-close. Headless не содержит real-time sleep.

## Выполненные запуски

| Каталог | Режим | Симуляция, с | Цикл с окном/без окна, с | Падение, с |
|---|---|---:|---:|---:|
| clean | B1, headless, без сбоев | 20 | 0,560 | нет |
| blocks | B1, headless, по 6 потерь каждого потока | 20 | 0,561 | нет |
| compensation | M, headless, те же интервалы | 20 | 0,556 | нет |
| window | Первое окно, 1000×750, 50 Гц | 20 | 111,480 | нет |
| window_fast | 640×480, 25 Гц, без теней/отражений | 20 | 28,141 | 6,644 |
| window_final | 640×480, 12,5 Гц, прежний план ожиданий | 20 | 24,835 | 5,724 |
| window_realtime | Финальный цикл с привязкой к настенным часам | 20 | 20,300 | нет |

Во всех оконных запусках кроме заданных интервалов были ручные нажатия.
Их нельзя сравнивать как одинаковые условия: различаются расписания
и физические траектории. Отрицательные результаты сохранены.
В последнем запуске потеряны 261 суставный и 62 IMU-пакета,
максимальные возраста — 1,06 и 0,50 с соответственно; падения нет.
Это единичная демонстрация, не оценка вероятности падения.

Рендеринг оптимизирован: финальный размер 640×480, 12,5 Гц, без теней
и отражений, программный OpenGL. Физика и политика не замедлены по частоте
симуляционного времени. Между кадрами контроллер догоняет расписание
настенных часов. На проверенном CPU цикл близок к реальному времени;
при высокой нагрузке/большом окне возможна задержка. Число 20,300 с
включает подготовку окна и цикл, исключает загрузку модели/политики до него,
запись итоговых файлов и ожидание на итоговом кадре.

## Проверки и воспроизводимость

- 14 существующих проверок причинности, очередей и компенсации прошли:
  [sensor_checks.log](sensor_checks.log).
- Четыре проверки демонстрации прошли: удержание обоих потоков и отпускание,
  PRESS/REPEAT/RELEASE и потеря фокуса, вращение без изменения расстояния/центра,
  аудит всех семи настоящих запусков: [checks.log](checks.log).
- На свежих данных все 1000 записанных положений/наклонов и итоговое смещение
  побитово совпали с исходным baseline. Смещение: 8,343844 м по X,
  0,147613 м по Y, падения нет.
- В автоматических коротких блокировках потеряны ровно шесть пакетов
  каждого потока, максимальный возраст 120 мс. На свежих тактах возраст
  возвращается к нулю. Проверено равенство age = now − sample_time.
- Все запуски завершили 1000 решений политики, без предупреждений MuJoCo.
- Проверена синтаксическая корректность bash launcher.

CPU Intel Core i9-12900H, Python 3.10.12, Torch 2.5.1+cpu, MuJoCo 3.2.7,
NumPy 1.26.4; PyTorch и OMP/OPENBLAS/MKL по одному потоку.
Новые зависимости не устанавливались. [Полный список](requirements.lock.txt).
Фактические CPU/affinity, upstream commit, хеши весов, конфигурации,
моделей и исходников сохранены в summary.json каждого запуска.
Память LSTM после проверочного forward восстановлена до начала движения.

[Реальные команды](REPORT.md#commands). В каждом каталоге: stdout log рядом с
каталогом, trajectory.csv, signals.npz, key_events.json и summary.json.
Оконные логи содержат фактически применённые ручные события; движение после
падения сохраняется. Закрытая тестовая серия не запускалась.

## Дальнейшие эксперименты

Записанные ручные интервалы можно воспроизводить в парных B1/M-запусках.
Полезны отдельные регуляторы постоянной задержки/jitter, систематическая
проверка длительности и фазы блокировок, одновременные сбои двух потоков
и отдельно проверяемый аварийный режим при большом возрасте наблюдения.
Демонстрация относится к входу этой политики в симуляции со свежим PD;
перенос на реальные датчики и робота здесь не проверяется.


---

<a id="commands"></a>

# Реальные команды проверки демонстрации

Рабочий каталог: `/home/zohan/projects/rl_locomotion_sensor_age`.
Все команды выполнялись в существующей `.venv`, без установки зависимостей.
Launcher устанавливает OMP/OPENBLAS/MKL_NUM_THREADS=1 и LIBGL_ALWAYS_SOFTWARE=1.

```bash
chmod +x scripts/launch_interactive_h1.sh
mkdir -p results/interactive_h1_checks
scripts/launch_interactive_h1.sh --headless --output results/interactive_h1_checks/clean > results/interactive_h1_checks/clean.log 2>&1
scripts/launch_interactive_h1.sh --headless --block imu:5:5.12 --block joints:7:7.12 --output results/interactive_h1_checks/blocks > results/interactive_h1_checks/blocks.log 2>&1
scripts/launch_interactive_h1.sh --headless --method M --block imu:5:5.12 --block joints:7:7.12 --output results/interactive_h1_checks/compensation > results/interactive_h1_checks/compensation.log 2>&1
.venv/bin/python -I scripts/check_sensor_age.py > results/interactive_h1_checks/sensor_checks.log 2>&1
scripts/launch_interactive_h1.sh --auto-close --block imu:5:5.12 --block joints:7:7.12 --output results/interactive_h1_checks/window > results/interactive_h1_checks/window.log 2>&1
.venv/bin/python -I -m pip freeze > results/interactive_h1_checks/requirements.lock.txt
.venv/bin/python -I scripts/check_interactive_h1.py > results/interactive_h1_checks/checks.log 2>&1
scripts/launch_interactive_h1.sh --auto-close --block imu:5:5.12 --block joints:7:7.12 --output results/interactive_h1_checks/window_fast > results/interactive_h1_checks/window_fast.log 2>&1
scripts/launch_interactive_h1.sh --auto-close --block imu:5:5.12 --block joints:7:7.12 --output results/interactive_h1_checks/window_final > results/interactive_h1_checks/window_final.log 2>&1
scripts/launch_interactive_h1.sh --auto-close --block imu:5:5.12 --block joints:7:7.12 --output results/interactive_h1_checks/window_realtime > results/interactive_h1_checks/window_realtime.log 2>&1
.venv/bin/python -I scripts/check_interactive_h1.py > results/interactive_h1_checks/checks.log 2>&1
bash -n scripts/launch_interactive_h1.sh
```

Четыре оконных запуска выполнены с разрешением доступа к графическому сеансу.
В первом окне пользователь дополнительно нажимал I/J; реальные интервалы
сохранены в `window/key_events.json`. Это демонстрация, не парная научная серия.
Первый оконный запуск выполнен до оптимизации рендеринга: 1000×750, 50 Гц,
с тенями/отражениями. Исходники каждого запуска идентифицируются по хешам
в его summary.json. Промежуточный вариант window_fast: 640×480, 25 Гц,
без теней/отражений. Финальный вариант window_final: 640×480, 12,5 Гц,
без теней/отражений. Окончательный window_realtime использует ту же графику,
но привязывает расписание ожиданий к началу эпизода: такты без отрисовки
догоняют реальное время после дорогого кадра. Во всех окнах могли быть ручные
вмешательства, сохранённые в key_events.json.
