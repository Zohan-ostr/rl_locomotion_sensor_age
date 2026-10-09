# Проверка готовой политики Unitree H1 на CPU

Дата: 8 октября 2026. Выполнена только первая практическая стадия.
Штатная политика ходит в MuJoCo со свежими наблюдениями. Подготовлены отдельное
`.venv`, CPU runner, численные логи и видео. Обучение, очереди сенсоров,
компенсация, пилотные и закрытые исследовательские серии не выполнялись.

## Объект и условия

- Официальный upstream: `unitreerobotics/unitree_rl_gym`.
- Commit: `276801e46c5d433564f24658bac64f254b7d2d4b`, локальный checkout без изменений.
- Веса: `deploy/pre_train/h1/motion.pt`, реальные байты обычного git checkout,
  не Git LFS-указатель. SHA-256:
  `44a0fbceb81f3877833ae9a398d039bea1759cb0d3c8188181013885f70589eb`.
- Сцена: `resources/robots/h1/scene.xml`, SHA-256:
  `88ffcccad2c5df71a7472b4b30ec7e4ea06e045efc6b4e11ccd8bebc2848f0b3`.
- Включённый `h1.xml`, SHA-256:
  `2bee8abddbf4bc093bcdb2266c7139cd1c78e279bf65f7ee9fde41fe599900d4`.
- Хеши всех файлов H1, включая меши: `headless20/summary.json`.
  Хеши исходного deploy-скрипта и YAML совпали с `docs/source_manifest.json`.
- LICENSE репозитория: BSD 3-Clause, копия — `UNITREE_LICENSE`.
  Отдельного файла лицензии checkpoint в проверенном пути не найдено;
  здесь фиксируется его происхождение из официального репозитория.
- Linux 6.8.0-124-generic, Intel Core i9-12900H, 20 доступных логических CPU,
  14 физических ядер по `lscpu`, 15 ГиБ RAM, около 11 ГиБ available при диагностике,
  swap отсутствует. Полная диагностика — `lscpu.txt`, `memory.txt`, `environment.json`.
- Python 3.10.12; PyTorch 2.5.1+cpu (`torch.version.cuda=None`), MuJoCo 3.2.7,
  NumPy 1.26.4, PyYAML 6.0.2. Полный runtime lock — `requirements.lock.txt`,
  bootstrap-пакеты — также `packages.json`.
- Torch intra-op: 1 поток, inter-op: 1; OMP/OpenBLAS/MKL: 1 для 20-секундных
  запусков. Программный рендерер и ffmpeg могут использовать дополнительные
  CPU-потоки. CPU affinity не ограничивалась: доступны 0–19.

Runner адаптирован из штатного `deploy_mujoco.py` с сохранением лицензии.
Используется `torch.jit.load(..., map_location="cpu")`, `eval()` и
`torch.inference_mode()`. Подтверждены вход `[1,41]`, выход `[1,10]`, CPU всех
параметров, конечность выходов на каждом вызове. Checkpoint содержит LSTM
с hidden/cell state `[1,1,64]`; исходная память нулевая. После пробного forward
для проверки размеров все buffers восстанавливаются из копии состояния
загруженного checkpoint. Во время эпизода память сохраняется между решениями,
как в исходном примере. Каждый эпизод запускается новым процессом с новой
загрузкой политики. Размеры заданы исходным YAML
и проверены реальным вызовом сети; структура параметров — `policy_inspection.log`.

Сохранены исходные масштабы: угловая скорость 0,25, положения 1,0,
скорости суставов 0,05, действие 0,25, команда `[2,2,0.25]`.
Порядок входа: angular velocity, gravity, command, relative joint positions,
joint velocities, previous action, sin/cos фазы. Период фазы — 0,8 с.
Порядок суставов: left hip yaw/roll/pitch, knee, ankle;
затем right hip yaw/roll/pitch, knee, ankle. Проверено совпадение адресов
`qpos/qvel` и порядка приводов с этим порядком модели.

PD: kp `[150,150,150,200,40,150,150,150,200,40]`,
kd `[2,2,2,4,2,2,2,2,4,2]`; target velocity — ноль.
Default targets `[0,0,-0.1,0.3,-0.2,0,0,-0.1,0.3,-0.2]` плюс action×0,25.
Физический шаг — 0,002 с, decimation — 10 (политика 50 Гц).
Первый policy-вызов после 10 физических шагов, как в upstream.

Начальное состояние — неизменённое `qpos0` XML: pelvis `[0,0,1.05]`,
кватернион `[1,0,0,0]`, суставы нулевые, скорости нулевые. Суставная
XML-поза и default targets различаются; это штатный переход исходного примера,
он не заменялся предварительным приседанием. Полное `initial_qpos` сохранено.
Команда `[0.5,0,0]` с начала, поверхность штатная ровная.
PD использует свежую локальную обратную связь. Истинное состояние используется
в метриках. Задержек, шума и случайных возмущений нет; seed не применяется.

## Реальные измерения

Три принятых запуска: короткий smoke 5 с, основной headless 20 с и 20 с с видео.
Это проверки одного детерминированного начального состояния, а не три
независимых испытания для оценки вероятности падения.

| Показатель | Smoke 5 с | Основной headless 20 с |
|---|---:|---:|
| Физические шаги / вызовы политики | 2500 / 250 | 10000 / 1000 |
| Продвижение по мировой X, м | 1,875 | 8,344 |
| Смещение по мировой Y, м | 0,077 | 0,148 |
| Длина горизонтальной траектории, м | 2,023 | 8,730 |
| Средняя продольная скорость после 2 с, м/с | 0,430 | 0,431 |
| RMSE горизонтальной скорости после 2 с, м/с | 0,150 | 0,147 |
| Минимальная высота pelvis, м | 0,990 | 0,990 |
| Средняя высота pelvis после 2 с, м | 1,033 | 1,033 |
| Максимальный модуль крена / тангажа, ° | 2,123 / 2,147 | 2,123 / 2,151 |
| Начала контактов левой / правой стопы | 8 / 7 | 26 / 26 |
| Контакт pelvis с полом | 0 | 0 |
| Падение по диагностическому критерию | не зарегистрировано | не зарегистрировано |

Скорость для tracking вращается из мировых осей в горизонтальные оси,
ориентированные по yaw корпуса. RMSE —
`sqrt(mean((vx_yaw-0.5)^2 + (vy_yaw-0)^2))`, участок t≥2 с.
Средняя скорость ниже команды примерно на 0,069 м/с; RMSE включает
колебания походки и поперечную скорость. Робот продвигается и циклически
меняет опору; успешный запуск не сводится к стоянию.

Диагностический критерий падения: pelvis ниже 0,63 м или модуль крена/тангажа
выше 60° непрерывно 0,2 с. Время падения — null, а не предсказанное будущее
значение. Номинальная высота для будущего критерия ещё должна быть зафиксирована
на пилоте: текущий диагностический порог использует начальные 1,05 м.
Контакты считаются по активным соприкосновениям соответствующего тела с
world/floor при `dist<=0`. Контактные данные `mj_step` относятся к его
вычислению контактов, тогда как `qpos` уже после интегрирования; возможен
сдвиг этих данных в пределах физического шага 2 мс.

В основном headless-запуске:

- Время цикла симуляции по настенным часам 1,074 с; process CPU 1,074 с.
- Отношение симуляционного времени к времени цикла: 18,62.
- Inference: медиана 0,110 мс, p95 0,134 мс.
- Один физический шаг: медиана 0,021 мс, p95 0,032 мс.
- Полный интервал 10 шагов + inference + метрики: медиана 1,053 мс,
  p95 1,196 мс при периоде политики 20 мс.

Загрузка сети/модели, проверочный forward и запись итоговых CSV/JSON
не включены в эти времена. Сбор метрик включён. Это одно измерение на данном
компьютере; принятый headless завершён до старта video20. Это не гарантия real-time
при любой нагрузке. В headless-цикле нет `sleep` или ограничения настенными
часами; завершение задают число физических шагов и симуляционное время.

Видео: `video20/walk.mp4`, 20 с, H.264, 640×480, 25 fps, программный EGL/Mesa
(`MUJOCO_GL=egl`, `LIBGL_ALWAYS_SOFTWARE=1`). Рендер и кодирование заняли
71,028 с настенного и 278,642 с process CPU времени; этот запуск не используется
для оценки скорости headless-серии. Просмотрены реальные кадры около 2, 8
и 18 с (`video20/frames.png`). Физические поля всех 1000 строк video20 и
headless20 совпадают точно; кодирование не меняет физическое время.

## Проверки и проблемы

- `pip check`: `No broken requirements found` в изолированном окружении.
- Gravity projection совпадает с `R.T @ [0,0,-1]` MuJoCo; отдельно проверены
  знаки при +90° pitch и roll. Максимальная ошибка в проверенных позах — 0.
- Конечность состояний/выходов, реальные размеры, адреса суставов и приводов,
  10000 шагов, 1000 строк, шаг журнала 20 мс и конечное время 20 с проверены.
- Предупреждения MuJoCo: все счётчики нулевые.
- Перемещение, повторные контакты обеих стоп, отсутствие контакта pelvis
  и диагностического падения проверены; результаты — `checks.json`.
- Первые три запуска исключены: проверочный forward изменял память LSTM
  до начала физики. Они сохранены в `probe_contaminated/` вместе с логами,
  trajectory, manifests и хешем прежнего runner. Ошибка исправлена восстановлением
  всех buffers после probe; затем все три запуска повторены. Итоговая таблица
  использует только исправленные запуски. Нулевая начальная память и её
  восстановление проверяются явно; успешная ходьба ранних запусков не служит
  доказательством штатного baseline.
- Системный Python не содержит ensurepip. Решено bootstrap pip внутри `.venv`.
- Сеть песочницы недоступна. Загрузки выполнены с разрешением; неудачные
  установки сохранены, см. `commands.md` и `install_*.log`.
- Внешний ROS `PYTHONPATH` исключён флагом `-I`; глобальные пакеты не менялись.
- Видео работает без NVIDIA и окна, но программная визуализация существенно
  медленнее численного цикла. MP4 локально сохранён, по `.gitignore` не включается
  в git автоматически; vendor и .venv также локальные.

Блокирующих проблем для следующей реализации сенсорных очередей не выявлено.
Остаются ненулевое отклонение скорости от команды и ограниченность проверки:
одна модель, одна политика, один начальный режим, свежие данные, ровный пол.
Устойчивость к асинхронному устареванию и полезность компенсатора не проверены;
гипотеза не подтверждена. Перенос на реального робота не исследован.
Критерий падения, параметры компенсатора и тестовая серия ещё не фиксировались.

## Воспроизведение

Из корня проекта, в уже подготовленном окружении, с новым каталогом вывода:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -I scripts/run_h1_baseline.py \
  --seconds 20 --threads 1 --output results/baseline_h1/reproduction
```

Для видео добавить `MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1`, `--video`
и отдельный `--output`. Для проверки сохранённых исходных запусков:

```bash
.venv/bin/python -I scripts/check_h1_baseline.py
```

На чистой машине создание окружения при отсутствии ensurepip можно выполнить
явно через `python3 -m venv --without-pip .venv`, загрузить get-pip.py как
в `commands.md` и выполнить `.venv/bin/python -I /tmp/h1-get-pip.py`.
Точные runtime-зависимости восстановить командой:

```bash
.venv/bin/python -I -m pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r results/baseline_h1/requirements.lock.txt
```

Официальный репозиторий получить и переключить на зафиксированный commit
командами из `commands.md`; после загрузки сверить `sha256.txt`.
Изменять глобальное окружение, устанавливать Isaac Gym или обучать сеть не нужно.
После этой проверки работа остановлена по запросу пользователя.


---

<a id="commands"></a>

# Фактические команды сеанса 8 октября 2026

Рабочий каталог: `/home/zohan/projects/rl_locomotion_sensor_age`.
Ниже команды подготовки и запусков, включая неудачные попытки. Вывод первых
диагностик и bootstrap был в журнале сеанса; для установок runtime и всех
симуляций вывод перенаправлен в файлы этой папки.

```bash
pwd
cat docs/README.md#research docs/README.md#experiment_protocol docs/README.md#practical_steps configs/protocol.yaml docs/source_manifest.json README.md
cat docs/README.md#experiment_protocol docs/README.md#practical_steps THIRD_PARTY.md .gitignore
sed -n '100,290p' docs/README.md#research
sed -n '1,105p' docs/README.md#research
git status --short
lscpu
free -h
python3 --version
printenv VIRTUAL_ENV CONDA_PREFIX
ls -a
mkdir -p scripts results/baseline_h1 vendor
git clone https://github.com/unitreerobotics/unitree_rl_gym.git vendor/unitree_rl_gym
```

Первая попытка clone в песочнице: exit 128, `Couldn't connect to server`.
Та же команда с разрешённым сетевым доступом: exit 0.

```bash
python3 -m venv .venv
```

Exit 1: `ensurepip is not available`. Частично созданное окружение содержит
свой Python и `include-system-site-packages = false`; оно использовано далее.
Глобальный `python3.10-venv` не устанавливался.

```bash
curl -fL https://bootstrap.pypa.io/get-pip.py -o /tmp/h1-get-pip.py
.venv/bin/python /tmp/h1-get-pip.py
.venv/bin/python /tmp/h1-get-pip.py --no-cache-dir
```

Первый curl: exit 7, сеть песочницы; повтор с сетевым разрешением: exit 0.
Первый bootstrap: exit 1, сеть песочницы; bootstrap с `--no-cache-dir` и
разрешением: exit 0, установлены pip 26.2.1, setuptools 84.0.0,
wheel 0.48.0, packaging 26.3 в `.venv`. Его вывод также содержал предупреждения
о зависимостях ROS из внешнего `PYTHONPATH`. Последующие команды используют
`-I`, исключающий этот путь. SHA-256 bootstrap сохранён в `bootstrap_sha256.txt`.

```bash
git -C vendor/unitree_rl_gym checkout 276801e46c5d433564f24658bac64f254b7d2d4b
cat vendor/unitree_rl_gym/LICENSE vendor/unitree_rl_gym/.gitattributes vendor/unitree_rl_gym/deploy/deploy_mujoco/configs/h1.yaml vendor/unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco.py
file vendor/unitree_rl_gym/deploy/pre_train/h1/motion.pt
cat vendor/unitree_rl_gym/resources/robots/h1/scene.xml
sed -n '1,75p' vendor/unitree_rl_gym/resources/robots/h1/h1.xml
cat vendor/unitree_rl_gym/legged_gym/__init__.py
df -h .
ldconfig -p | rg 'OSMesa|libEGL|libGL.so'
cat .venv/pyvenv.cfg
printenv PYTHONPATH
.venv/bin/python -I -m pip list
```

Checkout успешен; `.gitattributes` в этом commit отсутствует, поэтому
совместный `cat` вернул exit 1, остальные запрошенные файлы прочитаны.
Checkpoint — настоящий ZIP/TorchScript из обычного git checkout.

```bash
.venv/bin/python -I -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu > results/baseline_h1/install_torch.log 2>&1
.venv/bin/python -I -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu > results/baseline_h1/install_torch_network.log 2>&1
.venv/bin/python -I -m pip install --no-cache-dir mujoco==3.2.7 numpy==1.26.4 PyYAML==6.0.2 imageio==2.37.0 imageio-ffmpeg==0.6.0 > results/baseline_h1/install_runtime.log 2>&1
.venv/bin/python -I -m pip install --no-cache-dir mujoco==3.2.7 numpy==1.26.4 PyYAML==6.0.2 imageio==2.37.0 imageio-ffmpeg==0.6.0 > results/baseline_h1/install_runtime_network.log 2>&1
```

Команды с логами `install_torch.log` и `install_runtime.log` неуспешны из-за
сети песочницы. Повторы с сетевым разрешением и суффиксом `_network` успешны.
CUDA wheels и тренировочный стек не устанавливались.

```bash
git -C vendor/unitree_rl_gym rev-parse HEAD > results/baseline_h1/upstream_commit.txt
lscpu > results/baseline_h1/lscpu.txt
free -h > results/baseline_h1/memory.txt
cp vendor/unitree_rl_gym/LICENSE results/baseline_h1/UNITREE_LICENSE
sha256sum vendor/unitree_rl_gym/deploy/pre_train/h1/motion.pt vendor/unitree_rl_gym/resources/robots/h1/scene.xml vendor/unitree_rl_gym/resources/robots/h1/h1.xml vendor/unitree_rl_gym/deploy/deploy_mujoco/configs/h1.yaml vendor/unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco.py > results/baseline_h1/sha256.txt
.venv/bin/python -I -m py_compile scripts/run_h1_baseline.py
.venv/bin/python -I -m pip freeze > results/baseline_h1/requirements.lock.txt
.venv/bin/python -I -m pip check > results/baseline_h1/pip_check.log 2>&1
.venv/bin/python -I -m pip list --format=json > results/baseline_h1/packages.json
.venv/bin/python -I scripts/run_h1_baseline.py --seconds 5 --threads 1 --output results/baseline_h1/smoke > results/baseline_h1/smoke.log 2>&1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I scripts/run_h1_baseline.py --seconds 20 --threads 1 --output results/baseline_h1/headless20 > results/baseline_h1/headless20.log 2>&1
MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I scripts/run_h1_baseline.py --seconds 20 --threads 1 --video --output results/baseline_h1/video20 > results/baseline_h1/video20.log 2>&1
.venv/bin/python -I scripts/check_h1_baseline.py > results/baseline_h1/checks.json 2> results/baseline_h1/checks.stderr.log
```

Все симуляции: exit 0. Проверка выполнялась сначала до завершения видео,
затем повторно после его завершения для сравнения физических траекторий.
Headless20 и video20 запускались с небольшим перекрытием по настенному
времени; оценка быстродействия является разовым измерением при этой нагрузке.

Эти первые три запуска впоследствии **исключены** из итогового baseline:
shape probe изменял память LSTM. После исправления runner файлы перенесены
и те же команды повторены; финальный headless завершился до запуска видео.
Команды сохранения и повтора:

```bash
mkdir -p results/baseline_h1/probe_contaminated
mv results/baseline_h1/smoke results/baseline_h1/headless20 results/baseline_h1/video20 results/baseline_h1/smoke.log results/baseline_h1/headless20.log results/baseline_h1/video20.log results/baseline_h1/checks.json results/baseline_h1/video_metadata.log results/baseline_h1/runner_sha256.txt results/baseline_h1/probe_contaminated/
.venv/bin/python -I scripts/run_h1_baseline.py --seconds 5 --threads 1 --output results/baseline_h1/smoke > results/baseline_h1/smoke.log 2>&1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I scripts/run_h1_baseline.py --seconds 20 --threads 1 --output results/baseline_h1/headless20 > results/baseline_h1/headless20.log 2>&1
MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I scripts/run_h1_baseline.py --seconds 20 --threads 1 --video --output results/baseline_h1/video20 > results/baseline_h1/video20.log 2>&1
.venv/bin/python -I scripts/check_h1_baseline.py > results/baseline_h1/checks.json 2> results/baseline_h1/checks.stderr.log
sha256sum scripts/run_h1_baseline.py scripts/check_h1_baseline.py > results/baseline_h1/runner_sha256.txt
```

После завершения исправленного video20 повторены checks и извлечение кадров
той же командой ниже. Это фиксирует также физическую идентичность итоговых
headless/video траекторий.

Дополнительные команды для сохранения метаданных и просмотра реальных кадров:

```bash
.venv/bin/python -I -c 'import json,os,platform,sys; from pathlib import Path; Path("results/baseline_h1/environment.json").write_text(json.dumps({"python_executable":sys.executable,"python_version":sys.version,"virtualenv_prefix":sys.prefix,"base_prefix":sys.base_prefix,"cpu_affinity":sorted(os.sched_getaffinity(0)),"logical_cpus":os.cpu_count(),"platform":platform.platform(),"isolated_mode":sys.flags.isolated},indent=2)+"\n")'
sha256sum /tmp/h1-get-pip.py > results/baseline_h1/bootstrap_sha256.txt
.venv/bin/python -I -c 'import imageio.v2 as io,numpy as np; r=io.get_reader("results/baseline_h1/video20/walk.mp4"); frames=[r.get_data(i) for i in (50,200,450)]; io.imwrite("results/baseline_h1/video20/frames.png",np.concatenate(frames,axis=1)); print(r.get_meta_data()); r.close()' > results/baseline_h1/video_metadata.log 2>&1
.venv/bin/python -I -c 'import torch,zipfile; p="vendor/unitree_rl_gym/deploy/pre_train/h1/motion.pt"; m=torch.jit.load(p,map_location="cpu").eval(); print(m); print("forward schema:",m.forward.schema); print("parameter shapes:",[(k,list(v.shape),str(v.device)) for k,v in m.named_parameters()]); print("archive members:",zipfile.ZipFile(p).namelist())' > results/baseline_h1/policy_inspection.log 2>&1
sha256sum scripts/run_h1_baseline.py scripts/check_h1_baseline.py > results/baseline_h1/runner_sha256.txt
```
