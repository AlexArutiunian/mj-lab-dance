# WearBench-G1 — с чего начинаем прямо сейчас

## Цель первого MVP

Не пытаться сразу предсказывать реальный срок службы Unitree G1. Первый шаг — проверить **замкнутый контур деградации** на одном длинном (~2 мин) разнообразном танце в MuJoCo:

1. один раз прогнать танец на «новом» роботе;
2. записать по каждому суставу `qdot(t)` и actuator/joint torque `tau(t)`;
3. из одного танца посчитать относительную тяжесть нагрузки;
4. ввести **явно искусственный коэффициент ускоренного старения**;
5. связать накопленную деградацию с уменьшением доступного усилия приводов;
6. не симулировать 1000 предыдущих танцев: аналитически выставлять здоровье робота «как после N-1 повторений» и запускать только N-й танец;
7. бинарным поиском найти первый N, при котором робот падает / теряет tracking;
8. визуализировать: какие суставы деградировали, как падал health/torque limit и что произошло на failing iteration.

Это называется **accelerated degradation stress-test / integration sanity test**. Пока коэффициент не откалиброван по железу, число `N_fail` НЕ является прогнозом реального ресурса.

## Почему именно так

Полный танец длится ~120 с. Если буквально проигрывать его 10 000 раз, это дорого и бессмысленно для первого теста. В MVP предполагаем, что относительный профиль повреждения одного повторения близок к baseline-профилю. Тогда состояние после N повторений можно вычислить мгновенно и симулировать только сам кандидатный N-й танец. Это позволяет искать «первую падающую итерацию» за O(log N) полных прогонов.

## Порядок запуска

Перейдите в `mvp_mujoco/`.

### 0. Подключить вашу среду

Скопировать:

```bash
cp user_env_template.py user_env.py
```

И заполнить в `user_env.py` только 4 обязательных hook-а:

- `make_model_and_data()` — загрузка вашего G1 MJCF/XML;
- `reset_for_dance(model, data)` — начальная стойка/состояние;
- `compute_control(t, model, data)` — ваш уже существующий контроллер/trajectory tracker танца;
- `tracking_error(t, model, data)` — одна скалярная ошибка tracking (на MVP можно вернуть 0.0, если пока нет).

Также указать длительность танца и, при необходимости, правила падения.

### 1. Baseline: один здоровый танец

```bash
python scripts/01_run_baseline.py
```

Получаем `outputs/baseline_dance.npz`.

### 2. Посчитать относительный damage одного танца

```bash
python scripts/02_compute_damage.py
```

Получаем:

- `outputs/damage_profile.json`
- `outputs/damage_profile.csv`

Базовая метрика MVP:

\[
S_j = \int_0^{T_{dance}} |\tau_j(t)\dot q_j(t)|dt,
\]

после чего она нормируется по суставам. Это **не физический wear law**, а быстрый нагрузочный proxy для проверки пайплайна.

### 3. Задать показательное «старение к N-му танцу»

В `config.json` задайте, например:

```json
"target_failure_repetition": 50,
"target_weakest_joint_torque_scale": 0.30
```

Скрипт автоматически подберет коэффициент ускоренного старения так, чтобы к 50-му танцу наиболее нагруженный сустав имел около 30% исходного доступного actuator force. Это только начальная точка — фактическое падение определит MuJoCo.

### 4. Быстро найти первую failing iteration

```bash
python scripts/04_search_failure_iteration.py --max-repetitions 1000000
```

Скрипт НЕ прогоняет миллион танцев. Для каждого кандидата N он:

1. вычисляет damage/health «как после N-1 повторений»;
2. уменьшает `actuator_forcerange` соответствующих приводов;
3. запускает ровно один 2-минутный танец;
4. проверяет fall / tracking failure;
5. бинарным поиском локализует границу.

Результаты:

- `outputs/failure_search.csv`
- `outputs/failure_summary.json`
- `outputs/last_success.npz`
- `outputs/first_failure.npz`

### 5. Визуализация

```bash
python scripts/05_visualize.py
```

Получаем готовые figure-файлы:

- `outputs/fig_damage_per_joint.png`
- `outputs/fig_health_vs_repetition.png`
- `outputs/fig_base_height_comparison.png`
- `outputs/fig_tracking_comparison.png`

Для первой демонстрации нам достаточно показать:

**Healthy dance → aged state near N_fail → first failing dance.**

## Что именно является результатом первого дня

Минимально успешный результат:

- baseline танец проходит;
- top-5 наиболее нагруженных joints определяются из реальных логов симуляции;
- wear state сохраняется математически между повторениями;
- actuator capability уменьшается с damage;
- существует граница `N_success < N_failure`, которую мы находим автоматически;
- на графике видно, какой joint первым становится лимитирующим;
- failing candidate реально падает/теряет tracking в MuJoCo.

Если это работает, архитектура статьи доказана программно. После этого заменяем искусственный коэффициент на физически содержательную multi-channel модель и параллелим среды.

## Важная научная формулировка

Для MVP в статье/репозитории писать:

> We use an intentionally accelerated, uncalibrated degradation coefficient to validate the closed-loop lifetime simulation pipeline. The resulting repetition counts are not interpreted as real Unitree G1 lifetime predictions.

Нельзя писать, что реальный G1 сломается после найденного числа танцев, пока нет калибровки.
