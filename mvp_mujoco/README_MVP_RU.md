# MuJoCo MVP: один танец -> accelerated degradation -> первая падающая итерация

Главная инструкция находится в `../START_HERE_RU.md`.

## Структура

- `user_env_template.py` — единственный файл, который нужно адаптировать под вашу среду.
- `wearbench/damage.py` — damage/health model MVP.
- `wearbench/mujoco_degradation.py` — mapping joints/actuators и torque-limit derating.
- `wearbench/run_utils.py` — логирование одного танца и failure detector.
- `scripts/01_run_baseline.py` — здоровый танец.
- `scripts/06_run_dance_sim_baseline.py` — адаптер к `../dance_sim`: ONNX/mjlab прогон и запись `baseline_dance.npz`.
- `scripts/02_compute_damage.py` — relative damage profile.
- `scripts/03_run_aged_candidate.py` — проверить конкретную виртуальную repetition.
- `scripts/04_search_failure_iteration.py` — быстрый binary search первой failing repetition.
- `scripts/05_visualize.py` — графики.
- `scripts/07_search_dance_sim_failure.py` — binary search через bundled `../dance_sim`.
- `scripts/15_run_native_health_sweep.py` — 100 different virtual wear-rate
  realisations on the deterministic native-MuJoCo deploy reference, with
  trajectory deviation against the healthy trace.
- `config.json` — параметры первой демонстрации.

## Быстрый путь через bundled dance_sim

Если используется уже лежащий рядом `../dance_sim`, сначала пишем baseline из
реального ONNX/mjlab playback:

```bash
./run_dance_sim_baseline.sh --duration 120 --out outputs/baseline_dance.npz --device cuda:0
```

Потом считаем нагрузочный профиль:

```bash
../dance_sim/.venv/bin/python scripts/02_compute_damage.py
```

Для короткой smoke-проверки можно заменить `--duration 120` на `--duration 2`.

Проверить конкретную виртуальную репетицию через тот же `dance_sim`:

```bash
./run_dance_sim_candidate.sh --duration 120 --device cuda:0 --repetition 50 --out outputs/candidate_rep_0000050.npz
```

Найти первую падающую репетицию binary search:

```bash
../dance_sim/.venv/bin/python scripts/07_search_dance_sim_failure.py --device cuda:0
```

Старые `scripts/03_run_aged_candidate.py` и `scripts/04_search_failure_iteration.py`
оставлены для варианта с ручным `user_env.py`; для bundled `dance_sim` используйте
`run_dance_sim_candidate.sh` и `scripts/07_search_dance_sim_failure.py`.

## MVP-модель

Нагрузочный proxy одного танца:

`severity_j = integral(abs(tau_j * qdot_j) dt)`.

Нормируем относительно максимума по joints:

`severity_norm_j = severity_j / max(severity)`.

Damage после R полностью прошедших повторов:

`D_j(R) = R * alpha * severity_norm_j`.

Health:

`h_j = clip(1 - D_j, 0, 1)`.

Torque/force capability:

`scale_j = floor + (1-floor) * h_j**p`.

Скрипт подбирает `alpha` из желаемого демонстративного состояния наиболее нагруженного сустава на `target_failure_repetition`. Фактический момент падения определяется симуляцией, а не формулой.

## Ключевое ограничение

`alpha` в MVP — искусственный accelerated-aging coefficient. Он нужен только для демонстрации, что closed loop работает. Это не число ресурса реального G1.

## Ensemble из 100 виртуальных роботов

```bash
./run_native_health_sweep.sh --trials 100 --workers 6 --seed 20260826 \
  --wear-rate-cv 0.25 --out-dir outputs/rb_y_16s/native_health_ensemble_100
```

На checkpoint `0` все 100 прогонов обязаны совпадать: это проверка
детерминизма одного и того же робота, а не оценка вероятности. На остальных
checkpoint у каждого виртуального робота свой множитель скорости износа
`LogNormal(mean=1.0, CV=0.25)`, но одинаковы старт, policy, motion и physics.
В `summary.json`/`summary.csv` записываются завершения/падения и p05/p50/p95
отклонений joint state, action, pelvis, torso и ориентации. До калибровки по
реальным данным это только условная чувствительность модели, не прогноз
надёжности реального G1.

Сделать 4K коллажи `10x10` по всем 100 траекториям и отдельные 1080p ролики
характерных исходов можно без повторного simulation run:

```bash
MUJOCO_GL=egl ./run_render_native_health_ensemble.sh
```

Рендер проигрывает сохранённые `.npz` traces, поэтому кадры строго
соответствуют `trials.csv`. MP4 не входят в Git: это большие generated
artifacts в каталоге output эксперимента.
