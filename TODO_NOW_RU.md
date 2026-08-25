# TODO — текущий ближайший этап: один 2-минутный танец в MuJoCo

## P0 — сегодня: получить первую показательную деградацию

- [ ] Поднять ваш G1 в MuJoCo и убедиться, что 120-секундный танец проходит без деградации.
- [ ] В `mvp_mujoco/user_env.py` подключить существующий dance controller / reference tracker.
- [ ] Проверить mapping actuator -> hinge joint для всех управляемых DoF.
- [ ] Записать baseline: `q`, `qdot`, `qfrc_actuator`/joint torque, base height, roll/pitch, tracking error.
- [ ] Посчитать для каждого сустава `S_j = integral |tau_j*qdot_j| dt`.
- [ ] Нормировать `S_j` и получить ранжирование joints по тяжести одного танца.
- [ ] Включить accelerated damage: `Delta D_j = alpha * S_j_norm` на одно повторение.
- [ ] Связать health с силой привода: `force_limit_j = force_limit_j0 * scale(h_j)`.
- [ ] Выбрать демонстративный target, например 30–100 повторений до сильного ослабления самого нагруженного joint.
- [ ] Не прогонять все повторения: выставлять состояние после N-1 повторений аналитически.
- [ ] Запускать N-й танец и фиксировать success/failure.
- [ ] Бинарным поиском найти `N_last_success` и `N_first_failure`.
- [ ] Сохранить оба лога и сделать 4 графика.

## Definition of Done для MVP

- [ ] `baseline_dance.npz` — здоровый полный танец.
- [ ] `damage_profile.csv/json` — damage одного танца по joint.
- [ ] `failure_summary.json` — первая failing iteration при выбранном ускоренном коэффициенте.
- [ ] `last_success.npz`, `first_failure.npz` — две соседние итерации.
- [ ] `fig_damage_per_joint.png` — какие joints нагружает танец.
- [ ] `fig_health_vs_repetition.png` — health/torque capacity по мере виртуального старения.
- [ ] `fig_base_height_comparison.png` — healthy/last-success/first-failure.
- [ ] `fig_tracking_comparison.png` — tracking degradation перед падением.

## P1 — следующий этап после MVP

- [ ] Damage обновлять внутри каждого танца, а не только между повторами.
- [ ] Добавить impact channel по контактам/landing impulses.
- [ ] Добавить thermal state.
- [ ] Разделить reversible temperature и irreversible thermal aging.
- [ ] Добавить friction/backlash degradation, не только torque derating.
- [ ] Повторить на 3–5 разных движениях.
- [ ] Проверить, сохраняется ли ranking суставов при изменении коэффициентов.

## P2 — scientific version

- [ ] Перейти от искусственного `alpha` к параметризированной fatigue/wear модели.
- [ ] Калибровать torque/power/temperature по реальному G1.
- [ ] Добавить неопределенности параметров и Monte Carlo.
- [ ] Массово параллелить virtual lifetimes.
- [ ] Строить survival curves и first-joint-to-failure.
- [ ] Отдельно оценивать physical RUL и functional RUL.
- [ ] Добавить degradation-aware RL.

## P3 — final RA-L experiments

См. `article_RA-L/EXPERIMENTS_TODO.md`.
