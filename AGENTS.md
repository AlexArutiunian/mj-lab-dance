# AGENTS.md

Этот файл задаёт правила работы coding/research-агентов в репозитории WearBench-G1.

Цель проекта — построить воспроизводимый контур оценки накопленного износа Unitree G1 при многократном выполнении танца: измерить нагрузки в MuJoCo/MJLab, обновить параметризованное состояние здоровья приводов и оценить функциональную работоспособность робота. Нельзя подгонять модель ради красивой survival curve или выдавать accelerated-aging коэффициент за реальный ресурс.

## 1. Текущий этап

Только simulation/research mode. Текущая реализация является MVP модели деградации, а не сертифицированным предсказателем срока службы.

Основной танец для deploy-сценария:

```text
controller trigger: RB + Y
motion: dance1_subject2_16s_faststart.npz
policy: dance1_subject2_16s_faststart/exported/policy.onnx
duration: 16.24 s
source interval in full motion: 20.50..36.72 s
```

Короткий клип `t=8.0..9.0 s` соответствует полному motion `t=28.5..29.5 s`. Это один и тот же сложный элемент, а не два независимых события. Recut численно эквивалентен исходному интервалу, но не является бит-в-бит копией: максимум расхождения joint position около `1e-4 rad`, joint velocity около `0.003 rad/s`.

## 2. Что уже измерено

Здоровая политика в одном MJWarp world проходит танец:

```text
RB+Y, 1 world, 16.24 s: 0/1 falls
full motion, 1 world, 120 s: 0/1 falls
```

Старые большие batched runs дали ложные baseline-падения:

```text
1 world:   0/1
4 worlds:  0/4
8 worlds:  0/8 in one probe
16 worlds: 1/16
32 worlds: 2/32
64 worlds: 4/64
256 worlds: 32/256 in the RB+Y probe
```

Дополнительная проверка с общей локальной системой координат не исправила проблему: `10/64` здоровых worlds упали, а идентичные состояния разошлись до `max |dq| = 4.37`. Следовательно, старые survival/failure rates из больших `nworld` нельзя использовать как результат износа.

## 3. Главный исследовательский вопрос

> Можно ли по измеряемым torque, velocity, contact и temperature channels построить модель накопления повреждений, которая физически согласована, калибруема по реальной телеметрии и корректно предсказывает относительный риск и функциональный RUL робота?

До калибровки допустимы только:

- сравнительное ранжирование суставов;
- sensitivity analysis;
- accelerated degradation stress tests;
- условные прогнозы при явно записанных assumptions.

Абсолютное утверждение «G1 выдержит N танцев» запрещено без экспериментальной калибровки.

## 4. Приоритет источников

Перед изменениями читай:

```text
AGENTS.md
README.md
docs/status.md
docs/experiment_protocol.md
docs/math_model.md
docs/known_issues.md
код и тесты
официальную документацию MuJoCo/MJLab/MJWarp
```

Каждое assumption помечай как assumption. Каждый measured result должен иметь команду запуска, seed, motion/policy hash, simulator versions, device, `num_envs` и критерий отказа.

## 5. Experiment loop

```text
single-world healthy baseline
 -> motion/policy identity check
 -> telemetry capture
 -> per-joint load features
 -> damage update with units and parameters
 -> actuator/sensor degradation mapping
 -> single-world candidate rollout
 -> paired healthy/worn comparison
 -> repeated seeds and uncertainty
 -> convergence and sensitivity checks
 -> only then batched acceleration
```

Приоритет: correctness -> baseline validity -> physical meaning -> reproducibility -> observability -> performance.

## 6. Simulator quality gates

Научный прогон считается валидным только если:

- healthy control включён в тот же protocol;
- healthy control имеет `0` падений;
- policy и motion соответствуют deploy `RB+Y`;
- failure определяется уровнем pelvis/torso у пола с hold time, а не кратким наклоном;
- нет NaN/Inf, solver overflow и contact-buffer overflow;
- actuator degradation применена ко всем нужным worlds/joints;
- joint-ordered degradation разрешена в actuator IDs через MuJoCo
  `actuator_trnid`; совпадение позиций в массивах не считается mapping;
- при идентичных входах нет необъяснённого межмирового расхождения;
- результаты содержат provenance и конфигурацию.

Если healthy batch падает или идентичные worlds расходятся сильнее tolerance, весь batch помечается `INVALID_SIMULATOR_BASELINE`. Нельзя вычитать baseline fall rate постфактум и называть остаток wear effect.

До устранения MJWarp `nworld` артефакта primary evidence получает только `num_envs=1`. Параллелизм переносится на независимые процессы/GPU jobs, а не на worlds внутри одного solver batch.

Randomized reset/Monte Carlo считается wear evidence только после калибровки
распределения стартового состояния по telemetry реального deploy. Если
perturbation вызывает healthy fall, paired trial целиком исключается из wear
statistics и остаётся отдельным robustness stress-test.

Primary wear protocol называется `deploy_exact` и использует нулевые pose/yaw/
joint offsets. `stress_jitter` нельзя смешивать с lifetime statistics. Для
uncertainty analysis при `deploy_exact` варьируй откалиброванные параметры
damage law и actuator degradation, а не произвольное начальное состояние.

## 7. Failure definition

Primary fall criterion:

```text
pelvis_z < 0.35 m
AND torso_z < 0.30 m
continuously for >= 0.50 s
```

Tracking loss, joint-limit violation, thermal shutdown и actuator fault должны логироваться отдельными event types. Их нельзя смешивать с физическим падением.

## 8. Математическая модель

Текущий MVP proxy:

```text
severity_j = integral(abs(tau_j * qdot_j) dt)
D_j(R) = R * alpha * severity_norm_j
h_j = clip(1 - D_j, 0, 1)
torque_scale_j = floor + (1-floor) * h_j**p
```

`alpha` сейчас не откалиброван. Для scientific version развивать multi-channel state:

- mechanical fatigue from torque/load cycles;
- impact/contact damage;
- temperature state and thermal aging;
- gearbox/bearing wear proxies;
- backlash/friction growth;
- actuator torque/current derating;
- uncertainty distribution over model parameters.

Fast-forward до 1M допустим только внутри явно выбранного закона накопления повреждений. Линейную экстраполяцию малой дельты нельзя выдавать за проверенный физический закон.

## 9. Валидация модели

Минимум:

- dimensional and monotonicity checks;
- zero-load and low-load limits;
- timestep and sample-rate convergence;
- ranking stability under coefficient changes;
- comparison of several motions/load regimes;
- paired healthy/worn simulator runs;
- Monte Carlo parameter uncertainty;
- comparison with real G1 current, torque, temperature and fault logs;
- calibration/validation split if real degradation data become available.

## 10. Tests

Минимум перед значимым результатом:

- unit tests damage/health/torque mapping;
- motion interval identity test;
- actuator-to-joint mapping test;
- healthy single-world smoke test;
- baseline contamination gate test;
- deterministic output schema/provenance test;
- failure detector hold-time test;
- no-wear invariance test.

Если tests не запускались локально, не утверждай, что они green.

## 11. Results and claims

Разделяй статусы:

```text
VALIDATED_SINGLE_WORLD
EXPLORATORY_BATCH
INVALID_SIMULATOR_BASELINE
MODEL_ASSUMPTION_ONLY
REAL_ROBOT_VALIDATED
```

Видео из загрязнённого baseline batch является диагностикой simulator artifact, а не демонстрацией падения здоровой политики.

## 12. Git/docs

После значимого изменения синхронизируй `README.md`, `docs/status.md`, `WORKLOG.md` и команды воспроизведения. Не коммить `.venv`, логи, большие videos/NPZ и generated outputs. Изменения upstream `unitree_rl_mjlab` сохраняй в `patches/` и документируй base commit.
