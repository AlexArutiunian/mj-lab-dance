# Known issues

## MJWarp multi-world healthy falls

Severity: critical for batched survival statistics.

Symptom: healthy robots that are stable with `num_envs=1` diverge and fall when many MJWarp worlds are evaluated together. Failures concentrate near the dynamic motion segment at short-clip 8–9 s / full-motion 28.5–29.5 s.

Evidence:

```text
single world: stable
16 worlds: 1 healthy fall in an earlier probe
32 worlds: 2 healthy falls
64 worlds: 4 healthy falls in grid mode
256 worlds: 32 healthy falls in an RB+Y probe
64 worlds, local origins: 10 healthy falls, max state spread 4.37
```

Interpretation: the dynamic segment amplifies solver/world numerical divergence. It is not evidence that the real policy has a 6–12% fall probability.

Mitigation:

- primary runs use one world per process;
- every experiment includes a healthy baseline;
- the runner aborts when healthy worlds fail or identical worlds exceed state-spread tolerance;
- large-batch videos and rates are labeled diagnostic/invalid.

Open investigation:

- compare native MuJoCo and MJWarp single-world trajectories;
- compare CUDA graph enabled/disabled;
- compare solver/integrator/contact settings;
- capture first divergence time in qpos, contacts, controls and observations;
- test upstream MJWarp versions and a minimal reproducible scene.

## Uncalibrated wear coefficient

The accelerated `alpha` is selected to create observable degradation. It has no current mapping to real dance repetitions, operating hours or component life.

Mitigation: report conditional/sensitivity results only and calibrate against real telemetry or component endurance data.

## Reset-perturbation sensitivity near 9 seconds

Severity: critical for Monte Carlo interpretation.

The exact deploy start completes healthy, but some tiny randomized reset
offsets produce healthy floor-level falls around 8.3--8.6 s. A wider stress
test also produced a healthy fall at 9.74 s. The perturbations are injected by
MJLab's motion-command reset and have not been calibrated to the real `RB+Y`
deployment-state distribution.

Mitigation:

- reject an entire paired trial when its healthy control falls;
- use exact-start runs only as deterministic regression/sensitivity tests;
- keep randomized runs labeled as robustness stress tests;
- derive future reset distributions from real robot telemetry before making
  survival-rate claims.

## Superseded joint-to-actuator indexing

Resolved on 2026-08-25. Earlier worn simulations assumed joint-list index was
the actuator index. MJLab uses a different actuator order, so torque derating
was applied to the wrong drives. The implementation now resolves each joint via
MuJoCo `actuator_trnid`, validates one actuator per joint, and logs the resolved
IDs. All worn results produced before this correction are superseded.

## Torque derating is not a complete wear mechanism

The current simulator implementation maps accumulated damage mainly to available actuator force/torque. Real degradation can also increase friction, backlash, thermal resistance, noise and fault probability.

Mitigation: keep these mechanisms separate and add them only with measurable parameters and ablation tests.
