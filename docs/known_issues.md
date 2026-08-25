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

## Torque derating is not a complete wear mechanism

The current simulator implementation maps accumulated damage mainly to available actuator force/torque. Real degradation can also increase friction, backlash, thermal resistance, noise and fault probability.

Mitigation: keep these mechanisms separate and add them only with measurable parameters and ablation tests.
