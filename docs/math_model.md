# Wear model

## Current MVP

For joint `j`, one dance produces the load proxy

```text
S_j = integral(abs(tau_j(t) * qdot_j(t)) dt)
```

The current implementation normalizes `S_j` across joints, accumulates scalar damage and maps health to available torque:

```text
s_j = S_j / max_k(S_k)
D_j(R) = R * alpha * s_j
h_j(R) = clip(1 - D_j(R), 0, 1)
u_j(R) = u_floor + (1-u_floor) * h_j(R)^p
```

`u_j` scales the actuator force range. This is useful for a closed-loop integration test, but it is not yet a physical fatigue law.

The implementation maps joint-ordered `u_j` values to MuJoCo actuators through
`actuator_trnid`. This mapping is part of the model contract: using array
position as an actuator ID silently degrades the wrong joints.

## Why direct extrapolation to 1M is conditional

Multiplying a small measured delta by one million is mathematically correct only under the assumed linear damage law, stationary dance/load distribution and unchanged operating conditions. Those assumptions can fail through nonlinear fatigue, temperature, lubrication, impacts, controller adaptation and load redistribution as joints degrade.

Therefore the correct claim is:

> We estimate the conditional state at one million repetitions under a specified accumulation model and parameter distribution.

It is not correct to claim a measured one-million-dance lifetime without calibration.

## Proposed scientific model

Use a state vector per joint:

```text
x_j = [fatigue_j, impact_j, thermal_age_j, wear_j, backlash_j]
```

Candidate update channels:

```text
fatigue: rainflow/counting or torque-cycle equivalent damage
impact: thresholded contact impulse / peak load accumulation
temperature: thermal state plus Arrhenius-like irreversible aging
wear: load-speed-time proxy with lubrication/temperature dependence
backlash: stochastic monotone growth linked to accumulated wear
```

Simulator effects should remain explicit:

```text
available torque/current limit
joint friction
backlash/dead zone
sensor bias/noise
fault/shutdown probability
```

Do not collapse all mechanisms into one health scalar when fitting data. A scalar dashboard score may be derived after the physical states are updated.

## Parameter validation

Parameters require one of:

- manufacturer/component endurance data;
- dedicated actuator/gearbox bench tests;
- real G1 telemetry with maintenance/fault labels;
- literature priors with uncertainty ranges.

Report parameter distributions and sensitivity, not only a best-fit value.
