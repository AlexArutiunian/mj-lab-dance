# Cross-Policy Wear Experiments

Updated: 2026-08-27.

## Scope

This note consolidates the current native-MuJoCo evidence for the Unitree G1
under three deployed policies:

1. short `RB+Y` dance clip, 16.24 s;
2. full `dance1_subject2`, 131.48 s;
3. Unitree velocity-walking policy at `vx=0.4 m/s`, 60 s.

The completed cross-policy curves are a **conditional simulator sensitivity analysis**, not a measured
physical lifetime or fleet failure probability for a real G1. The repetition
coordinate is a normalized accelerated-wear coordinate until calibrated using
real-robot or component-life data.

## Common Protocol

- Physics/control backend: native C MuJoCo, Unitree `scene_g1.xml`, 2 ms
  physics, 20 ms ONNX policy control.
- Every trial starts from the same exact initial pose and uses the deployed
  policy and matching `deploy.yaml` observation/action interface.
- Floor-level failure: pelvis below `0.45 m` **and** torso below `0.55 m` for
  at least `0.5 s`.
- Completed cross-policy sweep: all 29 actuators receive the same relative
  short-dance joint-load damage profile;
  health maps to available torque with floor `0.10` and exponent `1.5`.
- Population sensitivity: 100 virtual robots per wear checkpoint. Each gets a
  lognormal accelerated wear-rate multiplier, mean `1.0`, CV `0.25`.
- Common random numbers: virtual robot `i` retains its multiplier at every
  checkpoint. Changes across the curve are therefore due to wear coordinate,
  not re-sampling a new virtual population.
- Orange bands are 95% Wilson intervals for the conditional fraction of
  floor-level falls.

## Healthy Controls

Every reported sweep included a healthy exact-start gate: `100/100` completed
without floor-level falls. Additional direct healthy probes completed:

| Policy | Duration | Result |
| --- | ---: | --- |
| Short dance | 16.24 s | 100/100 completed |
| Full dance | 131.48 s | 100/100 completed in the full-dance sweep |
| Walk, `vx=0.4` | 60 s | 100/100 completed in the walk sweep |
| Walk, `vx=0.8` | 60 s | single direct healthy probe completed |

The full dance can transiently lower pelvis height to about `0.374 m` during
the choreography while torso remains about `0.673 m`; this is why a pelvis-only
threshold would create false falls for that motion.

## Results

## Critical Interpretation Boundary

The completed three curves answer only:

> How tolerant is each closed-loop skill to one common actuator-degradation pattern?

They do **not** answer which skill physically causes faster degradation per
execution. Reusing the short-dance profile means that full dance and walking
were evaluated with an externally imposed, identical degradation pattern.
Therefore the apparent ordering below must not be described as a measured
motion-induced wear ranking.

### Motion-Specific Load Profiles: Next Experiment

The required causal chain is:

`motion -> its own joint load -> its own degradation -> actuator capability -> functional failure`.

For each healthy native trace we now calculate
`S_j,m = integral(abs(tau_j(t) * qdot_j(t)) dt)`. Both the 29-joint shape and
the total `sum_j S_j,m` matter. The current reference alpha is scaled by the
ratio of total work-like proxy, rather than normalizing away the magnitude.

| Motion | Trace duration | Total work-like proxy | Relative to short-dance reference |
| --- | ---: | ---: | ---: |
| Full dance | 131.48 s | 21717.7 | 4.758x |
| Walk `vx=0.4` | 60.00 s | 3047.4 | 0.668x |

These are still accelerated proxy values, not calibrated physical wear rates.
They are, however, motion-specific and will be used by the next sweeps instead
of the common profile. The profiles are generated at
`outputs/experiments/motion_profiles/`.

### Short Dance: 16.24 s

![Short-dance transition](figures/short_dance_transition.png)

| Completed dances | Falls / 100 |
| ---: | ---: |
| 500k | 1 |
| 550k | 3 |
| 600k | 3 |
| 650k | 12 |
| 700k | 18 |
| 750k | 26 |
| 800k | 37 |
| 850k | 47 |
| 900k | 59 |
| 950k | 64 |
| 1M | 71 |

The transition is broad, roughly `650k--950k`; the 50% crossing is between
`850k` and `900k`. Individual outcomes are not strictly monotone under
nonlinear closed-loop contacts, so this is an aggregate conditional curve, not
an irreversible per-robot damage threshold.

### Full Dance: 131.48 s

![Full-dance transition](figures/full_dance_transition.png)

| Completed dances | Falls / 100 |
| ---: | ---: |
| 700k | 49 |
| 800k | 69 |
| 850k | 81 |
| 900k | 86 |
| 950k | 91 |
| 1M | 91 |

The full dance is the least robust of the tested policies. Its aggregate 50%
crossing is immediately above `700k`; the largest observed increment is from
`800k` to `850k`.

### Walking: `vx=0.4 m/s`, 60 s

![Walking transition](figures/walk_vx04_transition.png)

| Completed dances | Falls / 100 |
| ---: | ---: |
| 1.5M | 42 |
| 1.6M | 53 |
| 1.7M | 66 |
| 1.8M | 71 |
| 1.9M | 79 |
| 2.0M | 83 |

Walking is substantially more robust under this particular wear mapping. Its
50% crossing lies between `1.5M` and `1.6M`; the largest observed increment is
from `1.6M` to `1.7M`.

## Completed Common-Pattern Interpretation

Under the completed common-pattern assumption, the approximate ordering of
**tolerance to that common pattern** is:

`full dance < short dance < walk vx=0.4`.

This is a useful policy-level result: the same per-joint torque degradation
produces different failure transitions depending on the closed-loop task. It
does **not** establish that full dance accumulates wear faster than walking,
nor that a physical G1 will fail at these repetition counts.

The RMS panels include frames after a robot has fallen, so their large values
quantify total trajectory loss rather than pre-fall tracking quality. A useful
next analysis is a pre-fall-only trajectory-error curve and a time-to-first-fall
distribution per checkpoint.

## Reproducibility and Artifacts

Canonical generated experiment directories:

- `outputs/experiments/full_dance_transition/`
- `outputs/experiments/walk_vx04_transition/`
- `mvp_mujoco/outputs/rb_y_16s/native_health_transition_500k_1m/`

Each contains `summary.json`, `summary.csv`, detailed per-trial CSV files and
raw `.npz` traces. Raw output is excluded from Git due to size; the three
summary figures above are versioned in `docs/figures/`.

Relevant runners:

```bash
# Dense sweep engine
./dance_sim/.venv/bin/python mvp_mujoco/scripts/15_run_native_health_sweep.py --help

# Native policies
./dance_sim/run_native_mujoco_deploy.sh --help
./dance_sim/run_native_mujoco_velocity.sh --help

# Plot a completed sweep
MPLBACKEND=Agg ./dance_sim/.venv/bin/python \
  mvp_mujoco/scripts/17_plot_native_health_transition.py \
  --input outputs/experiments/walk_vx04_transition/summary.json
```

Two illustrative local videos are available but intentionally not committed:

- `outputs/cross_policy_wear/full_dance_r1m_fall.mp4`
- `outputs/cross_policy_wear/walk_vx04_r2m_fall.mp4`
