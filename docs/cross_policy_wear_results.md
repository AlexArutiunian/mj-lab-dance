# Cross-Policy Wear Experiments

Updated: 2026-08-27.

## Scope

This note consolidates the current native-MuJoCo evidence for the Unitree G1
under the same accelerated, per-joint wear model and three deployed policies:

1. short `RB+Y` dance clip, 16.24 s;
2. full `dance1_subject2`, 131.48 s;
3. Unitree velocity-walking policy at `vx=0.4 m/s`, 60 s.

The result is a **conditional simulator sensitivity analysis**, not a measured
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
- Wear: all 29 actuators receive the same relative joint-load damage profile;
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

## Cross-Policy Interpretation

Under the current assumptions, the approximate ordering of robustness is:

`full dance < short dance < walk vx=0.4`.

This is a useful policy-level result: the same per-joint torque degradation
produces different failure transitions depending on the closed-loop task. It
does **not** establish that a physical G1 will fail at those repetition counts.
Calibration may shift every horizontal coordinate substantially.

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
