# Project status

Updated: 2026-08-26.

## Confirmed

- Correct deploy action is `RB + Y`.
- Correct motion is `dance1_subject2_16s_faststart.npz`, duration 16.24 s.
- Correct policy is the ONNX file from the same `dance1_subject2_16s_faststart` directory.
- The short clip is a numerically equivalent recut of full `dance1_subject2` starting at 20.50 s. Export rounding is bounded at about `1e-4 rad` in joint position and `0.003 rad/s` in joint velocity.
- A single healthy MJWarp world completes both the short clip and the full motion in existing probes.
- The floor-level fall detector uses pelvis, torso and a 0.50 s hold.
- Health/torque state can be rendered in diagnostic videos.
- Joint load telemetry and actuator derating are now aligned through MuJoCo
  `actuator_trnid`; joint-list position is never treated as an actuator ID.

## Invalidated

All survival/failure percentages obtained from large `nworld` runs before the simulator baseline issue is fixed are invalid for scientific claims. This includes healthy falls near 28–29 s in the full motion and 8–9 s in the short clip.

The timing match is expected because:

```text
20.50 + 8.00 = 28.50 s
20.50 + 9.00 = 29.50 s
```

The full-motion and short-clip incidents are the same motion segment.

## Current blockers

MJWarp worlds initialized with identical policy, motion and no randomization diverge in large batches. A 64-world local-origin control reached a maximum state spread of 4.37 and produced 10 healthy falls. This remains true after removing the XY origin grid, so coordinate placement is not the root cause.

Artificial reset perturbations are not yet a validated Monte Carlo distribution.
Even `2 mm`, `0.5 deg` and `0.002 rad` perturbations can make the healthy policy
fall around short-clip 8.3--8.6 s. These runs measure simulator/policy robustness,
not wear, and are rejected by the healthy-control quality gate.

Channel isolation with seed 1 showed that `2 mm` XY alone, `0.5 deg` yaw alone
and `0.002 rad` joint jitter alone each completed. The failure appears only for
some combined offsets, so it is a nonlinear policy robustness boundary rather
than one broken reset field. Primary wear runs now use the explicit
`deploy_exact` protocol; combined perturbations are labeled `stress_jitter`.

An early `deploy_exact` gate completed five independent MJWarp processes with
healthy `5/5`. This was only a smoke test and was not sufficient: subsequent
larger gates found healthy failures. The active incident and attempted fixes are
documented in `docs/mjwarp_healthy_fall_investigation.md`.

## Qualified healthy reference

The primary one-robot deploy reference is now native MuJoCo using Unitree's
`scene_g1.xml`, the exact deploy ONNX and the observation/control equations in
`State_Mimic.cpp` and `deploy.yaml`. A 100-process gate completed `100/100`
dances with zero falls. Every process produced:

```text
minimum pelvis height: 0.492385 m
minimum torso height:  0.784822 m
final root height:     0.758102 m
```

Three traced full runs were bit-identical across observations, actions, qpos,
qvel, controls and contact counts. Native execution reached roughly `5x` real
time headless and supports smooth realtime viewing.

Native conditional right-knee capability points produced:

```text
scale 1.0: completed
scale 0.8: completed
scale 0.5: completed
scale 0.3: floor-level fall at 3.74 s
```

These points validate simulator sensitivity only. The mapping from repetitions
to torque scale remains uncalibrated.

## Legacy CPU MJWarp reference

The CPU MJWarp physics + ONNX `CPUExecutionProvider` path passed 100 fresh
processes with seeds 1--100: `100/100` successful, `0` floor-level falls, all
trials valid. Every run produced identical extrema, including minimum pelvis
height `0.410171 m` and minimum torso height `0.702344 m`. Wall time with eight
CPU workers was 2070.27 s. The summary SHA-256 is
`03e166095b0c5cf97e90466c5178d072daea267ba502b17c0d4e5c506f16e414`.

This CPU MJWarp result remains a useful cross-check, but native MuJoCo is the
primary deploy reference because it follows the Unitree simulation-deployment
path and runs fast enough for realtime one-robot playback.

GPU MJWarp remains unqualified. Same-seed paired traces first differ in qpos at
control step 44--50 depending on the repeat and later show different contacts
and constraints. CG did not eliminate the divergence and PGS is unsupported in
MJWarp 3.5.0.

Corrected CPU conditional checkpoints on seed 1 then produced:

```text
R=0:       scale 1.0000, completed
R=100k:    right-knee scale 0.9159, completed
R=500k:    right-knee scale 0.6085, completed
R=1M:      right-knee scale 0.3000, floor-level fall at 2.90 s
```

This is a deterministic sensitivity curve under an uncalibrated wear law, not
a physical lifetime or survival probability.

## Next work

1. Finish a validated single-world experiment runner with repeatable seeds and provenance.
2. Recompute the load profile from the exact 16.24 s deploy clip.
3. Separate simulator validation from wear-model calibration.
4. Replace the scalar accelerated proxy with a parameterized multi-channel model.
5. Add real-robot telemetry schema for current, torque, temperature and faults.
6. Run paired healthy/worn Monte Carlo jobs as independent processes.

## Corrected conditional checkpoint run

An audit found that earlier scripts applied joint-ordered torque scales directly
to actuator indices. MJLab's joint and actuator orders differ; for example the
right knee is joint-list index 9 but MuJoCo actuator 18. Earlier worn outcomes
are therefore superseded. Load telemetry itself was in joint order and the
right-knee load ranking remains valid.

After mapping through `actuator_trnid`, single-world exact-start `RB+Y`
checkpoints produced:

```text
R=0:       right-knee scale 1.0000, fall 0/1
R=100k:    right-knee scale 0.9159, fall 0/1
R=500k:    right-knee scale 0.6085, fall 0/1
R=1M:      right-knee scale 0.3000, fall 1/1 at 2.76 s
```

An exact-start dose check gave:

```text
right-knee target scale 0.8: completed
right-knee target scale 0.5: completed
right-knee target scale 0.3: floor-level fall at 2.76 s
```

These are deterministic simulator sensitivity points, not survival
probabilities. The mapping from `1M` to `scale=0.3` is still an uncalibrated
model assumption and must not be described as measured physical lifetime.

## Native conditional 100-member health ensemble

The deterministic reference gate and the worn-population sensitivity test are
separate experiments. The first has 100 exact copies of the healthy model;
they are required to agree and completed `100/100`. The second keeps that
exact start, policy, motion and native-MuJoCo physics, but samples one
accelerated wear-rate multiplier per virtual robot from
`LogNormal(mean=1.0, CV=0.25)`. It therefore represents uncertainty in the
uncalibrated wear model, not a measured G1 fleet failure rate.

The completed native run (`seed=20260826`, 100 virtual robots per checkpoint,
16.24 s `RB+Y` dance) produced:

| Completed dances | Completed / 100 | Falls / 100 | Median weakest-joint health | Median weakest torque scale | Median joint-position RMS | Median pelvis RMS | Median torso orientation RMS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 0 | 1.000 | 1.000 | 0.000 rad | 0.000 m | 0.000 deg |
| 100k | 100 | 0 | 0.938 | 0.918 | 0.009 rad | 0.022 m | 0.818 deg |
| 500k | 99 | 1 | 0.683 | 0.608 | 0.020 rad | 0.040 m | 1.380 deg |
| 1M | 31 | 69 | 0.393 | 0.322 | 0.255 rad | 0.628 m | 57.965 deg |

The full p05/p50/p95 metrics and every trace are written under
`mvp_mujoco/outputs/rb_y_16s/native_health_ensemble_100/`. This supports the
claimed qualitative conclusion: under this assumed damage law, increasing wear
first increases trajectory tracking error and only later produces floor-level
falls. It does **not** establish when a physical G1 reaches those checkpoints.

## Dense transition sweep: 500k--1M

A denser native-MuJoCo conditional sweep used 100 virtual robots at every
`50k`-dance checkpoint. It reused each virtual robot's same wear-rate
multiplier across all checkpoints (`common_random_numbers=true`), keeping the
initial state, policy, motion and physics exact. This avoids mistaking a new
parameter sample for a change due to wear.

| Completed dances | Floor falls / 100 |
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

The aggregate conditional fall curve is a broad transition rather than a
single discontinuity. Its 50% crossing lies between `850k` and `900k`; the
steepest observed aggregate increment is `+12/100` over that interval. PNG,
PDF, Wilson intervals and all source traces are in
`mvp_mujoco/outputs/rb_y_16s/native_health_transition_500k_1m/`.

Individual closed-loop outcomes are not strictly monotone: there were 82
adjacent `completed -> fall` switches and 12 `fall -> completed` switches.
Thus an individual first fall is not an irreversible-damage threshold in this
model. The appropriate reported result is the aggregate conditional fall
fraction with its uncertainty interval, not a claimed physical bifurcation or
real-G1 lifetime limit.
