# Project status

Updated: 2026-08-25.

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

## Current decision

Use `num_envs=1` as the reference simulator path. Parallelize independent processes across GPUs for throughput. Any future `nworld > 1` optimization must first pass an identical-world healthy-control test and a paired comparison against the single-world reference.

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
