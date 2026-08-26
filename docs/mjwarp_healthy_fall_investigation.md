# MJWarp healthy fall investigation

Updated: 2026-08-26. Status: **NATIVE DEPLOY REFERENCE QUALIFIED; GPU MJWARP NONDETERMINISTIC**.

## Problem statement

The tested artifact is the deploy `RB + Y` motion
`dance1_subject2_16s_faststart.npz` and the ONNX policy from the same directory.
The clip is 16.24 s and corresponds to full-motion time 20.50--36.72 s.

The robot operator reports that the same policy and 16 s segment were run
sequentially on the physical G1 approximately 100 times without a fall. This is
user-provided real-robot evidence; raw telemetry for those runs has not yet been
imported into this repository.

In MJLab/MJWarp, nominally healthy robots sometimes reach floor level around
short-clip time 8--11 s. The primary detector requires both
`pelvis_z < 0.35 m` and `torso_z < 0.30 m` continuously for 0.50 s. These are
full-body floor contacts, not temporary dance crouches or angle excursions.

Consequently, GPU simulator survival/failure rates remain blocked. A qualified
CPU reference is now available for subsequent simulator experiments.

## Qualified resolution

MJWarp physics on `device=cpu` together with ONNX Runtime
`CPUExecutionProvider` completed 100 fresh processes, seeds 1--100:

```text
valid trials:       100/100
successful dances: 100/100
floor-level falls: 0/100
minimum pelvis z:  0.4101710319519043 m in every process
minimum torso z:   0.7023436427116394 m in every process
wall time:         2070.267 s with eight CPU workers
summary SHA-256:   03e166095b0c5cf97e90466c5178d072daea267ba502b17c0d4e5c506f16e414
```

The identical extrema across all 100 processes demonstrate deterministic CPU
playback. The operational fix is to use CPU physics and CPU ONNX as the
scientific reference. It does not repair GPU MJWarp internals; GPU acceleration
remains disabled for evidence until trajectory equivalence is established.

## Immutable provenance

```text
motion SHA-256: db60ad9d3423e8261c03b6e2294cfbdcc12c3a54c744dbc6e204b9a981d46eb8
policy SHA-256: 0f1922f5b20189cc0307a8f91ac9eb655f07527805104d1653531d9e469c3231
MuJoCo:         3.5.0
mujoco-warp:    3.5.0
Warp:           1.15.0
physics dt:     0.005 s
control dt:     0.020 s
local GPU:      NVIDIA GeForce RTX 3080 Ti Laptop GPU, 16 GiB
```

## Reproduction

From `mvp_mujoco`:

```bash
./run_independent_trials.sh \
  --trials 100 \
  --devices cuda:0 \
  --seed-start 1 \
  --checkpoints 0 \
  --target-scale 1.0 \
  --duration 16.24 \
  --damage-profile outputs/rb_y_16s/damage_profile_1m_scale03.json \
  --out-dir outputs/rb_y_16s/deploy_exact_gate_100
```

Each trial is a fresh Python process, one MJWarp world, no domain randomization,
no observation corruption, no pose/yaw/joint jitter, motion start at 0, and no
wear. A healthy fall invalidates the trial and aborts its worn checkpoints.

## Evidence observed

An initial four-process run on the same physical GPU produced 7 healthy falls
among 100 attempted processes, at seeds 30, 31, 37, 49, 56, 81 and 98. Failure
times ranged from 8.06 to 11.34 s. Reusing one GPU for concurrent workers is now
forbidden because it changes outcomes and is not an independent-device setup.

Sequential post-fix gates also found healthy falls, including seeds 3, 6, 10,
11 and 30 in different sweeps. Several seeds passed when immediately repeated,
showing execution-path or numerical sensitivity rather than a stable physical
randomized initial condition.

The baseline telemetry logger frequently completes seeds that fail in the fast
survival runner. The logger performs per-step GPU-to-CPU telemetry copies, so
CUDA scheduling/synchronization remains a leading suspect. Explicit barriers
before ONNX reads and before MJWarp consumes actions improved targeted repeats
but did not make a larger sweep failure-free.

## Bugs already fixed

1. **Wrong joint-to-actuator derating mapping.** Joint-list indices were treated
   as actuator IDs. Mapping now uses MuJoCo `actuator_trnid`; right knee maps to
   actuator 18. This invalidated all older worn outcomes but did not explain all
   healthy falls.
2. **Healthy path enabled torque limiting.** Applying scale 1.0 set
   `actuator_forcelimited=1` for every drive. Healthy scale now returns without
   model mutation; mixed scales preserve nominal flags and enable limiting only
   for degraded drives.
3. **Single-world used rewritten dynamic ONNX.** `num_envs=1` now uses the
   original fixed-batch policy and direct CUDA output binding. Dynamic ONNX is
   diagnostic multi-world only.
4. **Duplicate workers on one GPU.** The independent runner rejects duplicate
   device names. Parallel jobs require distinct physical GPUs.
5. **Cross-library CUDA boundaries.** Explicit device synchronization was added
   before ORT reads MJWarp observations and before MJWarp consumes ORT actions.

## Attempts that did not solve the incident

- changing the fall criterion from transient orientation to held pelvis/torso
  floor level;
- using only one world;
- increasing environment spacing or moving worlds to local origins;
- disabling events, domain randomization, terminations and observation noise;
- exact motion-start reset with zero pose, velocity and joint jitter;
- replacing the standard environment wrapper with the minimal FastPlay loop;
- preserving a healthy model without force-limit graph rebuild;
- original fixed-batch ONNX with direct CUDA I/O binding;
- CUDA synchronization on both ORT/MJWarp boundaries.

These changes remove known confounders and must remain. They were insufficient
to qualify GPU execution but are retained in the qualified CPU reference.

## Paired-trace localization

The runner now optionally records policy observations, actions, qpos, qvel,
controls, contact counts, constraint counts and solver iterations at every
control step. CPU/GPU traces for seed 10 begin with float32-scale differences
and first disagree in contact/constraint counts at control step 34 (`0.68 s`).
The closed-loop policy then amplifies the state difference.

More importantly, repeated GPU/GPU traces with the same seed and command are
not reproducible. Newton repeats first diverged in qpos at steps 44--48 and in
contact sets later. CG produced two identical repeats followed by a third that
diverged at step 50. MJWarp 3.5.0 does not support PGS. This rules out CPU being
the sole source of the discrepancy and localizes the incident to the GPU
MJWarp execution path plus closed-loop amplification.

MuJoCo Warp's upstream determinism request remains open as issue `#562`.

## Native MuJoCo deploy reference

`dance_sim/scripts/native_mujoco_deploy.py` implements the Unitree deploy loop
without DDS: the upstream `scene_g1.xml`, 2 ms native MuJoCo physics, 20 ms
policy period, deploy `kp/kd`, action scale/offset, motion indexing and torso
orientation observation. The ONNX hash matches the Unitree deploy artifact.

Results from 100 fresh processes:

```text
completed:          100/100
floor-level falls:  0
min pelvis height:  0.492385 m
min torso height:   0.784822 m
final root height:  0.758102 m
```

Three full traced repeats were bit-identical. The backend runs around `5x`
realtime headless and therefore supports smooth realtime visualization without
GPU physics.

## Current hypotheses

1. MJWarp contact/constraint kernels are numerically nondeterministic near a
   marginally stable dynamic segment, potentially because of GPU atomics or
   solver ordering.
2. The policy is less robust in simulation than on hardware because contact,
   actuator, latency or state-estimation models differ from the deployed G1.
3. Confirmed: native MuJoCo and MJWarp diverge before the visible fall, and GPU
   MJWarp also diverges from itself across identical repeats.
4. Per-step telemetry synchronization changes scheduling enough to keep a
   marginal trajectory on the stable branch.

## Required next steps

1. Compare native simulator state against real G1 telemetry from the reported 100
   successful runs, especially around clip time 8--11 s.
2. Calibrate the native actuator degradation model against measured torque,
   current, temperature and fault telemetry.
3. Requalify a future MJWarp release only after same-seed self-repeats are
   deterministic and a 100-process healthy gate passes.

## Claim boundary

Do not claim that the physical robot has a baseline fall probability, that
scale 0.3 has a validated survival rate, or that one million dances predicts a
physical lifetime. Native MuJoCo is qualified as the deterministic deploy
simulator baseline. The wear law still requires physical calibration, and GPU
MJWarp results remain stress evidence rather than deterministic baseline data.
