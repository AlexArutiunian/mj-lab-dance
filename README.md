# WearBench-G1

Research prototype for simulation-based joint wear and functional lifetime analysis of a Unitree G1 executing a repeated dance.

The repository combines:

- the deploy-matched `RB + Y` 16.24 s motion and ONNX policy;
- MuJoCo/MJLab telemetry collection;
- per-joint load and degradation models;
- actuator torque-capability degradation;
- fall/survival evaluation;
- an RA-L working draft.

## Current conclusion

The one-robot deploy reference now uses native MuJoCo with the Unitree
`scene_g1.xml`, the deploy ONNX policy and the C++ deploy observation/control
semantics. It passed 100 fresh processes with `100/100` completed dances and
zero floor-level falls. All physical extrema were identical: minimum pelvis
height `0.492385 m`, minimum torso height `0.784822 m`, and final root height
`0.758102 m`. Headless execution is approximately `5x` real time on this
laptop, so the native viewer can display the dance smoothly without using GPU
physics.

GPU MJWarp remains **not qualified for deterministic single-world evidence**.
Paired traces show GPU runs with the same seed and inputs diverging around
control step 48--50, followed by different contact/constraint sets. Newton and
CG both reproduce the issue; PGS is unsupported by MJWarp 3.5.0. GPU MJWarp
remains appropriate for training and explicitly labeled robustness/stress
batches, not for the healthy deploy baseline.

Read [docs/mjwarp_healthy_fall_investigation.md](docs/mjwarp_healthy_fall_investigation.md)
before interpreting any survival result.

Run the deterministic deploy simulator:

```bash
./dance_sim/run_native_mujoco_deploy.sh --viewer
```

Run a headless right-knee torque-capability checkpoint:

```bash
./dance_sim/run_native_mujoco_deploy.sh \
  --joint-torque-scale right_knee_joint=0.5
```

Run the native 100-member conditional health ensemble:

```bash
cd mvp_mujoco
./run_native_health_sweep.sh \
  --trials 100 --workers 6 --seed 20260826 --wear-rate-cv 0.25 \
  --out-dir outputs/rb_y_16s/native_health_ensemble_100
```

This has two deliberately separate controls. At `0` completed dances all 100
processes are exact replicas: this is a deterministic regression gate, and
therefore they must have the same physical outcome. At worn checkpoints,
each virtual robot receives an independent lognormal multiplier for the
accelerated joint-wear rate (mean `1.0`, CV `0.25`) while preserving the exact
initial state, policy, motion and physics. Thus the resulting completion count
is conditional parameter sensitivity, **not** a population-level G1 failure
probability. The runner writes complete traces and percentiles of joint,
action, pelvis, torso and orientation deviation into `summary.json` and
`summary.csv`.

Render all saved ensemble traces as four 4K `10x10` collages plus four
full-HD representative videos:

```bash
cd mvp_mujoco
MUJOCO_GL=egl ./run_render_native_health_ensemble.sh
```

The renderer replays saved native traces rather than rerunning the controller,
so every video corresponds exactly to the recorded trial metrics. Videos are
generated under the ensemble output directory and excluded from Git because
the high-quality evidence set is large.

The deploy clip is a numerically equivalent recut of the full motion interval:

```text
short clip 0.00 s  == full motion 20.50 s
short clip 8.00 s  == full motion 28.50 s
short clip 9.00 s  == full motion 29.50 s
```

The recut is not bit-identical because of export rounding: measured maximum differences are approximately `1e-4 rad` for joint position and `0.003 rad/s` for joint velocity.

Primary evidence must use one MJWarp world per process until the batch implementation passes the healthy-control quality gate.

A transmission-index audit also found that earlier worn runs applied
joint-ordered scales to actuator-array positions. The corrected implementation
maps through MuJoCo `actuator_trnid`. With that correction, exact-start tests
complete at right-knee target scales `0.8` and `0.5`, while `0.3` produces a
floor-level fall at 2.76 s. This is conditional simulator sensitivity, not a
physical lifetime prediction.

## Repository map

```text
dance_sim/          deploy-matched playback tools and lightweight assets
mvp_mujoco/         wear model, experiment runners and visualization
article_RA-L/       working paper draft
docs/               architecture, protocol, model and known issues
patches/            local changes required in upstream unitree_rl_mjlab
```

Read [docs/status.md](docs/status.md) before interpreting any generated result.

## Setup

The local bundle already uses `dance_sim/.venv`. For a clean machine, install upstream `unitree_rl_mjlab` at commit `1425b15`, create a Python 3.11 environment, install `mvp_mujoco/requirements.txt`, and apply:

```bash
git -C dance_sim/external/unitree_rl_mjlab apply ../../../patches/unitree_rl_mjlab_motion_start.patch
```

## Legacy MJWarp smoke test

```bash
cd mvp_mujoco
./run_batched_wear_survival.sh \
  --motion-file ../dance_sim/assets/policies/mimic/dance1_subject2_16s_faststart/params/dance1_subject2_16s_faststart.npz \
  --policy ../dance_sim/assets/policies/mimic/dance1_subject2_16s_faststart/exported/policy.onnx \
  --num-envs 1 \
  --duration 16.24 \
  --checkpoints 0 \
  --target-scale 1.0 \
  --out-dir outputs/validated_single_world_baseline
```

The run is valid only if `failed_envs == 0` and `baseline_validation == "valid"`.

## Independent paired stress tests

Use independent single-world processes for paired robustness experiments:

```bash
cd mvp_mujoco
./run_independent_trials.sh \
  --trials 10 \
  --checkpoints 0 \
  --devices cpu
```

Multiple CPU worker processes may be requested with `--devices cpu,cpu,...`.
GPU execution remains diagnostic until it passes equivalence against the CPU
reference. Every trial runs its own healthy control before worn checkpoints
with the same seed. A failed healthy control invalidates the pair.

## Scientific scope

The current `alpha` is an intentionally accelerated and uncalibrated coefficient. Current repetition counts are integration stress-test coordinates, not predictions of physical G1 lifetime. See [docs/math_model.md](docs/math_model.md) and [docs/experiment_protocol.md](docs/experiment_protocol.md).
