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

The single-world healthy baseline is stable. Large MJWarp `nworld` runs are currently not scientifically valid: identical healthy worlds diverge numerically and some fall around the most dynamic part of the motion. Therefore the previously observed healthy failure counts such as `32/256` are simulator batch artifacts, not policy failures and not wear effects.

The deploy clip is a numerically equivalent recut of the full motion interval:

```text
short clip 0.00 s  == full motion 20.50 s
short clip 8.00 s  == full motion 28.50 s
short clip 9.00 s  == full motion 29.50 s
```

The recut is not bit-identical because of export rounding: measured maximum differences are approximately `1e-4 rad` for joint position and `0.003 rad/s` for joint velocity.

Primary evidence must use one MJWarp world per process until the batch implementation passes the healthy-control quality gate.

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

## Valid healthy smoke test

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

## Scientific scope

The current `alpha` is an intentionally accelerated and uncalibrated coefficient. Current repetition counts are integration stress-test coordinates, not predictions of physical G1 lifetime. See [docs/math_model.md](docs/math_model.md) and [docs/experiment_protocol.md](docs/experiment_protocol.md).
