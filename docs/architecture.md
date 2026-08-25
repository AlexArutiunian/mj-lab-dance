# Architecture

```text
deploy motion + ONNX policy
        |
        v
single-world MJLab rollout ----> simulator quality gates
        |
        v
telemetry (q, qdot, tau, contacts, pose, events)
        |
        v
load feature extraction
        |
        v
damage-state update / analytical fast-forward
        |
        v
explicit degradation mapping to MuJoCo actuators/joints
        |
        v
paired healthy/worn candidate rollout
        |
        v
failure events + survival/RUL summaries + article figures
```

`mvp_mujoco/wearbench/` owns model-independent damage and MuJoCo degradation utilities. `mvp_mujoco/scripts/` owns experiment orchestration. `dance_sim/` owns motion identity, playback and deploy-policy integration. Upstream MJLab modifications are kept as patches rather than silently vendored.

The simulator validity layer and mathematical wear model are separate. A failed healthy-control gate invalidates simulator evidence but does not by itself validate or invalidate the analytical damage equations.
