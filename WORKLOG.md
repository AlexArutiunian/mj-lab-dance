# Worklog

## 2026-08-25

- Confirmed the real deploy segment is `RB + Y`, `dance1_subject2_16s_faststart`, 16.24 s.
- Proved by motion-field matching that the short recut corresponds to full-motion interval 20.50..36.72 s, with small export rounding.
- Replaced transient-angle failure detection with pelvis/torso floor-level detection held for 0.50 s.
- Added motion start offset support to the local MJLab checkout.
- Added batched wear checkpoints, torque scaling and health overlays.
- Recorded diagnostic falls, then invalidated their scientific interpretation after healthy batches also fell.
- Measured batch-size dependence and tested larger environment spacing.
- Tested shared local origins; healthy 64-world batch still diverged and produced 10 falls.
- Added simulator quality gates that abort contaminated batched experiments.
- Established one-world-per-process as the reference execution path.
- Started repository/documentation cleanup for external review.
- Added reproducible tests, doctor command, Git repository and initial commit.
- Recomputed the load profile from the exact deploy-matched recut: right knee ranked first, left knee second.
- Added independent-process paired trial orchestration with multi-GPU scheduling and Wilson confidence intervals.
- Audited MuJoCo transmission indexing and found that earlier derating used joint-list indices as actuator IDs. Marked all earlier worn outcomes superseded.
- Added a shared `actuator_trnid` mapping layer and tests for reordered, missing and duplicate transmissions.
- Re-logged the exact deploy baseline with aligned torque, DOF, actuator-name and force-range telemetry. Right knee remained the highest load proxy: peak 113.56 Nm against a 139 Nm limit.
- Re-ran corrected exact-start checkpoints: 0, 100k and 500k completed; conditional 1M/right-knee 0.3x fell at 2.76 s.
- Re-ran exact-start dose points: right-knee 0.8x and 0.5x completed, while 0.3x fell.
- Demonstrated that uncalibrated reset perturbations can make healthy trials fall around 8--10 s. These trials are now treated as robustness stress tests and excluded from wear statistics by the healthy gate.
- Isolated reset channels: small XY-only, yaw-only and joint-only offsets each completed for seed 1; failures require some combined offsets and represent a nonlinear robustness boundary.
- Added explicit `deploy_exact` versus `stress_jitter` result classification.
- Passed a five-process `deploy_exact` healthy regression gate on seeds 1--5 with 5/5 successful and no floor-level falls.
- Superseded the 5/5 smoke conclusion after larger exact-start sweeps produced intermittent healthy floor-level falls around 8--11 s.
- Fixed healthy scale 1.0 mutating `actuator_forcelimited`; healthy runs now leave the model and CUDA graph untouched.
- Restored the original fixed-batch ONNX/direct CUDA binding for single-world runs and rejected duplicate workers on one physical GPU.
- Added explicit CUDA synchronization at MJWarp-observation/ORT-input and ORT-output/MJWarp-action boundaries. Targeted failing seeds improved, but a larger gate still failed; the simulator incident remains unresolved.
- Qualified CPU MJWarp physics with ONNX `CPUExecutionProvider`: 100 fresh processes on seeds 1--100 completed 100/100 with zero falls and identical trajectory extrema.
- Set CPU as the default independent-trial reference and labeled GPU evidence unqualified pending CPU trajectory equivalence.
- Re-ran corrected conditional checkpoints on the CPU reference: healthy, 100k/0.9159x and 500k/0.6085x completed; conditional 1M/0.3x fell at 2.90 s.
- Added per-step CPU/GPU forensic traces for observations, actions, state,
  controls, contacts, constraints and solver iterations.
- Demonstrated same-seed GPU self-divergence around control steps 44--50;
  Newton and CG are both affected, while PGS is unsupported in MJWarp 3.5.0.
- Implemented the Unitree native MuJoCo deploy loop with the exact ONNX,
  deploy observations, 50 Hz policy and 500 Hz torque-level PD control.
- Qualified native MuJoCo with 100 fresh processes: 100/100 completed, zero
  falls, identical minimum pelvis/torso and final-root metrics.
- Added realtime native viewing with health/torque-scale overlay and per-joint
  torque capability controls. Right-knee scales 1.0, 0.8 and 0.5 completed;
  scale 0.3 produced a held floor-level fall at 3.74 s.


## 2026-09-18

- Created `research/fem-calibrated-damage-reward` from `main`.
- Defined a two-level architecture where native MuJoCo/MJLab remains the fast
  closed-loop simulator and a structural solver is used offline to label
  representative impact events.
- Scoped the first high-fidelity model to the
  foot -> ankle -> lower-leg -> knee load path instead of a full-robot crash
  model.
- Defined impact-event features, high-fidelity labels, provenance and quality
  flags in `research/fem_damage_reward/DATA_SCHEMA.md`.
- Added a dependency-light damage-surrogate interface with explicit OOD
  rejection and deliberately no default physical coefficients.
- Defined the baseline / force / impulse / FEM-surrogate policy comparison,
  validation checks and claims boundary.
- No FEM result, fitted damage law, RL improvement or physical lifetime result
  is claimed by this branch yet.
