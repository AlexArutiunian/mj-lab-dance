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
