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
- Ran validated single-world checkpoints through conditional 1M state; all four checkpoints completed without a physical fall, including right-knee torque scale 0.3.
- Added independent-process paired trial orchestration with multi-GPU scheduling and Wilson confidence intervals; end-to-end healthy/1M smoke trial passed.
