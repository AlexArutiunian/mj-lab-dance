# Self review

## Current strengths

- Deploy motion/policy identity is now explicit.
- Failure detection corresponds to a robot physically reaching floor level.
- Simulator artifacts are separated from wear effects.
- Mathematical assumptions and scientific limitations are documented.
- Heavy generated data and local environments are excluded from Git.

## Current weaknesses

- The wear coefficient is uncalibrated.
- The current degradation mechanism is dominated by torque derating.
- Large MJWarp batches are not equivalent to the single-world reference.
- Existing output directories contain many exploratory and invalid runs.
- Real G1 telemetry and component endurance data are not yet integrated.

## Before paper-quality claims

- Produce clean independent-process Monte Carlo runs.
- Recompute damage from the exact deploy clip.
- Add provenance hashes and confidence intervals.
- Validate simulator trajectories against native MuJoCo and real robot logs.
- Calibrate at least one degradation channel or explicitly present only normalized sensitivity results.
