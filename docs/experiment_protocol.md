# Experiment protocol

## A. Simulator qualification

1. Verify motion/policy paths and hashes.
2. Run the healthy deploy clip in one world.
3. Run the healthy full motion in one world.
4. Record qpos, qvel, action, actuator torque, contacts, pelvis/torso pose and solver warnings.
5. Repeat across seeds without physical randomization; results must be invariant within tolerance.
6. Compare against native MuJoCo if available.

No wear result is accepted until this stage passes.

## B. Load profile

Run the deploy-matched 16.24 s recut and compute per-joint:

```text
energy proxy integral |tau*qdot| dt
RMS/peak torque
RMS/peak velocity
torque cycle ranges/counts
contact impulse contribution
temperature/current when available
```

Check timestep convergence and repeatability.

## C. Conditional degradation

Choose a declared parameter set or distribution. Fast-forward mathematical state to requested repetitions, apply degradation to the simulator and run exactly one candidate dance.

Each worn run has a paired healthy run with the same seed and perturbation. Report both absolute and paired outcomes.

## D. Failure events

Primary physical fall:

```text
pelvis_z < 0.35 m AND torso_z < 0.30 m for >= 0.50 s
```

Also report tracking loss, joint limit, actuator saturation, thermal limit, NaN/Inf and solver failure separately.

## E. Reset protocols

Primary wear sensitivity uses `deploy_exact`: the exact initial state of the
real `RB+Y` clip, with no synthetic pose, yaw or joint offset. Repeated exact
seeds are a determinism/regression gate, not Monte Carlo samples.

`stress_jitter` may perturb the initial state, but its output is a policy and
simulator robustness test. It is not a wear survival estimate unless the joint
distribution of initial pose, orientation and joint error has been measured on
the real robot. A healthy fall invalidates the complete paired trial.

For lifetime uncertainty, vary calibrated wear-law and actuator parameters
while retaining `deploy_exact`; do not manufacture a lifetime distribution by
randomizing an uncalibrated initial state.

## F. Statistical reporting

Report:

- trials and seeds;
- valid/invalid trial counts;
- survival proportion with confidence interval;
- failure-time distribution;
- per-joint health/damage distribution;
- sensitivity to model parameters;
- simulator and hardware provenance.

Do not report contaminated batch failures as robot survival statistics.

## G. Acceleration

Correctness-preserving order:

1. analytically fast-forward damage state;
2. run independent single-world processes;
3. distribute jobs across V100, RTX 3060 and the laptop GPU;
4. validate a small `nworld` implementation against independent jobs;
5. increase batch size only after equivalence tests pass.

GPU speed does not compensate for an invalid healthy baseline.

The implemented entry point is:

```bash
cd mvp_mujoco
./run_independent_trials.sh --trials 10 --devices cuda:0,cuda:1
```
