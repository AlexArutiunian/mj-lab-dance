# FEM-calibrated damage-aware dance policy

Status: **research design / scaffold only**.

This branch extends the existing WearBench-G1 multi-channel wear model with a
high-fidelity structural-impact channel. It does **not** replace the current
native-MuJoCo deploy reference and it does **not** claim that FEM output is
ground truth for a physical Unitree G1 before calibration.

## Research question

Can high-fidelity structural simulations of representative humanoid impacts be
compressed into a cheap damage surrogate and used inside RL so that a dance
policy preserves motion quality while reducing predicted structural damage?

The intended comparison is:

```text
baseline dance policy
vs
force/impulse-aware policy
vs
FEM-calibrated damage-aware policy
```

The key idea is to optimize a quantity closer to structural damage than a raw
contact-force threshold.

## Why this belongs in WearBench-G1

The current repository already uses a per-joint state

```text
x_j = [fatigue_j, impact_j, thermal_age_j, wear_j, backlash_j]
```

and currently has an uncalibrated load proxy based on

```text
integral(abs(tau_j * qdot_j) dt)
```

This branch focuses specifically on making the **impact** channel more physical.
The existing wear/fatigue/thermal channels remain separate.

## Two-level simulator architecture

Do not run a crash/FEM solver inside every RL control step.

```text
dance policy
    |
    v
MuJoCo / MJLab
fast closed-loop rollouts
    |
    | extract representative impact windows
    v
high-fidelity structural solver
Chrono::FEA / Abaqus Explicit / LS-DYNA
    |
    | stress, strain, plastic work, fastener/bearing loads
    v
labeled impact dataset
    |
    v
damage surrogate D_hat = f(features)
    |
    +----------------------+
                           |
                           v
                   MuJoCo / MJLab reward
```

The expensive solver is an **offline teacher**. The policy still trains in the
fast simulator.

## Event unit

The first implementation should model one contact/landing event, not an entire
16.24 s dance as one sample.

For every selected touchdown or collision, store a short window around first
contact. Suggested window:

```text
[-50 ms, +150 ms] around contact onset
```

The exact window is a parameter and must be checked for convergence.

## Fast-simulator feature vector

Initial candidate features:

```text
contact:
  F_peak
  J = integral(F dt)
  normal impulse
  tangential impulse
  impact energy proxy
  contact duration
  contact point / foot region

kinematics:
  foot linear velocity before contact
  foot angular velocity before contact
  base linear/angular velocity
  joint q
  joint qdot

actuation:
  joint torque
  actuator saturation margin

geometry/orientation:
  foot normal relative to ground
  knee/ankle configuration
  support phase
```

Do not assume this list is minimal. Feature ablations are part of the experiment.

## High-fidelity labels

A structural solver can produce multiple labels. Keep them separate first.

Candidate labels:

```text
peak von Mises stress
equivalent plastic strain
plastic work
peak fastener load
peak bearing load
peak gearbox-housing load
residual deformation
failure flag for a declared material/joint model
```

A derived scalar dashboard value may later be defined as

```text
D_impact = g(labels)
```

but the individual physical labels should remain available so that the scalar
cannot hide a bad fit.

## Surrogate

The surrogate approximates the high-fidelity mapping:

```text
D_hat = f_theta(z_event)
```

where `z_event` is the event feature vector extracted from MuJoCo/MJLab.

Candidate model order:

1. linear / polynomial baseline;
2. symbolic regression;
3. tree boosting;
4. small MLP only if the simpler models are inadequate.

Symbolic regression is especially valuable if it yields a compact interpretable
law, but an equation is not automatically a physical law: it remains an
empirical surrogate over the sampled domain.

## RL reward integration

Baseline example:

```text
r = r_motion + r_tracking - lambda_force * force_penalty
```

Candidate damage-aware reward:

```text
r = r_motion
  + r_tracking
  - lambda_damage * D_hat(event)
```

For repeated dance cycles, keep acute impact damage separate from long-term
fatigue:

```text
r = r_motion
  + r_tracking
  - lambda_impact * D_hat_impact
  - lambda_fatigue * Delta_D_fatigue
```

The reward coefficient controls policy preference; it is not a material
parameter and must not be interpreted as one.

## FEM sampling strategy

A naive grid over every state dimension will be too expensive. Use staged
sampling:

1. collect a large number of real policy impact events in native MuJoCo;
2. cluster events by landing/contact geometry and load regime;
3. select medoids plus tail events;
4. add targeted perturbations around high-risk events;
5. run the high-fidelity solver on the selected set;
6. fit a surrogate;
7. use surrogate uncertainty / residuals to request new high-fidelity samples.

This is an active-learning loop rather than a one-shot dataset.

## Recommended first structural scope

Do **not** begin with a detailed full-robot FEM model.

Start with one load path:

```text
foot -> ankle -> lower leg -> knee interface
```

and one event family:

```text
hard dance landings / one-foot touchdowns
```

This keeps material assumptions, meshing, contacts and solver cost auditable.

## Experiment matrix

At minimum compare:

| Condition | Motion tracking | Raw impact | FEM-surrogate damage | Energy | Falls |
|---|---:|---:|---:|---:|---:|
| Original policy | yes | yes | offline eval | yes | yes |
| Raw-force penalty | yes | yes | offline eval | yes | yes |
| Impulse penalty | yes | yes | offline eval | yes | yes |
| FEM-surrogate reward | yes | yes | yes | yes | yes |

Additional ablations:

- remove joint configuration from surrogate;
- remove impact orientation;
- use only `F_peak`;
- use only impulse;
- use only impact-energy proxy;
- symbolic model vs boosted trees / MLP;
- in-domain vs out-of-domain impact regimes.

## Validation requirements

A useful surrogate must report more than training error.

Required checks:

- held-out high-fidelity cases;
- error by impact regime, not only aggregate RMSE;
- calibration of high-risk tail;
- monotonicity/sanity checks where physically justified;
- extrapolation/OOD detector or confidence flag;
- timestep/contact-feature convergence in MuJoCo;
- mesh/material/contact sensitivity in the structural model;
- paired policy evaluation with identical initial conditions and motion target.

If the surrogate sees an event outside its training envelope, the run should
log `DAMAGE_SURROGATE_OOD` rather than silently extrapolate.

## Claims boundary

Until real hardware or validated component data are available, acceptable claims
are:

- the policy reduces **predicted** structural metrics under a declared model;
- the surrogate reproduces the chosen high-fidelity solver within measured
  held-out error;
- the policy trades tracking/energy against a model-based damage objective.

Do not claim:

- a physical G1 lifetime in hours/dances;
- real fracture probability;
- certified safe impact limits;
- that FEM alone is ground truth.

## Integration with the existing repository

Suggested data flow:

```text
native MuJoCo deploy runner
  -> event extraction
  -> research/fem_damage_reward/datasets/events.parquet  (generated, not Git)
  -> external structural-solver batch
  -> labeled_events.parquet                             (generated, not Git)
  -> fitted surrogate artifact                          (generated, not Git)
  -> damage reward adapter
  -> paired RL / evaluation runs
```

Generated datasets, solver outputs, meshes and trained artifacts should remain
outside Git unless a deliberately small reproducibility fixture is added.

## Milestones

### M0 — telemetry contract
Define event schema and prove deterministic extraction from the current
deploy-matched native MuJoCo run.

### M1 — impact baseline
Compare `F_peak`, impulse and impact-energy proxies on the existing dance.

### M2 — one-component structural model
Build and sensitivity-check the foot/ankle/lower-leg/knee load path.

### M3 — labeled impact dataset
Generate the first high-fidelity labels for representative and tail events.

### M4 — surrogate
Fit interpretable baselines and quantify held-out/OOD error.

### M5 — reward
Integrate the frozen surrogate into training/evaluation.

### M6 — policy comparison
Run the baseline/force/impulse/FEM-surrogate ablation with identical motion
targets and matched seeds.

### M7 — physical calibration
Only after simulator work is stable, use real G1 telemetry/component tests to
calibrate material/load parameters and validate ranking.

## Current branch deliverables

This branch currently contains:

- this design document;
- a machine-readable event/label schema;
- a dependency-light surrogate interface scaffold.

No FEM model, fitted coefficients, trained surrogate or RL result is claimed yet.
