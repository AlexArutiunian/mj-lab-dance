# Impact event dataset schema

This is the contract between the fast humanoid simulator, the high-fidelity
structural solver and the learned damage surrogate.

## Identity / provenance

Required per event:

```text
event_id
run_id
timestamp_s
seed
motion_sha256
policy_sha256
simulator_name
simulator_version
model_asset_hash
control_dt_s
physics_dt_s
protocol
```

## Contact features

```text
body_a
body_b
contact_region
contact_duration_s
force_peak_n
normal_impulse_ns
tangent_impulse_ns
impulse_norm_ns
impact_energy_proxy_j
contact_normal_x
contact_normal_y
contact_normal_z
```

## Pre-impact kinematics

```text
foot_vx_mps
foot_vy_mps
foot_vz_mps
foot_wx_radps
foot_wy_radps
foot_wz_radps
base_vx_mps
base_vy_mps
base_vz_mps
base_wx_radps
base_wy_radps
base_wz_radps
```

## Joint state

Store named vectors rather than relying on array positions:

```text
joint_names[]
joint_q_rad[]
joint_qdot_radps[]
joint_tau_nm[]
actuator_margin[]
```

The implementation must resolve actuator/joint mapping explicitly. Existing
WearBench-G1 rules about `actuator_trnid` still apply.

## High-fidelity labels

Keep raw labels separate:

```text
hf_solver
hf_solver_version
hf_model_hash
hf_mesh_hash
hf_material_set_hash

peak_von_mises_pa
peak_eq_plastic_strain
plastic_work_j
peak_fastener_load_n
peak_bearing_load_n
residual_deformation_m
declared_failure_flag
```

Optional derived target:

```text
impact_damage_target
impact_damage_definition_version
```

## Quality flags

```text
FAST_SIM_VALID
HF_SOLVER_CONVERGED
HF_MODEL_ASSUMPTION_ONLY
DAMAGE_SURROGATE_OOD
REAL_ROBOT_VALIDATED
```

A sample with a failed solver, invalid healthy baseline, incomplete provenance
or an OOD surrogate flag must not be silently mixed into headline metrics.
