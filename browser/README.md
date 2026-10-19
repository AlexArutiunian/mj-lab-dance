# WearBench-G1 Browser Demo

Experimental browser runtime for the three exported Unitree G1 ONNX mimic policies.

The browser build is separate from the native MuJoCo research benchmarks.
It requires ONNX Runtime Web, the official MuJoCo WASM bindings and a compatible G1 scene.

## Policy presets

- short: 16.24 s mimic policy
- full: 131.48 s mimic policy
- clip10: 10 s mimic policy

Policy output alone is insufficient to predict falls. Closed-loop physics, reference motions,
PD actuator control and damage parameters are required. The browser variant must be
validated against the native implementation before its outcomes are reported as results.

## Scope

The intended controls are motion selection, completed repetitions, start/pause and reset.
The accumulated wear coordinate is uncalibrated; it is not the robot's physical life.

Source assets remain in the root project's dance_sim/assets/policies/mimic/ directory.
The journal manuscript is maintained on publication/ieee-ra-l.
