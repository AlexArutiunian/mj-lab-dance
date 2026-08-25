# WearBench-G1: experiments needed for the RA-L paper

## Minimum viable paper experiment set

1. **Scaling / throughput**
   - 1,024 / 4,096 / 8,192 / 16,384 G1 environments if memory permits.
   - Report simulation steps/s, real-time factor, GPU memory, training throughput.

2. **Motion-to-degradation benchmark**
   - Stand.
   - Walk slow / nominal / fast.
   - Run.
   - Turn.
   - Squat / deep squat.
   - Jump + landing.
   - Kick.
   - Stairs or step-up/down.
   - Payload carry.
   - Dance tracking.
   - Kungfu / high-dynamic whole-body tracking.
   - Output: 29-joint heat map for fatigue / thermal / impact / wear channels.

3. **Persistent lifetime schedules**
   - Locomotion-heavy.
   - Jump-heavy.
   - Squat/carry-heavy.
   - Dance/kungfu-heavy.
   - Mixed random workload.
   - Do NOT reset irreversible degradation between tasks.
   - Output: survival curves and first-joint-to-failure distributions.

4. **Degradation-aware RL**
   - Baseline: normal task reward.
   - HealthObs: health state in observation, no damage penalty.
   - Ours: health state + incremental degradation reward penalty.
   - Compare at matched task success/tracking quality.
   - Output: performance-vs-degradation Pareto plot.

5. **Ablations**
   - No fatigue channel.
   - No thermal-aging channel.
   - No impact channel.
   - No plant-evolution feedback.
   - Reset health every episode vs persistent health.

6. **RUL / calibration**
   - First phase: relative degradation only.
   - Fit G1 actuator loss / thermal model from real logs.
   - If possible, accelerated cycling of one spare actuator or representative drive.
   - Measure backlash, current/power drift, temperature, tracking error, friction/vibration.
   - Hold out load profiles for validation.

## Figures planned for the final paper

- Fig. 1: WearBench-G1 closed lifetime loop.
- Fig. 2: 3D G1 per-joint degradation heat map across skills.
- Fig. 3: degradation channels over one high-dynamic dance.
- Fig. 4: survival curves for workload schedules.
- Fig. 5: first-joint-to-failure probability by workload.
- Fig. 6: performance vs degradation Pareto frontier.
- Fig. 7 (if space): hardware calibration / held-out RUL prediction.

## Tables planned

- Related-work positioning table (already drafted).
- Simulation and randomization parameters.
- Skill-level degradation benchmark.
- Baseline vs proposed controller.
- Ablation study.

## Claims that must NOT be made before calibration

- “The real G1 fails after exactly N repetitions.”
- “One dance removes X hours of real actuator life.”
- “D=1 is a physical failure threshold” unless calibrated to the selected component/failure mode.

Safe before calibration:

- Relative modeled degradation between motions under the declared model.
- Which joints are most loaded by each motion.
- How degradation-aware control changes the modeled damage at matched task performance.
- Sensitivity of those rankings to parameter uncertainty.
