"""Optional: sanity-check only the algebra without a robot/environment.
No generated values from this script are scientific results.
"""
import numpy as np
from wearbench.damage import alpha_for_target_torque_scale, damage_after_repetitions, health_from_damage, torque_scale_from_health

severity_norm = np.array([1.0, 0.8, 0.2])
alpha = alpha_for_target_torque_scale(50, 0.30, floor=0.10, exponent=1.5)
for n in [1, 10, 25, 50, 100]:
    d = damage_after_repetitions(severity_norm, n-1, alpha)
    h = health_from_damage(d)
    s = torque_scale_from_health(h, floor=0.10, exponent=1.5)
    print(n, "damage", np.round(d, 3), "health", np.round(h, 3), "torque_scale", np.round(s, 3))
