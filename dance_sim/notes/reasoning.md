# Reasoning: cropped full dance

Current deploy has three mimic folders:

- `dance1_subject2` - full, 6574 frames, 131.48s.
- `dance1_subject2_1mx1_10s` - 500 frames, 10.00s.
- `dance1_subject2_16s_faststart` - 812 frames, 16.24s.

The 16.24s version is a direct crop of full dance:

- best match starts at full frame `1025`
- at 50 FPS this is `20.50s`
- clip range is `1025:1837`, i.e. `20.50s .. 36.74s`
- joint/body differences against a direct recut are numerical noise only

All three mimic variants use the same ONNX policy files:

- same `policy.onnx`
- same `policy.onnx.data`

So a new cropped full-dance variant is not conceptually hard:

1. Pick a stable time window from full dance.
2. Cut the `.npz` motion.
3. Reuse the same policy folder/files.
4. Set `time_end = frames / fps` in deploy config.
5. Test first in geometry replay and then in RL/MuJoCo, before deploying.

Risk is not the crop operation itself. Risk is choosing a window that starts from a physically bad state for the real robot:

- too low root height
- high root velocity
- aggressive joint jumps
- bad foot contact / mid-air start
- starting in a pose the policy cannot enter from Velocity/FixStand safely

For deploy, the first frame matters more than average quality. A clip can be good inside but unsafe if the first 0.5-1.0s is abrupt.
