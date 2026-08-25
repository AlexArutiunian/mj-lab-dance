# Development note: one-dance accelerated-aging MVP

The first MuJoCo experiment intentionally uses an **uncalibrated accelerated degradation coefficient**. Its purpose is to validate the software/causal loop:

motion -> logged joint loading -> persistent damage -> reduced actuator capability -> degraded tracking/fall.

The resulting repetition number must not enter the RA-L manuscript as a physical lifetime estimate. It may later be reported only as a software sanity experiment or omitted entirely after calibrated results exist.

The final scientific model remains the multi-channel formulation already written in `main.tex` (fatigue, thermal aging, impact, wear) with explicit calibration requirements.
