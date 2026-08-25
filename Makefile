PYTHON := dance_sim/.venv/bin/python

.PHONY: doctor test smoke compile

doctor:
	./scripts/doctor.sh

test:
	PYTHONPATH=mvp_mujoco $(PYTHON) -m unittest discover -s tests -v

smoke:
	$(PYTHON) mvp_mujoco/SMOKE_TEST_WITHOUT_MUJOCO.py

compile:
	$(PYTHON) -m py_compile mvp_mujoco/scripts/*.py mvp_mujoco/wearbench/*.py dance_sim/scripts/*.py
