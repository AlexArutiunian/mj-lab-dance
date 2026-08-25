# dance_sim

Локальная песочница для экспериментов с танцевальными motion/policy без заливки на робота.
Эта копия лежит внутри `WearBench_G1_MuJoCo_MVP_bundle` и использует
относительные пути, чтобы запускаться из bundle.

## Что здесь есть

- `assets/policies/mimic/` - скопированные с deploy full/10s/16s motion и ONNX policy.
- `scripts/motion_info.py` - статистика `.npz` motion.
- `scripts/crop_motion.py` - обрезка `.npz` по кадрам или секундам.
- `scripts/compare_motions.py` - численное сравнение двух motion.
- `scripts/find_subsequence.py` - поиск, где короткая версия лучше всего совпадает с full dance по joint trajectory.
- `experiments/` - сюда складывать новые обрезки.

## Быстрые команды

```bash
cd /home/al/humanoid/WearBench_G1_MuJoCo_MVP_bundle/dance_sim
python3 scripts/motion_info.py assets/policies/mimic
```

Headless smoke test ONNX/MuJoCo:

```bash
cd /home/al/humanoid/WearBench_G1_MuJoCo_MVP_bundle/dance_sim
./run_onnx_headless_check.sh 50 cpu
```

Обрезать full dance:

```bash
python3 scripts/crop_motion.py \
  --input assets/policies/mimic/dance1_subject2/params/dance1_subject2.npz \
  --output experiments/full_clip_20p50_36p74.npz \
  --start-sec 20.50 \
  --end-sec 36.74 \
  --recompute-joint-vel
```

Найти, где текущая 16.24s версия похожа на full dance:

```bash
python3 scripts/find_subsequence.py \
  --long assets/policies/mimic/dance1_subject2/params/dance1_subject2.npz \
  --short assets/policies/mimic/dance1_subject2_16s_faststart/params/dance1_subject2_16s_faststart.npz \
  --window 120
```

## Про RL/MuJoCo viewer

Для `scripts/play_onnx_mjlab.py` используется локальная копия
`external/unitree_rl_mjlab`. Wrapper-скрипты вычисляют путь к `dance_sim`
от своего расположения и не завязаны на отдельный `/home/al/humanoid/dance_sim`.
