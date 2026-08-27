#!/usr/bin/env python3
"""Render a saved native-MuJoCo trace to an annotated MP4 without rerunning policy."""
from __future__ import annotations
import argparse
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "external/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/scene_g1.xml"

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--fps", type=int, default=15)
    a = p.parse_args()
    trace, model = np.load(a.trace), mujoco.MjModel.from_xml_path(str(MODEL))
    model.vis.global_.offwidth, model.vis.global_.offheight = 1920, 1080
    data, renderer = mujoco.MjData(model), mujoco.Renderer(model, height=1080, width=1920)
    writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=9, macro_block_size=1)
    try:
        for frame in range(int(np.ceil(len(trace['qpos']) * a.fps / 50.0))):
            i = min(round(frame * 50.0 / a.fps), len(trace['qpos']) - 1)
            data.qpos[:], data.qvel[:] = trace['qpos'][i], trace['qvel'][i]
            mujoco.mj_forward(model, data); renderer.update_scene(data)
            img = Image.fromarray(renderer.render()); draw = ImageDraw.Draw(img, "RGBA")
            draw.rectangle((24,24,900,96), fill=(0,0,0,180))
            draw.text((44,44), f"{a.label} | t={i/50:.2f}s", fill=(255,255,255,255))
            writer.append_data(np.asarray(img))
    finally:
        writer.close(); renderer.close(); trace.close()

if __name__ == "__main__": main()
