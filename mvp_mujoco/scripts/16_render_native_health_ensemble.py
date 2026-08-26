#!/usr/bin/env python3
"""Render saved native-MuJoCo health-ensemble traces as HD evidence videos."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DANCE_SIM = ROOT.parent / "dance_sim"
DEFAULT_INPUT = ROOT / "outputs/rb_y_16s/native_health_ensemble_100"
DEFAULT_MODEL = DANCE_SIM / "external/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/scene_g1.xml"


def _parse_checkpoints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_rows(input_dir: Path, checkpoint: int) -> list[dict[str, str]]:
    with (input_dir / f"checkpoint_{checkpoint}" / "trials.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def _label_tile(frame: np.ndarray, row: dict[str, str], checkpoint: int) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    failed = row["failed"] == "1"
    color = (218, 66, 66, 255) if failed else (40, 182, 105, 255)
    width, height = image.size
    draw.rectangle((0, 0, width - 1, height - 1), outline=color, width=4)
    draw.rectangle((4, 4, min(width - 4, 178), 51), fill=(0, 0, 0, 175))
    status = "FALL" if failed else "OK"
    draw.text((10, 9), f"#{int(row['sample']):03d}  {status}", fill=color)
    draw.text(
        (10, 28),
        f"h={float(row['weakest_health']):.2f}  s={float(row['weakest_torque_scale']):.2f}",
        fill=(255, 255, 255, 255),
    )
    if failed:
        draw.text((10, 45), f"fall {float(row['failure_t']):.1f}s", fill=(255, 220, 220, 255))
    return np.asarray(image).copy()


def _title_frame(width: int, height: int, checkpoint: int, rows: list[dict[str, str]]) -> np.ndarray:
    image = Image.new("RGB", (width, height), (13, 22, 31))
    draw = ImageDraw.Draw(image)
    falls = sum(row["failed"] == "1" for row in rows)
    draw.text((32, 22), f"Native MuJoCo health ensemble | {checkpoint:,} completed dances", fill=(240, 245, 250))
    draw.text(
        (32, 48),
        f"100 virtual robots | completed {len(rows) - falls}/100 | floor-level falls {falls}/100 | green=completed red=fall",
        fill=(180, 198, 214),
    )
    return np.asarray(image).copy()


def _render_state(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> np.ndarray:
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    return renderer.render()


def _render_collage(
    model: mujoco.MjModel,
    input_dir: Path,
    output_dir: Path,
    checkpoint: int,
    fps: int,
    tile_width: int,
    tile_height: int,
) -> Path:
    rows = _load_rows(input_dir, checkpoint)
    if len(rows) != 100:
        raise ValueError(f"checkpoint {checkpoint} has {len(rows)} rows, expected 100")
    rows.sort(key=lambda row: int(row["sample"]))
    traces = [np.load(input_dir / f"checkpoint_{checkpoint}" / f"trial_{int(row['sample']):03d}.npz") for row in rows]
    columns = 10
    title_height = 88
    width = tile_width * columns
    height = title_height + tile_height * math.ceil(len(rows) / columns)
    frame_count = int(math.ceil(16.24 * fps))
    output = output_dir / f"ensemble_{checkpoint:07d}_100robots_4k.mp4"
    renderer = mujoco.Renderer(model, height=tile_height, width=tile_width)
    data = mujoco.MjData(model)
    writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=9, macro_block_size=1)
    try:
        for video_frame in range(frame_count):
            control_frame = min(int(round(video_frame * 50.0 / fps)), len(traces[0]["qpos"]) - 1)
            canvas = _title_frame(width, height, checkpoint, rows)
            for index, (row, trace) in enumerate(zip(rows, traces, strict=True)):
                tile = _render_state(renderer, model, data, trace["qpos"][control_frame], trace["qvel"][control_frame])
                tile = _label_tile(tile, row, checkpoint)
                x = index % columns * tile_width
                y = title_height + index // columns * tile_height
                canvas[y : y + tile_height, x : x + tile_width] = tile
            writer.append_data(canvas)
            if video_frame % fps == 0 or video_frame + 1 == frame_count:
                print(f"[RENDER] checkpoint={checkpoint} frame={video_frame + 1}/{frame_count}", flush=True)
    finally:
        writer.close()
        renderer.close()
        for trace in traces:
            trace.close()
    return output


def _representatives(input_dir: Path) -> list[tuple[int, dict[str, str], str]]:
    selected: list[tuple[int, dict[str, str], str]] = []
    healthy = _load_rows(input_dir, 0)
    selected.append((0, healthy[0], "healthy_control"))
    rows_500k = _load_rows(input_dir, 500000)
    selected.append((500000, next(row for row in rows_500k if row["failed"] == "1"), "500k_floor_fall"))
    rows_1m = _load_rows(input_dir, 1000000)
    completed = [row for row in rows_1m if row["failed"] == "0"]
    failed = [row for row in rows_1m if row["failed"] == "1"]
    selected.append((1000000, min(completed, key=lambda row: abs(float(row["wear_rate_multiplier"]) - 1.0)), "1m_nominal_success"))
    selected.append((1000000, min(failed, key=lambda row: abs(float(row["failure_t"]) - 8.12)), "1m_mid_clip_fall"))
    return selected


def _render_single(model: mujoco.MjModel, input_dir: Path, output_dir: Path, checkpoint: int, row: dict[str, str], label: str) -> Path:
    trace = np.load(input_dir / f"checkpoint_{checkpoint}" / f"trial_{int(row['sample']):03d}.npz")
    width, height, fps = 1920, 1080, 30
    renderer = mujoco.Renderer(model, height=height, width=width)
    data = mujoco.MjData(model)
    output = output_dir / f"{label}.mp4"
    writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=9, macro_block_size=1)
    try:
        for video_frame in range(int(math.ceil(16.24 * fps))):
            control_frame = min(int(round(video_frame * 50.0 / fps)), len(trace["qpos"]) - 1)
            frame = _render_state(renderer, model, data, trace["qpos"][control_frame], trace["qvel"][control_frame])
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image, "RGBA")
            failed = row["failed"] == "1"
            color = (230, 72, 72, 255) if failed else (44, 192, 112, 255)
            draw.rectangle((24, 24, 680, 182), fill=(0, 0, 0, 175))
            draw.text((44, 44), f"{label} | virtual robot #{int(row['sample']):03d}", fill=color)
            draw.text((44, 75), f"checkpoint: {checkpoint:,} dances | health: {float(row['weakest_health']):.3f}", fill=(245, 245, 245, 255))
            draw.text((44, 106), f"weakest torque scale: {float(row['weakest_torque_scale']):.3f}x | t={video_frame / fps:.2f}s", fill=(245, 245, 245, 255))
            status = f"FLOOR FALL at {float(row['failure_t']):.2f}s" if failed else "COMPLETED"
            draw.text((44, 137), status, fill=color)
            writer.append_data(np.asarray(image))
    finally:
        writer.close()
        renderer.close()
        trace.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_INPUT / "videos_20260826")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--checkpoints", default="0,100000,500000,1000000")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--tile-width", type=int, default=384)
    parser.add_argument("--tile-height", type=int, default=216)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    for checkpoint in _parse_checkpoints(args.checkpoints):
        output = _render_collage(model, args.input_dir, args.out_dir, checkpoint, args.fps, args.tile_width, args.tile_height)
        print(f"[RENDER DONE] {output}", flush=True)
    for checkpoint, row, label in _representatives(args.input_dir):
        output = _render_single(model, args.input_dir, args.out_dir, checkpoint, row, label)
        print(f"[RENDER DONE] {output}", flush=True)


if __name__ == "__main__":
    main()
