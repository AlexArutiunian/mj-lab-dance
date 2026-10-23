#!/usr/bin/env python3
"""Export the three committed G1 ONNX policies and reference motions for browser use."""
import argparse
import gzip
import json
import shutil
from pathlib import Path
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
MIMIC = REPO / "dance_sim/assets/policies/mimic"
PRESETS = {
    "short": ("dance1_subject2_16s_faststart", "dance1_subject2_16s_faststart.npz"),
    "full": ("dance1_subject2", "dance1_subject2.npz"),
    "clip10": ("dance1_subject2_1mx1_10s", "dance1_subject2_1mx1_10s.npz"),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-source", type=Path, required=True,
                   help="Existing licensed web model directory containing manifest.json")
    p.add_argument("--walking-source", type=Path, required=True,
                   help="Official unitree_rl_mjlab checkout containing the G1 velocity policy")
    p.add_argument("--output", type=Path, default=REPO / "browser/public")
    args = p.parse_args()
    source = args.model_source.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["model"] != "g1_29dof":
        raise ValueError("Expected official 29-DoF G1 model")
    dest = args.output / "model"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)
    for label, (name, motion_name) in PRESETS.items():
        folder = MIMIC / name
        target = args.output / "policies" / label
        target.mkdir(parents=True, exist_ok=True)
        for asset in ("policy.onnx", "policy.onnx.data"):
            shutil.copy2(folder / "exported" / asset, target / asset)
        config = yaml.safe_load((folder / "params/deploy.yaml").read_text())
        if len(config["default_joint_pos"]) != 29 or abs(config["step_dt"] - 0.02) > 1e-12:
            raise ValueError("Incorrect deploy config " + name)
        (target / "deploy.json").write_text(json.dumps(config, separators=(",", ":")))
        with np.load(folder / "params" / motion_name) as n:
            q = np.asarray(n["joint_pos"], dtype=np.float32)
            dq = np.asarray(n["joint_vel"], dtype=np.float32)
            quat = np.asarray(n["body_quat_w"], dtype=np.float32)
            if q.ndim != 2 or q.shape[1] != 29 or dq.shape != q.shape:
                raise ValueError("Invalid joint reference arrays")
            if quat.ndim != 3 or quat.shape[0] != q.shape[0] or quat.shape[2] != 4:
                raise ValueError("Invalid body quaternion reference array")
            payload = dict(joint_pos=q.tolist(), joint_vel=dq.tolist(),
                           root_quat=quat[:, 0, :].tolist(), count=len(q))
        with gzip.open(target / "motion.json.gz", "wt", encoding="utf-8", compresslevel=8) as f:
            json.dump(payload, f, separators=(",", ":"))
        print(label, "frames", len(q), "ONNX", (target / "policy.onnx.data").stat().st_size)
    walking = args.walking_source.resolve() / "deploy/robots/g1/config/policy/velocity/v0"
    walk_target = args.output / "policies" / "walk"
    walk_target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(walking / "exported/policy.onnx", walk_target / "policy.onnx")
    walk_config = yaml.safe_load((walking / "params/deploy.yaml").read_text())
    if (len(walk_config["default_joint_pos"]) != 29
            or abs(walk_config["step_dt"] - 0.02) > 1e-12
            or len(walk_config["actions"]["JointPositionAction"]["scale"]) != 29):
        raise ValueError("Unexpected official Unitree G1 velocity deploy config")
    (walk_target / "deploy.json").write_text(json.dumps(walk_config, separators=(",", ":")))
    print("walk", "frames", int(60 / walk_config["step_dt"]),
          "ONNX", (walk_target / "policy.onnx").stat().st_size)
    print("Prepared browser assets in", args.output)

if __name__ == "__main__":
    main()
