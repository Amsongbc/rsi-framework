#!/usr/bin/env python3
"""One real EWAM model shared by sequential baseline and interactive episodes."""
from __future__ import annotations

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os

import argparse
import json
import logging
from multiprocessing.connection import Client, Listener
import os
from pathlib import Path
import random
import secrets
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECTS = Path(_export_os.environ.get("PROJECTS_ROOT", str(HERE.parent)))
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
JOINT_KEYS = ("left_arm_joint_state", "left_ee_joint_state",
              "right_arm_joint_state", "right_ee_joint_state")


def public_observation(obs, instruction=None):
    """Only inputs used by the released RGB EWAM adapter; no evaluator data."""
    if instruction is None:
        value = obs.get("instruction")
        for key in ("instructions", "task_instruction"):
            if value is None:
                value = obs.get(key)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if isinstance(value, np.ndarray):
            value = value.item() if value.shape == () else (value.reshape(-1)[0] if value.size else None)
        if isinstance(value, (bytes, np.bytes_)):
            value = bytes(value).decode("utf-8")
        instruction = "" if value is None else str(value).strip()
        instruction = instruction or "follow the instruction"
    return {
        "vision": {name: {"color": np.asarray(obs["vision"][name]["color"]).copy()}
                   for name in CAMERAS},
        "state": {name: np.asarray(obs["state"][name]).copy() for name in JOINT_KEYS},
        "instruction": str(instruction),
    }


class EwamClient:
    def __init__(self, root):
        self.root = Path(root)

    def call(self, op, **data):
        with Client(str(self.root / "ewam.sock"), family="AF_UNIX",
                    authkey=(self.root / "authkey").read_bytes()) as connection:
            connection.send({"op": op, **data})
            if not connection.poll(1200):
                raise TimeoutError("EWAM service did not answer within 1200 seconds")
            response = connection.recv()
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["result"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE / "service")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--preallocate-gib", type=int, default=0,
                        help="Warm a CUDA allocator pool for this model's expected inference memory")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.root, 0o700)
    socket = args.root / "ewam.sock"
    if socket.exists():
        try:
            EwamClient(args.root).call("health")
        except (OSError, EOFError, ConnectionError):
            socket.unlink()
        else:
            raise RuntimeError("A live EWAM service already owns this directory")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["DS_ACCELERATOR"] = "cuda"
    os.environ["CUTLASS_PATH"] = str(HERE / "empty-cutlass")
    for name in list(os.environ):
        if name.lower().endswith("_proxy") or "CONDA" in name:
            os.environ.pop(name)
    os.environ.update(NO_PROXY="*", no_proxy="*", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      TRITON_CACHE_DIR=str(HERE / ".triton-cache"),
                      TORCH_EXTENSIONS_DIR=str(HERE / ".torch-extensions"))
    sys.path.insert(0, str(PROJECTS / "RoboDojo"))
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    from flash_attn import flash_attn_varlen_func  # noqa: F401 - fail before loading weights
    if args.preallocate_gib:
        free, _ = torch.cuda.mem_get_info()
        wanted = args.preallocate_gib * 1024**3
        if free < wanted + 5 * 1024**3:
            raise RuntimeError("Not enough free GPU memory for the requested model allocator pool")
        pool = torch.empty(wanted, dtype=torch.uint8, device="cuda")
        del pool  # caching allocator reuses this memory for weights and inference workspaces
    import yaml
    from XPolicyLab.policy.EWAM.model import Model, _activate_ewam_root

    policy_dir = PROJECTS / "RoboDojo/XPolicyLab/policy/EWAM"
    cfg = yaml.safe_load((policy_dir / "deploy.yml").read_text())
    cfg.update(bench_name="RoboDojo", env_cfg_type="arx_x5", action_type="joint",
               ckpt_name="robodojo", device="cuda", scene_prefix="",
               image_layout="robodojo", action_stride=3, replan_steps=None,
               num_inference_steps=10, clip_gripper=False, allow_dummy_policy=False)
    _activate_ewam_root(None)
    from models.ewam import Ewam
    original_load = Ewam.load_checkpoint

    def checked_load(self, path, strict=True):
        return original_load(self, path, strict=True)

    Ewam.load_checkpoint = checked_load
    logging.basicConfig(level=logging.INFO)
    started = time.monotonic()
    model = Model(cfg)
    metadata = {"status": "ready", "dummy": False, "strict_checkpoint": True,
                "gpu": args.gpu, "startup_s": time.monotonic() - started,
                "preallocated_gib": args.preallocate_gib,
                "config": cfg, "pid": os.getpid()}
    (args.root / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    key = args.root / "authkey"
    key.write_bytes(secrets.token_bytes(32))
    key.chmod(0o600)
    if socket.exists():
        raise RuntimeError("Existing EWAM socket: refuse to replace a potentially live service")
    active = None
    with Listener(str(socket), family="AF_UNIX", authkey=key.read_bytes()) as listener:
        print("EWAM service ready", flush=True)
        while True:
            with listener.accept() as connection:
                try:
                    message = connection.recv()
                    op = message["op"]
                    if op == "health":
                        result = metadata
                    elif op == "reset":
                        session_id = message["session_id"]
                        if active is not None and active != session_id:
                            raise RuntimeError("Another experiment episode owns this EWAM service")
                        active = session_id
                        seed = int(message["seed"])
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        model.reset()
                        result = {"seed": seed, "session_id": session_id}
                    elif op == "release":
                        if active != message["session_id"]:
                            raise RuntimeError("Wrong EWAM episode lease")
                        active = None
                        result = {}
                    elif op == "infer":
                        if active != message["session_id"]:
                            raise RuntimeError("Wrong EWAM episode lease")
                        obs = public_observation(message["obs"], message.get("instruction"))
                        tick = time.monotonic()
                        actions = model._infer(obs)
                        result = {"actions": actions, "inference_s": time.monotonic() - tick,
                                  "instruction": obs["instruction"]}
                    else:
                        raise ValueError(f"Unknown EWAM operation: {op}")
                    connection.send({"ok": True, "result": result})
                except Exception:
                    connection.send({"ok": False, "error": traceback.format_exc()})


if __name__ == "__main__":
    main()
