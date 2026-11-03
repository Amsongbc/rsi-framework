#!/usr/bin/env python3
"""Check real EWAM forward output without driving any simulator or robot."""
import json
from pathlib import Path

import numpy as np

from ewam_service import CAMERAS, EwamClient

HERE = Path(__file__).resolve().parent
client = EwamClient(HERE / "service")
health = client.call("health")
print({k: health[k] for k in ["status", "gpu", "strict_checkpoint", "startup_s"]}, flush=True)
obs = {"vision": {camera: {"color": np.full((480, 640, 3), 128, np.uint8)} for camera in CAMERAS},
       "state": {"left_arm_joint_state": np.zeros(6, np.float32),
                 "right_arm_joint_state": np.zeros(6, np.float32),
                 "left_ee_joint_state": np.ones(1, np.float32),
                 "right_ee_joint_state": np.ones(1, np.float32)},
       "instruction": "Keep the arms still."}
client.call("reset", session_id="synthetic_inference_smoke", seed=12345)
try:
    result = client.call("infer", session_id="synthetic_inference_smoke", obs=obs)
    actions = result["actions"]
    assert len(actions) == 48
    for action in actions:
        for key, dim in [("left_arm_joint_state", 6), ("right_arm_joint_state", 6),
                         ("left_ee_joint_state", 1), ("right_ee_joint_state", 1)]:
            value = np.asarray(action[key])
            assert value.shape == (dim,) and np.isfinite(value).all()
    record = {"real_model": True, "benchmark_episode": False, "actuated": False,
              "native_actions": len(actions), "inference_s": result["inference_s"],
              "all_actions_finite": True}
    (HERE / "ewam_inference_smoke.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record, flush=True)
finally:
    client.call("release", session_id="synthetic_inference_smoke")
