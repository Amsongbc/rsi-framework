#!/usr/bin/env python3
"""Render a synthetic cube with Isaac Sim; never counts as a benchmark episode."""

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["XDG_RUNTIME_DIR"] = str(HERE / "xdg")
Path(os.environ["XDG_RUNTIME_DIR"]).mkdir(exist_ok=True, mode=0o700)
os.chmod(os.environ["XDG_RUNTIME_DIR"], 0o700)
started = time.monotonic()
sys.argv.extend(["--/app/updateOrder/checkForHydraRenderComplete=1000",
                 "--/app/renderer/waitIdle=true", "--/app/hydraEngine/waitIdle=true"])

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "active_gpu": gpu, "physics_gpu": 0,
                     "multi_gpu": False, "renderer": "RayTracedLighting",
                     "width": 640, "height": 480})

import numpy as np
from PIL import Image
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.sensors.camera import Camera
from pxr import UsdLux
import omni.replicator.core as rep

world = World(stage_units_in_meters=1.0)
world.scene.add(FixedCuboid(prim_path="/World/Ground", name="ground",
                           position=np.array([0.0, 0.0, -0.05]),
                           scale=np.array([10.0, 10.0, 0.1]), color=np.array([0.6, 0.6, 0.6])))
UsdLux.DomeLight.Define(world.stage, "/World/Light").CreateIntensityAttr(1000.0)
world.scene.add(DynamicCuboid(prim_path="/World/Cube", name="cube",
                             position=np.array([0.0, 0.0, 0.5]),
                             scale=np.array([0.5, 0.5, 0.5]),
                             color=np.array([0.1, 0.8, 0.2])))
camera = Camera(prim_path="/World/Camera", position=np.array([0.0, 0.0, 3.0]),
                orientation=np.array([2**-0.5, 0.0, 2**-0.5, 0.0]),
                resolution=(320, 240), frequency=20)
world.reset()
camera.initialize()
for _ in range(90):
    world.step(render=True)
pixels = np.asarray(camera.get_rgba())
if pixels.size == 0:
    for _ in range(20):
        world.render()
        app.update()
    pixels = np.asarray(camera.get_rgba())
if pixels.size == 0:
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([camera.get_render_product_path()])
    for _ in range(4):
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False,
                              wait_for_render=False)
        app.update()
    pixels = np.asarray(annotator.get_data())
if pixels.size == 0 or pixels.ndim != 3 or pixels[..., :3].std() < 1:
    error = {"real_camera_render": False, "benchmark_episode": False,
             "shape": list(pixels.shape), "gpu": gpu,
             "rgb_mean": None if pixels.size == 0 else float(pixels[..., :3].mean()),
             "rgb_std": None if pixels.size == 0 else float(pixels[..., :3].std())}
    if pixels.size and pixels.ndim == 3:
        Image.fromarray(pixels.astype(np.uint8)).save(HERE / "sim_smoke_failed_frame.png")
    (HERE / "sim_smoke_failed.json").write_text(json.dumps(error, indent=2) + "\n")
    print(json.dumps(error), flush=True)
    os._exit(1)
Image.fromarray(pixels.astype(np.uint8)).save(HERE / "sim_smoke.png")
result = {"real_camera_render": True, "benchmark_episode": False,
          "shape": list(pixels.shape), "rgb_std": float(pixels[..., :3].std()),
          "gpu": gpu, "wall_s": time.monotonic() - started}
(HERE / "sim_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
# This is an isolated diagnostic worker. Kit's headless teardown has crashed;
# exit only this worker after the evidence files have been closed and flushed.
os._exit(0)
