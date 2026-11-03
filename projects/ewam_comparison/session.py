#!/usr/bin/env python3
"""Paired EWAM baseline or current-conversation Sol + EWAM + PyRUA episode.

The interactive planner receives only RGB images, the instruction and robot state.
The evaluator's verdict is saved separately after the episode has ended.
"""
from __future__ import annotations

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os

import argparse
import contextlib
import dataclasses
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from ewam_service import CAMERAS, EwamClient, public_observation

HERE = Path(__file__).resolve().parent
PROJECTS = Path(_export_os.environ.get("PROJECTS_ROOT", str(HERE.parent)))
def plain(value):
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


class CountingEnv:
    """A single native action budget shared by EWAM and every analytic primitive."""
    def __init__(self, env):
        self.env = env
        self.issued_actions = 0
        self.owner = "primitive"
        self.actions_by_owner = {"primitive": 0, "ewam": 0}

    def get_obs(self):
        return self.env.get_obs()

    def get_status(self):
        return self.env.get_status()

    def step(self, action):
        status = self.get_status()
        limit = int(status["step_limit"])
        if max(int(status.get("step", 0)), self.issued_actions) >= limit:
            from pyrualean._backend import EpisodeFinished
            raise EpisodeFinished("Shared native action budget exhausted")
        self.issued_actions += 1
        self.actions_by_owner[self.owner] += 1
        return self.env.step(action)

    def episode_over(self):
        return self.env.episode_over()

    def conclude(self):
        return self.env.conclude()


def execute_ewam(env, client, session_id, *, instruction=None, chunks=1,
                 model_actions=16, on_action=None):
    if not isinstance(chunks, int) or not 1 <= chunks <= 4:
        raise ValueError("chunks must be between 1 and 4")
    if not isinstance(model_actions, int) or not 1 <= model_actions <= 16:
        raise ValueError("model_actions must be between 1 and 16")
    reports = []
    for _ in range(chunks):
        if env.episode_over() or env.issued_actions >= int(env.get_status()["step_limit"]):
            break
        obs = public_observation(env.get_obs(), instruction)
        result = client.call("infer", session_id=session_id, obs=obs)
        report = {k: v for k, v in result.items() if k != "actions"}
        report.update(planned_native_actions=len(result["actions"]), executed_native_actions=0)
        for action in result["actions"][:model_actions * 3]:
            if env.episode_over() or env.issued_actions >= int(env.get_status()["step_limit"]):
                break
            _, _, done, _ = env.step(action)
            report["executed_native_actions"] += 1
            if on_action:
                on_action(action)
            if done:
                break
        reports.append(report)
    return reports


class Session:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / "commands").mkdir()
        (self.out / "responses").mkdir()
        (self.out / "workspace").mkdir()
        self.identifier = self.out.name
        self.client = EwamClient(args.service)
        self.started = time.monotonic()
        self.events = []
        self.ewam_calls = 0
        self.cells = 0
        self.observations = 0
        self.errors = []
        self.daemons = []
        self.toolkit = None
        self.bridge = None
        self.phase = "booting"
        self.stop_reason = None
        self.ready_at = None

    def boot(self):
        from pyrualean.hosts import xpolicylab_robodojo as host
        from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot
        from pyrualean.sandbox import make_builtins
        from pyrualean.hosts.sam3_local import LocalSam3

        original_layout = host.lay_out

        def layout(*a, **kw):
            root = original_layout(*a, **kw)
            runtime_assets = HERE / "runtime_assets/Assets"
            if not runtime_assets.is_dir():
                raise RuntimeError("Run prepare_runtime_assets.py before launching experiments")
            (root / "Assets").unlink()
            (root / "Assets").symlink_to(runtime_assets, target_is_directory=True)
            script = root / "scripts/eval_policy.sh"
            old = script.read_text()
            new = old.replace('KIT_ARGS=""',
                              f'KIT_ARGS=" --/renderer/activeGpu={self.args.sim_gpu}'
                              ' --/renderer/multiGpu/enabled=false"')
            if new == old:
                raise RuntimeError("Could not set simulator GPU and disable multi-GPU rendering")
            script.write_text(new)
            write_json(self.out / "client_launcher_change.json", {
                "scope": "Only the run-local shell launcher; no task/reward/observation changes",
                "activeGpu": self.args.sim_gpu, "multiGpu": False})
            return root

        host.lay_out = layout
        cell = host.Cell(suite="arx_x5", task=self.args.task, seed=self.args.layout,
                         cuda_device=self.args.sim_gpu)
        # host.boot owns the normal bridge/server/client lifecycle; env_only skips SAM3
        # so the identical simulator observation path is used by both experiment groups.
        self.toolkit, self.daemons = host.boot(
            cell, self.out, env_only=True, depth=False,
            source_root=str(PROJECTS / "RoboDojo"), eval_seed=self.args.eval_seed,
            boot_timeout_s=900)
        self.bridge = self.toolkit._primitives.env.bridge
        self.env = CountingEnv(self.toolkit._primitives.env)
        self.toolkit._primitives.env = self.env
        if self.args.mode == "hybrid":
            self.toolkit._primitives.sam3_client = LocalSam3()
        self.robo = RobodojoOfficialColourRobot(host.make_backend(self.toolkit), primitives="no-vla")
        self.robo.ewam = self.ewam
        self.namespace = {"__builtins__": make_builtins(self.out / "workspace"),
                          "robo": self.robo, "np": np}
        if self.args.mode != "smoke":
            self.client.call("reset", session_id=self.identifier,
                             seed=100000 + self.args.eval_seed * 1000 + self.args.layout)
        write_json(self.out / "run_config.json", {
            "mode": self.args.mode, "task": self.args.task,
            "eval_seed": self.args.eval_seed, "layout": self.args.layout,
            "planner": "current conversation" if self.args.mode == "hybrid" else None,
            "requested_planner_model": "gpt-6.1-sol" if self.args.mode == "hybrid" else None,
            "planner_api_usage": None, "primitive_set": "PyRUA no-vla + EWAM",
            "EWAM_seed": 100000 + self.args.eval_seed * 1000 + self.args.layout,
            "observation_protocol": "Same PyRUA bridge in both groups; observations after native actions; EWAM consumes chunk-start RGB/state only",
            "step_limit": self.env.get_status()["step_limit"],
            "sim_gpu": self.args.sim_gpu, "wall_timeout_s": self.args.timeout})
        self.phase = "awaiting_planner" if self.args.mode == "hybrid" else "baseline_running"
        self.snapshot()
        self.ready_at = time.monotonic()

    def append(self, event):
        event = plain({"time": time.time(), **event})
        self.events.append(event)
        with (self.out / "events.jsonl").open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def record_action(self, action):
        self.append({"kind": "ewam_action", "issued_index": self.env.issued_actions,
                     "action": action})

    def ewam(self, instruction=None, chunks=1, model_actions=16):
        # Pure EWAM always sees the unmodified benchmark instruction and full chunks.
        if self.args.mode == "baseline" and (instruction is not None or model_actions != 16):
            raise ValueError("Baseline instruction/chunk length must remain unchanged")
        tick = time.monotonic()
        previous_owner, self.env.owner = self.env.owner, "ewam"
        try:
            reports = execute_ewam(self.env, self.client, self.identifier, instruction=instruction,
                                   chunks=chunks, model_actions=model_actions,
                                   on_action=self.record_action)
        finally:
            self.env.owner = previous_owner
        self.ewam_calls += len(reports)
        obs = self.env.get_obs()
        self.toolkit._primitives._last_obs = obs
        self.toolkit.state.record(obs, self.env.get_status())
        self.robo._last_state = None
        self.append({"kind": "ewam", "instruction_override": instruction,
                     "model_actions": model_actions, "chunks": reports,
                     "wall_s": time.monotonic() - tick})
        return reports

    def snapshot(self):
        from PIL import Image
        folder = self.out / "observations" / f"{self.observations:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        obs = self.env.get_obs()
        if self.env.episode_over():
            self.phase = "episode_ended"
        paths = {}
        for camera in CAMERAS:
            path = folder / f"{camera}.png"
            Image.fromarray(np.asarray(obs["vision"][camera]["color"])[..., :3].astype(np.uint8)).save(path)
            paths[camera] = str(path)
        value = {"phase": self.phase, "instruction": str(obs.get("instruction") or ""),
                 "robot_state": plain(self.robo.state()),
                 "step": int(self.env.get_status().get("step", 0)),
                 "issued_actions": self.env.issued_actions,
                 "step_limit": int(self.env.get_status()["step_limit"]),
                 "images": paths, "observation_index": self.observations}
        self.observations += 1
        write_json(folder / "observation.json", value)
        write_json(self.out / "latest.json", value)
        return value

    def python(self, code):
        from pyrualean.sandbox import audit_source, blocked_names
        from pyrualean._backend import EpisodeFinished
        if self.args.mode != "hybrid":
            raise ValueError("Primitive/code interventions are disabled for baseline")
        if self.cells >= 100:
            raise RuntimeError("100-cell planner budget exhausted")
        flags = audit_source(code, workspace=self.out / "workspace")
        blocked = blocked_names(code)
        if flags or blocked:
            raise ValueError(f"Refused policy audit flags={flags}, blocked={blocked}")
        self.cells += 1
        (self.out / "workspace" / f"cell_{self.cells:03d}.py").write_text(code)
        stdout, stderr = io.StringIO(), io.StringIO()
        error = None
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exec(compile(code, f"cell_{self.cells:03d}.py", "exec"), self.namespace)
            except EpisodeFinished as exc:
                self.stop_reason = exc.reason
            except Exception:
                error = traceback.format_exc()
        result = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue(), "error": error}
        self.append({"kind": "python", "index": self.cells, "code": code, **result})
        return result

    def loop(self):
        if self.args.mode == "smoke":
            self.stop_reason = "connectivity_smoke_no_actions"
            return
        while not self.toolkit.episode_over():
            if self.env.issued_actions >= int(self.env.get_status()["step_limit"]):
                self.stop_reason = "native_budget"
                break
            if time.monotonic() - self.started > self.args.timeout:
                self.stop_reason = "wall_timeout"
                break
            if not all(getattr(d, "alive", lambda: True)() for d in self.daemons):
                raise RuntimeError("Evaluation client exited before a verdict")
            if self.args.mode == "baseline":
                self.ewam()
                self.snapshot()
                continue
            commands = sorted((self.out / "commands").glob("*.json"), key=lambda f: f.stat().st_mtime_ns)
            if not commands:
                time.sleep(0.2)
                continue
            path = commands[0]
            command = json.loads(path.read_text())
            identifier = command["id"]
            destination = self.out / "commands" / (identifier + ".accepted")
            os.replace(path, destination)
            try:
                op = command["op"]
                self.append({"kind": "planner_command", **command})
                if op == "finish":
                    self.stop_reason = "planner_finish"
                    write_json(self.out / "responses" / (identifier + ".json"), {"finishing": True})
                    break
                if op == "ewam":
                    output = self.ewam(command.get("instruction"), int(command.get("chunks", 1)),
                                       int(command.get("model_actions", 16)))
                elif op == "python":
                    output = self.python(command["code"])
                elif op == "observe":
                    output = None
                else:
                    raise ValueError(f"Unknown command {op}")
                response = {"output": output, "observation": self.snapshot()}
            except Exception:
                response = {"error": traceback.format_exc()}
                self.append({"kind": "command_error", **response})
            write_json(self.out / "responses" / (identifier + ".json"), response)

    def run(self):
        from pyrualean._backend import EpisodeFinished
        try:
            self.boot()
            self.loop()
        except EpisodeFinished as exc:
            self.stop_reason = exc.reason
        except Exception:
            self.errors.append(traceback.format_exc())
            self.stop_reason = "infrastructure_error"
            print(self.errors[-1], flush=True)
        finally:
            policy_end = time.monotonic()
            verdict = None
            if self.toolkit is not None:
                self.toolkit.conclude()
                verdict = self.bridge.verdict(timeout_s=60)
            from pyrualean.hosts.common import stop_all
            stop_all(self.toolkit, self.daemons)
            if self.args.mode != "smoke":
                with contextlib.suppress(Exception):
                    self.client.call("release", session_id=self.identifier)
            client_results = []
            for path in (self.out / "bench/eval_result").glob("**/_result.json"):
                client_results.append({"path": str(path), "value": json.loads(path.read_text())})
            valid = bool(verdict is not None and not self.errors and client_results)
            connectivity_ok = verdict is not None and not self.errors
            if self.args.mode == "smoke":
                valid = False
            detail = None
            if valid:
                matches = [entry for row in client_results
                           for entry in row["value"].get("details", {}).values()
                           if entry.get("layout_id") == self.args.layout]
                valid = len(matches) == 1 and all(row["value"].get("eval_time") == 1 for row in client_results)
                detail = matches[0] if len(matches) == 1 else None
                valid = valid and bool(detail.get("success")) == bool(verdict.get("success"))
            ledger = getattr(getattr(self, "robo", None), "_ledger", None)
            if ledger is not None:
                ledger.write_jsonl(self.out / "primitive_calls.jsonl")
            result = {"mode": self.args.mode, "task": self.args.task,
                      "benchmark_episode": self.args.mode != "smoke",
                      "connectivity_ok": connectivity_ok,
                      "eval_seed": self.args.eval_seed, "layout": self.args.layout,
                      "valid": valid, "success": bool((verdict or {}).get("success")) if valid else None,
                      "client_verdict": verdict, "client_results": client_results,
                      "client_score": None if detail is None else detail.get("score"),
                      "ewam_calls": self.ewam_calls, "python_cells": self.cells,
                      "planner_commands": sum(e["kind"] == "planner_command" for e in self.events),
                      "primitive_calls": 0 if ledger is None else ledger.stateful_calls(),
                      "native_actions_by_owner": getattr(getattr(self, "env", None), "actions_by_owner", {}),
                      "native_actions_issued": getattr(getattr(self, "env", None), "issued_actions", 0),
                      "startup_s": None if self.ready_at is None else self.ready_at - self.started,
                      "control_wall_s": None if self.ready_at is None else policy_end - self.ready_at,
                      "wall_s": time.monotonic() - self.started,
                      "stop_reason": self.stop_reason, "errors": self.errors,
                      "planner_usage": None}
            write_json(self.out / "result.json", result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "hybrid", "smoke"], required=True)
    parser.add_argument("--task", choices=["push_T", "cover_blocks", "build_tower", "general_pickup"], required=True)
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--sim-gpu", type=int, default=2)
    parser.add_argument("--perception-gpu", type=int, default=1)
    parser.add_argument("--service", type=Path, default=HERE / "service")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=5400)
    args = parser.parse_args()
    if args.task == "general_pickup" and args.mode != "smoke":
        parser.error("general_pickup is for connectivity smoke only, outside the comparison")
    devices = [(args.sim_gpu, 16 * 1024)] + ([(args.perception_gpu, 10 * 1024)] if args.mode == "hybrid" else [])
    for gpu, required_free in devices:
        result = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True)
        if int(result.stdout.strip()) < required_free:
            raise RuntimeError(f"GPU{gpu} lacks the {required_free} MiB free memory required for this role")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.perception_gpu)
    os.environ["PATH"] = str(HERE / ".venv-sim/bin") + os.pathsep + os.environ["PATH"]
    os.environ["PYTHONPATH"] = str(PROJECTS / "PyRUA-Lean/src")
    os.environ["SAM3_CHECKPOINT_PATH"] = str(PROJECTS / "RoboDojo-resources/backbones/sam3/sam3.pt")
    temporary = HERE / "runtime_tmp"
    temporary.mkdir(exist_ok=True, mode=0o700)
    os.environ.update(OMNI_KIT_ACCEPT_EULA="YES", ROBODOJO_MAX_BASH_RETRIES="1",
                      NO_PROXY="*", no_proxy="*", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                      TRITON_CACHE_DIR=str(HERE / ".triton-cache"),
                      TORCH_EXTENSIONS_DIR=str(HERE / ".torch-extensions"),
                      VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json",
                      XDG_RUNTIME_DIR=str(HERE / "xdg"), TMPDIR=str(temporary),
                      XDG_CACHE_HOME=str(HERE / ".cache"),
                      CUDA_CACHE_PATH=str(HERE / ".cuda-cache"))
    for name in list(os.environ):
        if name.lower().endswith("_proxy") and name.lower() != "no_proxy":
            os.environ.pop(name)
        if name in {"ROBODOJO_ORACLE_STATE", "ROBODOJO_IK_FEEDBACK", "ROBODOJO_TASK_STEP_LIM",
                    "ROBODOJO_STEP_LIM", "ROBODOJO_FIXED_INSTRUCTION", "ROBODOJO_ORACLE_INSTRUCTION"}:
            os.environ.pop(name)
    sys.path.insert(0, str(PROJECTS / "PyRUA-Lean/src"))
    result = Session(args).run()
    good = result["connectivity_ok"] if args.mode == "smoke" else result["valid"]
    raise SystemExit(0 if good else 1)


if __name__ == "__main__":
    main()
