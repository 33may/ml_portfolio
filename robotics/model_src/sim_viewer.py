#!/usr/bin/env python
"""Universal RoboTurk demo viewer (Sawyer, *pegs-full*) that survives **all**
robosuite forks.

Highlights
==========
* **Controller-safe** – registers `JOINT_VELOCITY` regardless of registry name.
* **Bug-free** – patches read-only `torque_compensation` seen in forks.
* **Schema-free** – finds arm block, no hard-coded `body_parts → arms`.
* **Lean** – <200 logical lines, zero external helpers.

```bash
export MUJOCO_GL=glfw
python sim_viewer.py /ABS/PATH/TO/pegs-full --fps 20
```
"""
from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path

import h5py
import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config

# ---------------------------------------------------------------------------
# 0. Controller shim + hot-patch for torque_compensation
# ---------------------------------------------------------------------------
try:
    from robosuite.controllers.parts.generic.joint_vel import JointVelocityController  # type: ignore
except ModuleNotFoundError as err:
    raise ImportError("JointVelocityController missing — update robosuite.") from err

# Fix read-only property
if isinstance(getattr(JointVelocityController, "torque_compensation", None), property):
    prop = JointVelocityController.torque_compensation  # type: ignore[attr-defined]
    if prop.fset is None:
        def _set_tc(self, val):  # noqa: D401
            self.__dict__["_torque_compensation"] = bool(val)
        JointVelocityController.torque_compensation = prop.setter(_set_tc)  # type: ignore[assignment]

# Register JV controller in whichever registry exists
import robosuite.controllers as _rc  # noqa: E402
for _attr in dir(_rc):
    reg = getattr(_rc, _attr)
    if isinstance(reg, dict) and "JOINT_POSITION" in reg:
        reg.setdefault("JOINT_VELOCITY", JointVelocityController)
        break

# ---------------------------------------------------------------------------
# 1. Build velocity composite controller
# ---------------------------------------------------------------------------

def _find_arm_block(cfg: dict) -> dict:
    if isinstance(cfg.get("body_parts"), dict) and isinstance(cfg["body_parts"].get("arms"), dict):
        return cfg["body_parts"]["arms"]
    stack = [cfg]
    while stack:
        node = stack.pop()
        if isinstance(node, dict) and any(k in node for k in ("right", "left", "right_arm", "left_arm")):
            return node
        if isinstance(node, dict):
            stack.extend(v for v in node.values() if isinstance(v, dict))
    raise KeyError("No arm controller block found in BASIC config")


def build_vel_controller() -> dict:
    cfg = load_composite_controller_config(controller="BASIC")
    arms = _find_arm_block(cfg)
    for acfg in arms.values():
        if isinstance(acfg, dict):
            acfg.update(type="JOINT_VELOCITY",
                        input_max=4, input_min=-4,
                        output_max=4, output_min=-4)
    return cfg

# ---------------------------------------------------------------------------
# 2. Env factory (keyword-safe across versions)
# ---------------------------------------------------------------------------

def make_env(task: str, ctrl_cfg: dict, xml_path: str | None, fps: int):
    kwargs = dict(env_name=task,
                  robots="Sawyer",
                  controller_configs=ctrl_cfg,
                  has_renderer=True,
                  has_offscreen_renderer=False,
                  control_freq=fps,
                  use_camera_obs=False,
                  ignore_done=True)
    if xml_path and "mujoco_model_path" in inspect.signature(suite.make).parameters:
        kwargs["mujoco_model_path"] = xml_path
    env = suite.make(**kwargs)
    env.reset(); env.viewer.set_camera(camera_id=0)
    return env

# ---------------------------------------------------------------------------
# 3. Helper
# ---------------------------------------------------------------------------

def grip_bin_to_pm1(val: float) -> float:
    return 2.0 * val - 1.0

# ---------------------------------------------------------------------------
# 4. Replay
# ---------------------------------------------------------------------------

def replay(dataset_dir: Path, fps: int):
    h5_path, models_dir = dataset_dir / "demo.hdf5", dataset_dir / "models"
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    ctrl_cfg = build_vel_controller()
    env, current_xml = None, None

    with h5py.File(h5_path, "r") as h5:
        demos = h5["data"]
        task = demos.attrs["env"].replace("Sawyer", "")
        print(f"Loaded {len(demos)} demos – task: {task}")

        for name, demo in demos.items():
            xml_file = demo.attrs.get("model_file")
            xml_path = str(models_dir / xml_file) if xml_file else None

            # (Re)build environment if XML changes
            if env is None or xml_file != current_xml:
                if env:
                    env.close()
                env = make_env(task, ctrl_cfg, xml_path, fps)
                current_xml = xml_file
                env.robots[0].print_action_info()

            dt = 1.0 / env.control_freq
            for dq, g in zip(demo["joint_velocities"], demo["gripper_actuations"]):
                action = np.concatenate([dq, [grip_bin_to_pm1(float(g))]], dtype=np.float32)

                tic = time.perf_counter()
                env.step(action)
                env.render()
                lag = time.perf_counter() - tic
                if lag < dt:
                    time.sleep(dt - lag)
    if env:
        env.close()


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path, help="Folder with demo.hdf5 + models/")
    ap.add_argument("--fps", type=int, default=20)
    opts = ap.parse_args()
    replay(opts.dataset.expanduser().resolve(), fps=opts.fps)