import random
import time

import h5py
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper

from robosuite.utils.input_utils import choose_controller


# ─── зависимости ──────────────────────────────────────────
import os, json, hashlib, collections, h5py, numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm                    # ✓ прогресс-бар
import robosuite as rs
from robosuite.environments.base import MujocoEnv
# ───────────────────────────────────────────────────────────

def xml_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# ==========================================================
#                     ENV-КЭШ
# ==========================================================
class EnvCache:
    def __init__(self, task_name: str, models_dir: str | None,
                 cam_name="agentview", img_size=128,
                 default=True, max_envs=None, render=False):

        self.task   = task_name
        self.models = models_dir
        self.cam    = cam_name
        self.size   = img_size
        self.render = render
        self.deflt  = default
        self.maxN   = max_envs or 32

        self._one : MujocoEnv | None = None
        self._lru : "collections.OrderedDict[str, MujocoEnv]" = collections.OrderedDict()
        self._has_xml = "mujoco_model_path" in rs.environments.base.make.__code__.co_varnames

    def _new_env(self, xml: str | None = None) -> MujocoEnv:
        kw = dict(
            env_name               = self.task,
            robots                 = "Sawyer",
            has_renderer           = self.render,
            has_offscreen_renderer = True,      # ← ВСЕГДА TRUE!
            use_camera_obs         = True,
            camera_names           = [self.cam],
            camera_heights         = self.size,
            camera_widths          = self.size,
        )
        if xml and self._has_xml:
            kw["mujoco_model_path"] = xml
        return rs.make(**kw)

    def get_env(self, xml_file: str | None = None) -> MujocoEnv:
        if self.deflt or not xml_file:
            if self._one is None:
                self._one = self._new_env()
            return self._one

        path = os.path.join(self.models, xml_file)
        key  = xml_md5(path)
        if key in self._lru:                       # hit
            self._lru.move_to_end(key)
            return self._lru[key]

        env = self._new_env(path)                  # miss → создать
        self._lru[key] = env; self._lru.move_to_end(key)
        if len(self._lru) > self.maxN:             # LRU-ограничение
            _, old = self._lru.popitem(last=False)
            old.close()
        return env

# ==========================================================
#                  DATASET + РЕНДЕР
# ==========================================================
class SawyerDataset(Dataset):
    def __init__(self, data_path, horizon_left=2, horizon_right=8,
                 image_size=128, camera_name="agentview",
                 img_batch=4096, limit_demo=None):

        self.data_path = data_path
        self.img_size   = image_size
        self.camera     = camera_name
        self.device     = torch.device("cpu")

        f = h5py.File(os.path.join(data_path, "demo.hdf5"), "r")
        grp = f["data"]
        self.task = grp.attrs["env"].replace("Sawyer", "")
        demos = list(grp.keys())[:limit_demo] if limit_demo else list(grp.keys())

        # ── собираем все массивы ────────────────────────────
        idx, states, vel, grip, xmls, ends = [], [], [], [], [], [0]
        for d in demos:
            g  = grp[d]
            st = g["states"][:]
            n  = len(st)
            win = np.clip(np.arange(n-1)[:,None] + np.arange(-horizon_left, horizon_right+1), 0, n-1)
            idx.append(win + ends[-1])
            states.append(st)
            vel.append(g["joint_velocities"][:])
            grip.append(g["gripper_actuations"][:])
            xmls.extend([g.attrs["model_file"]]*n)
            ends.append(ends[-1]+n)

        self.idx   = np.concatenate(idx)
        self.state = np.concatenate(states)
        self.vel   = np.concatenate(vel)
        self.grip  = np.concatenate(grip)
        self.xmls  = np.array(xmls)

        # ── off-screen кэш ─────────────────────────────────
        self.ecache = EnvCache(self.task,
                               models_dir=os.path.join(data_path,"models"),
                               cam_name=self.camera,
                               img_size=image_size,
                               default=True,        # можно False, если нужен XML
                               render=False)

        # ── memmap для RGB ────────────────────────────────
        self.img_dir  = os.path.join(data_path, f"images_{camera_name}_{image_size}")
        self.img_bin  = os.path.join(self.img_dir, "images.dat")
        self.meta_js  = os.path.join(self.img_dir, "images.meta")
        os.makedirs(self.img_dir, exist_ok=True)

        if not (os.path.isfile(self.img_bin) and os.path.isfile(self.meta_js)):
            self._render_and_store(batch=img_batch)

        with open(self.meta_js) as fp:
            meta = json.load(fp)
        N,H,W = meta["N"], meta["H"], meta["W"]
        self.img_mm = np.memmap(self.img_bin, mode="r", dtype=np.uint8,
                                shape=(N,H,W,3))

    # ───────────────────────────────────────────────────────
    def _render_and_store(self, batch: int):
        N,H,W = len(self.state), self.img_size, self.img_size
        img_mm = np.memmap(self.img_bin, mode="w+", dtype=np.uint8,
                           shape=(N,H,W,3))

        for start in tqdm(range(0, N, batch), desc="render", unit="img"):
            for i in range(start, min(start+batch, N)):
                env = self.ecache.get_env(self.xmls[i])
                env.sim.set_state_from_flattened(self.state[i])
                env.sim.forward()
                bgr = env.sim.render(width=W, height=H, camera_name=self.camera)
                img_mm[i] = bgr[..., ::-1]            # BGR→RGB

        img_mm.flush()
        json.dump({"N":N,"H":H,"W":W}, open(self.meta_js,"w"))
        print(f"RGB-кадры сохранены в {self.img_bin}")

    # ── pytorch API ────────────────────────────────────────
    def __len__(self):  return len(self.idx)

    def __getitem__(self, i):
        c = i
        img = torch.from_numpy(self.img_mm[c]).permute(2,0,1).float()/255.
        state  = torch.tensor(self.state[c], dtype=torch.float32)
        action = torch.tensor(self.vel[c], dtype=torch.float32)
        return {"pixels": img.to(self.device),
                "state" : state.to(self.device),
                "action": action.to(self.device)}



if __name__ == "__main__":

    MAX_FR = 25


    controller_name = 'JOINT_VELOCITY'

    arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)


    joint_dim = 7
    controller_settings = [joint_dim, joint_dim, -0.1]

    action_dim = joint_dim
    num_test_steps = joint_dim
    test_value = -0.1

    # Define the number of timesteps to use per controller action as well as timesteps in between actions
    steps_per_action = 75
    steps_per_rest = 75

    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(
        suite.make(
            "NutAssembly",
            robots="Sawyer",  # use Sawyer robot
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=True,  # make sure we can render to the screen
            horizon=(steps_per_action + steps_per_rest) * num_test_steps,
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
        )
    )

    env.reset()
    env.viewer.set_camera(camera_id=0)

    data_path = "../data/robot_demonstrations/RoboTurkPilot/pegs-full"
    ds = SawyerDataset(data_path=data_path,
                       horizon_left=2, horizon_right=8,
                       limit_demo=5)  # первый запуск — появится tqdm

    n = 0
    gripper_dim = 0
    for robot in env.robots:
        gripper_dim = robot.gripper["right"].dof
        n += int(robot.action_dim / (action_dim + gripper_dim))

    # Define neutral value
    neutral = np.zeros(action_dim + gripper_dim)

    # Keep track of done variable to know when to break loop
    count = 0
    # Loop through controller space

    # the gripper_pos is 6th element of action

    for i in range(10000):
        start = time.time()

        sample = ds[0]

        action = np.array(sample["action"])

        obs = env.step(action)
        env.render()

        # limit frame rate if necessary
        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)

    # Shut down this env before starting the next test
    env.close()

