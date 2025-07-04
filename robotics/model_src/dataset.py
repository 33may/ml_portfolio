import itertools
import time
from pathlib import Path
from typing import Any, Dict, OrderedDict

import h5py
import torch
import zarr
import random
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np


# ==== Relevant Code ====

# ---------- helpers -----------------------------------------------------------
def create_trajectory_indices(episode_ends: np.ndarray,
                              horizon_left: int,
                              horizon_right: int) -> np.ndarray:
    """
    Pre‑compute every possible window (one row = full window indices).

    episode_ends  – cumulative end indices, e.g. [0, 4, 8, 10]
    horizon_left  – how many frames *before* current step to feed as obs
    horizon_right – how many future actions to predict

    Returns
    -------
    np.ndarray, shape = (N_windows, W)
        W = horizon_left + horizon_right + 1
        Each row already clipped to the episode boundaries.
    """
    all_windows = []
    start_idx = 0
    window_template = np.arange(-horizon_left, horizon_right + 1) # [W,]
    for i in range(len(episode_ends) - 1):
        end_idx = episode_ends[i + 1]
        if i > 0:
            start_idx = episode_ends[i] + 1 # first valid frame in ep

        base = np.arange(start_idx, end_idx)[:, None] # [L, 1]

        windows = base + window_template # [L, W]

        np.clip(windows, start_idx, end_idx, out=windows) # padding

        all_windows.append(windows)

    return np.concatenate(all_windows, axis=0) # (N, W)


def normalize_data(arr, scale, dtype=np.float32):
    # map raw values from [0, scale] into canonical [0, 1] range
    return arr.astype(dtype, copy=False) / scale




def denormalize_data(arr, scale, dtype=np.float32):
    # recover original units by reversing the previous scaling
    return arr.astype(dtype, copy=False) * scale


# ---------- dataset -----------------------------------------------------------
class PushTDataset(Dataset):
    """
    PyTorch dataset that returns:
        img_obs  – images for the observation horizon (oh,H,W,C)
        act_obs  – actions for the observation horizon (oh, 2)
        act_pred – actions for the prediction horizon (ph, 2)
    All indices are pre‑computed once in create_trajectory_indices().
    """
    def __init__(self, data_path, obs_horizon, prediction_horizon, image_size = None, images = None, actions = None, episode_ends = None):
        self.obs_horizon = obs_horizon
        self.prediction_horizon = prediction_horizon

        dataset = zarr.open(data_path, mode="r")  # action, img, keypoint, n_contacts, state

        if data_path:
            image_data = dataset["data"]["img"][:]  # ndarray [0-255], shape = (total, 224, 224, 3)
            image_data = np.moveaxis(image_data, -1, 1)
            actions_data = dataset["data"]["action"][:]  # ndarray [0-512], shape = (total, 2)
            self.episode_ends = dataset['meta']['episode_ends'][:] - 1
        else:
            image_data = images
            actions_data = actions
            self.episode_ends = episode_ends[:] - 1

        # --- images ---------------------------------------------------------
        self.image_data_transformed = normalize_data(image_data, 255).astype(np.float32) # ndarray [0-1], shape = (total, 224, 224, 3)

        # --- actions --------------------------------------------------------
        self.actions_data_transformed = normalize_data(actions_data, 512).astype(np.float32) # ndarray [0-1], shape = (total, 2)

        # --- windows --------------------------------------------------------
        self.indexes = create_trajectory_indices(self.episode_ends, obs_horizon, prediction_horizon)

    # total number of windows
    def __len__(self):
        return len(self.indexes)

    # slice arrays by pre‑computed row of indices
    def __getitem__(self, idx):
        trajectory_idx = self.indexes[idx]

        img_obs  = self.image_data_transformed[trajectory_idx[:self.obs_horizon + 1]]
        act_obs  = self.actions_data_transformed[trajectory_idx[:self.obs_horizon + 1]]
        act_pred = self.actions_data_transformed[trajectory_idx[self.obs_horizon + 1:]]

        return {
            "img_obs" : img_obs,
            "act_obs" : act_obs,
            "act_pred" : act_pred,
        }


class RobosuiteImageActionDataset(Dataset):
    """
    PyTorch dataset that returns:
        img_obs  – images for the observation horizon (oh,H,W,C)
        act_obs  – actions for the observation horizon (oh, 2)
        act_pred – actions for the prediction horizon (ph, 2)
    All indices are pre‑computed once in create_trajectory_indices().
    """
    def __init__(self, data_path, camera_type = "agentview", obs_horizon = 2, pred_horizon = 8, image_size = 224, demos = None):
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.camera_type = camera_type + "_image_normalized" if camera_type else camera_type

        self.image_size = image_size

        f = h5py.File(data_path, "r")

        self.data = f["data"]

        self.demos = demos if demos else self.data.keys()

        self.indexes = None

        episode_ends = [0]
        actions = []
        episode_lens = []
        states = []

        first_flag = True
        for demo_name in self.demos:
            demo_data = self.data[demo_name]

            demo_actions = demo_data["actions"][:]
            demo_states = demo_data["states"][:]

            episode_len, _ = demo_actions.shape

            episode_lens.append(episode_len)

            episode_end = episode_ends[-1] + episode_len

            if first_flag:
                episode_end -= 1
                first_flag = False


            actions.append(demo_actions)
            states.append(demo_states)
            episode_ends.append(episode_end)

        actions_np = np.concatenate(actions, axis=0)

        self.episode_ends = np.array(episode_ends)
        self.episode_lens = episode_lens

        if camera_type:
            self.load_data()
        else:
            states_np = np.concatenate(states, axis=0)
            self.obs_data_transformed = states_np.astype(np.float32)

        self.obs_shape = self.obs_data_transformed[0].shape


        self.actions_data_transformed = actions_np.astype(np.float32)


    def drop_data(self):
        self.indexes = None
        self.obs_data_transformed = None

    def load_data_fn(self):
        images_np = np.ndarray((self.episode_ends[-1] + 1, self.image_size, self.image_size, 3), dtype=np.float32)

        offset = 0
        first = True

        for demo_name, ep_len in tqdm(zip(self.demos, self.episode_lens)):
            demo_data = self.data[demo_name]

            # n = ep_len - (1 if first else 0)

            n = ep_len

            buf = images_np[offset:offset + n]
            demo_data["obs"][self.camera_type].read_direct(buf)

            offset += n

        images_np = np.moveaxis(images_np, -1, 1)

        return images_np

    def load_data(self):
        images_np = self.load_data_fn()

        # self.obs_data_transformed = normalize_data(images_np, 255)
        self.obs_data_transformed = images_np

        self.indexes = create_trajectory_indices(self.episode_ends, self.obs_horizon, self.pred_horizon)


    # total number of windows
    def __len__(self):
        return len(self.indexes)

    # slice arrays by pre‑computed row of indices
    def __getitem__(self, idx):
        trajectory_idx = self.indexes[idx]

        img_obs  = self.obs_data_transformed[trajectory_idx[:self.obs_horizon + 1]]
        act_obs  = self.actions_data_transformed[trajectory_idx[:self.obs_horizon + 1]]
        act_pred = self.actions_data_transformed[trajectory_idx[self.obs_horizon + 1:]]

        return {
            "img_obs" : img_obs,
            "act_obs" : act_obs,
            "act_pred" : act_pred,
        }


class RobosuiteImageActionDatasetMem(Dataset):
    """
    Хранит hdf5-демонстрации на диске, но одновременно
    держит в оперативной памяти не более `max_eps_in_ram` эпизодов.
    По обращениям кэш ведёт себя как LRU: самые «холодные» выталкиваются.
    """

    # ------------------------- init ---------------------------
    def __init__(
        self,
        path: str | Path,
        camera: str = "agentview",
        obs_horizon: int = 1,
        pred_horizon: int = 8,
        *,
        max_eps_in_ram: int = 4,
    ) -> None:
        super().__init__()

        self._path = str(path)
        # увеличенный chunk-cache ускоряет последовательное чтение
        self.f = h5py.File(
            self._path,
            "r",
            libver="latest",
            rdcc_nbytes=64 * 1024**2,   # 64 MiB под «горячие» chunk’и
            rdcc_nslots=1_000_003,      # ≈1 млн хеш-слотов
        )

        # raw- vs normalised-кадры
        self.cam = f"{camera}_image_normalized"
        self.already_norm = True

        self.oh, self.ph = obs_horizon, pred_horizon
        self.span = obs_horizon + pred_horizon + 1

        # -------- метаданные эпизодов ---------------------------------
        self.demos_raw: list[Any] = list(self.f["data"].values())

        self.episode_ends = np.fromiter(
            itertools.accumulate([-1] +
                                 [len(d["actions"]) for d in self.demos_raw]),
            dtype=np.int64,
        )  # inclusive
        self.episode_starts = self.episode_ends[:-1] + 1

        self.demo_of_step = np.empty(self.episode_ends[-1] + 1, np.int32)
        for i, (lo, hi) in enumerate(
            zip(self.episode_starts, self.episode_ends[1:] + 1)
        ):
            self.demo_of_step[lo:hi] = i

        # -------- все допустимые окна ---------------------------------
        self.indexes = create_trajectory_indices(
            self.episode_ends, obs_horizon, pred_horizon
        )

        # -------- LRU-кэш эпизодов -----------------------------------
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self.max_eps_in_ram = max_eps_in_ram

    # ------------------- helpers ------------------------------------
    @staticmethod
    def _frames_to_tensor(frames: np.ndarray, already_norm: bool) -> torch.Tensor:
        t = torch.as_tensor(frames, dtype=torch.float32)
        if not already_norm:               # uint8 → [0,1]
            t = t.div_(255.0)
        return t.permute(0, 3, 1, 2)       # B HWC → B CHW

    def _get_demo(self, demo_id: int):
        """LRU-доступ к эпизоду."""
        d = self._cache.get(demo_id)
        if d is None:
            d = self.demos_raw[demo_id]
            self._cache[demo_id] = d
            self._cache.move_to_end(demo_id)          # mark MRU
            if len(self._cache) > self.max_eps_in_ram:
                self._cache.popitem(last=False)       # выгнать LRU
        return d

    # ---------------- Dataset API -----------------------------------
    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row_global = self.indexes[idx]                 # (span,)
        demo_id    = self.demo_of_step[row_global[0]]
        d          = self._get_demo(demo_id)
        row_local  = row_global - self.episode_starts[demo_id]

        # уникальные временные шаги, чтобы не читать кадр дважды
        uniq, inv  = np.unique(row_local, return_inverse=True)
        frames_u   = d["obs"][self.cam][uniq]
        acts_u     = d["actions"][uniq].astype(np.float32)

        frames = frames_u[inv]                        # восстановить дубликаты
        acts   = acts_u[inv]

        return {
            "img_obs":  self._frames_to_tensor(frames, self.already_norm)
                         [: self.oh + 1],
            "act_obs":  torch.from_numpy(acts)[: self.oh + 1],
            "act_pred": torch.from_numpy(acts)[self.oh + 1:],
        }

    # ---------------- pickling (DataLoader workers) -----------------
    def __getstate__(self):
        st = self.__dict__.copy()
        st["f"] = None
        st["_cache"] = None
        st["demos_raw"] = None
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        if self.f is None:                              # внутри воркера
            self.f = h5py.File(
                self._path,
                "r",
                libver="latest",
                rdcc_nbytes=64 * 1024**2,
                rdcc_nslots=1_000_003,
            )
            self.demos_raw = list(self.f["data"].values())
            self._cache = OrderedDict()



# ==== Utility and old Code ====

def generate_sample_dataset(n):
    actions = []
    images = []
    step_in_episode = 0
    episode_ends = [0]

    action = 0
    counter = 0

    for _ in range(n):
        action += 1
        counter += 1

        images.append(f"img{counter}")
        actions.append(action)

        step_in_episode += 1

        if random.random() > 0.8:
            episode_ends.append(episode_ends[-1] + step_in_episode)
            step_in_episode = 0
            action = 0

    return images, actions, episode_ends

