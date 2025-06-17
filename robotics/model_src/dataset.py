import time

import h5py
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
    for i in tqdm(range(len(episode_ends) - 1), total=len(episode_ends) - 1):
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
    def __init__(self, data_path, camera_type = "agentview", obs_horizon = 2, prediction_horizon = 8, image_size = 124):
        self.obs_horizon = obs_horizon
        self.prediction_horizon = prediction_horizon
        self.camera_type = camera_type + "_image" if camera_type else camera_type

        f = h5py.File(data_path, "r")

        data = f["data"]

        episode_ends = [0]
        actions = []
        episode_lens = []
        states = []

        first_flag = True
        for demo_name in tqdm(data.keys()):
            demo_data = data[demo_name]

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

        if camera_type:
            time_start = time.time()
            images_np = np.ndarray((episode_ends[-1] + 1, image_size, image_size, 3), dtype=np.uint8)

            offset = 0
            first = True

            for demo_name, ep_len in zip(data.keys(), episode_lens):
                demo_data = data[demo_name]

                # n = ep_len - (1 if first else 0)

                n = ep_len

                buf = images_np[offset:offset + n]
                demo_data["obs"][self.camera_type].read_direct(buf)

                offset += n

            images_np = np.moveaxis(images_np, -1, 1)

            self.obs_data_transformed = normalize_data(images_np, 255)
        else:
            states_np = np.concatenate(states, axis=0)
            self.obs_data_transformed = states_np.astype(np.float32)

        self.obs_shape = self.obs_data_transformed[0].shape

        self.episode_ends = np.array(episode_ends)

        self.actions_data_transformed = actions_np.astype(np.float32)



        # --- windows --------------------------------------------------------
        self.indexes = create_trajectory_indices(self.episode_ends, obs_horizon, prediction_horizon)

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

