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

# def create_trajectory_indices(episode_ends, horizon_left, horizon_right):
#     """
#     The method to precompute all possible windows that will be used in training process. When the left/right horizon is outside the one episode, it is padded with the most left/right index.
#
#     Args:
#         episode_ends_array:
#         horizon_left:
#         horizon_right:
#
#     Returns:
#
#     """
#     def add_to_window(input_array, prediction_array, item, cur_item, input_limit):
#         if cur_item <= input_limit:
#             input_array.append(item)
#         else:
#             prediction_array.append(item)
#
#
#     windows = []
#     start_idx = 0
#     input_lim_idx = horizon_left + 1
#     for i in tqdm(range(len(episode_ends) - 1), total=len(episode_ends) - 1):
#         end_idx = episode_ends[i + 1]
#         if i > 0:
#             start_idx = episode_ends[i] + 1
#
#         # print("seq: ", start_idx, end_idx)
#
#         for cur_idx in range(start_idx, end_idx):
#             input = []
#             prediction = []
#             counter = 0
#             for displace in range(-horizon_left, horizon_right - horizon_left + 2):
#                 if cur_idx + displace < start_idx:
#                     # input.append(start_idx)
#                     idx_at_position = start_idx
#                 elif cur_idx + displace > end_idx:
#                     # prediction.append(end_idx)
#                     idx_at_position = end_idx
#                 else:
#                     idx_at_position = cur_idx + displace
#                     # prediction.append(cur_idx + displace)
#
#                 counter += 1
#                 add_to_window(input, prediction, idx_at_position, counter, input_lim_idx)
#
#             windows.append((input, prediction))
#
#     return windows