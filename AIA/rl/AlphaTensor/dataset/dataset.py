import numpy as np
import pandas as pd

from AIA.rl.AlphaTensor.helpers import outer


def sample_random_factor(s):
    factor = np.random.randint(-1,1, (s,))
    return -factor if factor[0] < 0 else factor

def shuffle_actions(actions):
    return list(np.random.permutation(actions))

def generate_trajectory(s=4, num_shuffle=4):
    num_factors = 7 + np.random.randint(0, 4)
    rows = []
    actions = []

    tensor = np.zeros((s, s, s))

    # build full tensor from random factors
    for _ in range(num_factors):
        u, v, w = sample_random_factor(s), sample_random_factor(s), sample_random_factor(s)
        actions.append([u, v, w])
        component = outer(u, v, w)
        tensor += component

    # step-by-step remove factors and save intermediate state
    for i in range(len(actions)):
        row = {
            "tensor": tensor.copy(),        # snapshot of current tensor
            "actions": actions[i:],         # remaining actions
            "value": len(actions[i:])       # ground truth value: steps to finish
        }
        rows.append(row)

        u, v, w = actions[i]
        tensor -= outer(u, v, w)  # undo one action

    return tensor, rows

def generate_synthetic_dataset(dataset_size, num_shuffle=4, tensor_size=2, save_to_file=True):
    rows = []
    trajectories_num = dataset_size // num_shuffle + 1

    for _ in range(trajectories_num):
        tensor, trajectory_variations = generate_trajectory(s=tensor_size, num_shuffle=num_shuffle)
        for traj in trajectory_variations:
            rows.append(traj)

    dataset = pd.DataFrame(rows, columns=["tensor", "actions", "value"])

    if save_to_file:
        dataset.to_pickle("synthetic_dataset.pkl")

    return dataset

dataset = generate_synthetic_dataset(20)

