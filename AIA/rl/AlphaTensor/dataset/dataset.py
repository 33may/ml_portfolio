import numpy as np
import pandas as pd

from AIA.rl.AlphaTensor.helpers import outer


def sample_random_factor(s):
    factor = np.random.randint(-2,2, (s,))
    return -factor if factor[0] < 0 else factor

def shuffle_actions(actions):
    return list(np.random.permutation(actions))

def generate_trajectory(s=4, num_shuffle = 4):
    num_factors = 7 + np.random.randint(0, 4)

    actions = []

    tensor = np.zeros((s, s, s))

    for step in range(num_factors):
        u, v, w = sample_random_factor(s), sample_random_factor(s), sample_random_factor(s)

        actions.append([u,v,w])

        component = outer(u, v, w)

        tensor += component

    actions = [actions]

    for _ in range(num_shuffle):
        shuffled_actions = shuffle_actions(actions[0])
        actions.append(shuffled_actions)


    return tensor, actions


def generate_synthetic_dataset(dataset_size, num_shuffle=4, tensor_size=2, save_to_file=True):
    dataset = pd.DataFrame(columns=["actions", "tensor"])

    trajectories_num = dataset_size // num_shuffle + 1

    for item in range(trajectories_num):
        tensor, trajectory_variations = generate_trajectory(s=tensor_size, num_shuffle=num_shuffle)

        for traj in trajectory_variations:
            row = {"actions": traj, "tensor": tensor}
            dataset = dataset.append(row, ignore_index=True)

    if save_to_file:
        dataset.to_pickle("synthetic_dataset.pkl")

    return dataset


