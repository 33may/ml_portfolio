import numpy as np
import pandas as pd

from AIA.rl.AlphaTensor.helpers import outer, get_random_action


def shuffle_actions(actions):
    return list(np.random.permutation(actions))

def generate_trajectory():
    num_factors = 7 + np.random.randint(0, 4)
    rows = []
    actions = []

    tensor = np.zeros((4, 4, 4))

    # build full tensor from random factors
    for _ in range(num_factors):
        u, v, w = get_random_action()
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

def generate_synthetic_dataset(dataset_size, num_shuffle=4, save_to_file=True):
    rows = []
    trajectories_num = dataset_size // num_shuffle + 1

    for _ in range(trajectories_num):
        tensor, trajectory_variations = generate_trajectory()
        for traj in trajectory_variations:
            rows.append(traj)

    dataset = pd.DataFrame(rows, columns=["tensor", "actions", "value"])

    if save_to_file:
        dataset.to_pickle("synthetic_dataset.pkl")

    return dataset