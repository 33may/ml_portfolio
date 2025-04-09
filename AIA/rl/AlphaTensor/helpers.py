import numpy as np
import torch

from AIA.rl.AlphaTensor.net.network import PolicyHead


def outer(u, v, w):
    return np.einsum('i,j,k->ijk', u, v, w, dtype=np.int32, casting="same_kind")

def is_zero_tensor(tensor):
    return np.all(tensor == 0)

def action_seq_to_actions(action_seq, separator = 3):
    action_seq = action_seq.detach().cpu().numpy()
    action_seq -= 1
    separator -= 1

    sep_idx = np.where(action_seq == separator)[0]

    return action_seq

def sample_random_factor():
    factor = np.random.randint(-1,1, (4,))
    return -factor if factor[0] < 0 else factor

def get_random_action():
    return sample_random_factor(), sample_random_factor(), sample_random_factor()

tensor_state = torch.randn((1, 512))
policy = PolicyHead()

a = policy(tensor_state)[0, :]

res = action_seq_to_actions(a)

# sep_indices = (a == separator).nonzero(as_tuple=True)[0]

# boundaries = torch.cat([torch.tensor([-1]), sep_indices, torch.tensor([len(a)])])
#
# segments = [a[boundaries[i]+1:boundaries[i+1]] for i in range(len(boundaries)-1)]

print(33)