import numpy as np


class Environment:
    def __init__(self, limit_steps = 9, size=2):
        self.state = self.init_state()
        self.action_history = []
        self.limit_steps = limit_steps
        self.s = size ** 2


    def one_hot(self, i, j):
        flat_idx = i * self.s + j
        res = np.zeros((self.s,))
        res[flat_idx] = 1
        return res

    def init_state(self):
        tensor = np.zeros((self.s, self.s, self.s))
        for i in range(self.s):
            for j in range(self.s):
                z_idx = self.one_hot(i, j) # z here is index of the resulting matrix C_{i,j}

                for k in range(self.s):
                    a_idx = self.one_hot(i, k) # a index of A_{ik}
                    b_idx = self.one_hot(k, j) # b index of B_{k,j}

                    # then the outer product z_idx (x) a_idx (x) b_idx will form then rank-1 tensor, which tells:
                    # to get C element with flattened coordinate where z_idx = 1 we need to multiply elements of A and B with flat coordinates on their respective axes
                    # since all the vectors are e_m then in the resulting tensor there will be only 1 non-zero element which encodes the combination

                    tensor += np.outer(a_idx, b_idx, z_idx)


        # DeepMind also explain change basis idea, for generating more data from single run and add variance to the system
        # //TODO change basis

        return tensor

