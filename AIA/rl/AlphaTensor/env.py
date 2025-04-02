import numpy as np

from AIA.rl.AlphaTensor.helpers import outer, is_zero_tensor


class Environment:
    def __init__(self, limit_steps = 9, size=2, terminate_penalty = 10):
        self.action_history = []
        self.limit_steps = limit_steps
        self.size = size
        self.s = size ** 2
        self.accumulate_reward = 0
        self.step_count = 0
        self.terminate_penalty = terminate_penalty
        self.state = self.init_state()


    def one_hot(self, i, j):
        flat_idx = i * np.sqrt(self.size) + j
        res = np.zeros((self.s,), dtype=np.int32)
        res[flat_idx] = 1
        return res

    def init_state(self):
        tensor = np.zeros((self.s, self.s, self.s), dtype=np.int32)
        for i in range(self.size):
            for j in range(self.size):
                z_idx = self.one_hot(i, j) # z here is index of the resulting matrix C_{i,j}

                for k in range(self.size):
                    a_idx = self.one_hot(i, k) # a index of A_{ik}
                    b_idx = self.one_hot(k, j) # b index of B_{k,j}

                    # then the outer product z_idx (x) a_idx (x) b_idx will form then rank-1 tensor, which tells:
                    # to get C element with flattened coordinate where z_idx = 1 we need to multiply elements of A and B with flat coordinates on their respective axes
                    # since all the vectors are e_m then in the resulting tensor there will be only 1 non-zero element which encodes the combination

                    tensor += outer(a_idx, b_idx, z_idx)


        # DeepMind also explain change basis idea, for generating more data from single run and add variance to the system
        # //TODO change basis

        return tensor

    def step(self, action):
        u, v ,w = action

        self.action_history.append((u, v, w))
        step_tensor = outer(u, v, w)
        self.state -= step_tensor
        self.accumulate_reward -= 1
        self.step_count += 1

        if self.is_terminated():
            return self.state, True
        if self.step_count == self.limit_steps:
            self.accumulate_reward -= self.terminate_penalty
            return self.state, True

        return self.state, False

    def is_terminated(self):
        return is_zero_tensor(self.state)

    def reset(self,
              init_state=None):
        if init_state is None:
            init_state = self.init_state()
        self.state = init_state
        self.accumulate_reward = 0
        self.step_count = 0

if __name__ == '__main__':
    test_env = Environment(size=4,
                           limit_steps=8)
    test_action = np.array([
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0]
    ])
    for _ in range(8):
        print(test_env.step(test_action))
        print(test_env.accumulate_reward)
    import pdb; pdb.set_trace()