import numpy as np
import copy
from typing import Optional

from AIA.rl.AlphaTensor.helpers import is_zero_tensor


class MCTSNode:
    def __init__(self, state, parent=None, action_taken=None, is_terminal=False):
        self.state = state                  # Current tensor state
        self.parent = parent                # Parent node in the tree
        self.action_taken = action_taken    # Action that led from parent to this node
        self.is_terminal = is_terminal

        self.children = []                  # List of child nodes
        self.visits = 0                     # Number of visits to this node
        self.value_sum = 0.0                # Sum of all values backpropagated through this node

        self.untried_actions = None         # Will be set during expansion
        # Note: self.untried_actions is a list of actions not yet used to create child nodes

    def is_fully_expanded(self):
        """
        Return True if all actions have already been expanded into children.
        """
        if self.untried_actions is None:
            return False  # Not expanded yet
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.0):
        """
        Use UCB1 formula to choose best child: Q + c * sqrt(log(N_parent) / N_child)
        Higher score means better.
        """
        choices = [
            (child.value_sum / (child.visits + 1e-8)) +
            c_param * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-8))
            for child in self.children
        ]
        return self.children[np.argmax(choices)]

    def expand(self, env, net):
        """
        Expand this node:
        1. Call the network on the current state to get value and policy
        2. Decode a set of promising actions from the policy
        3. Pick one untried action and apply it to get a new state
        4. Create a new child node with the resulting state
        """
        if self.untried_actions is None:
            # First time expansion — call NN to get policy and value
            # Prepare network input from state
            tensor_input, scalar_input = self.prepare_network_input()
            action_sequence, value = net(tensor_input, scalar_input)

            # TODO: Decode `action_sequence` into a list of valid action tensors
            self.untried_actions = self.decode_actions(action_sequence)

        # Pick actions to expand

        action = self.untried_actions.pop()
        env_copy = copy.deepcopy(env)
        next_state, done = env_copy.step(action)

        child_node = MCTSNode(state=next_state,
                              parent=self,
                              action_taken=action,
                              is_terminal=is_zero_tensor(next_state))
        self.children.append(child_node)
        return child_node, value

    def backpropagate(self, reward):
        """
        After simulation/evaluation, propagate the reward back up the tree.
        Each node in the path increases its visit count and adds reward to value_sum.
        """
        self.visits += 1
        self.value_sum += reward
        if self.parent is not None:
            self.parent.backpropagate(reward)

    def is_terminal(self):
        """
        Determine if this state is terminal. This usually means the game is over
        or all elements in the tensor are zero.
        """
        return self.env

    def prepare_network_input(self):
        """
        Prepares tensors and scalars for the network based on this node's state.
        Example: depth, previous actions, etc.
        """
        # TODO: Implement based on your model's expected input format
        raise NotImplementedError

    def decode_actions(self, token_sequence):
        """
        Decodes the output from the policy head into valid tensor actions.
        Each action should be a 3-vector outer product (u, v, w).
        """
        # TODO: Implement token decoding to action tensors
        raise NotImplementedError


class MCTS:
    def __init__(self, env, net, simulations=100):
        self.env = env            # Environment instance
        self.net = net            # Neural network (policy + value)
        self.simulations = simulations

    def search(self, initial_state):
        root = MCTSNode(initial_state)

        for _ in range(self.simulations):
            node = root
            env_copy = copy.deepcopy(self.env)
            env_copy.state = copy.deepcopy(initial_state)

            # === 1. SELECTION ===
            # Traverse tree using best_child() until we hit a leaf (not fully expanded)
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()
                env_copy.step(node.action_taken)

            # === 2. EXPANSION ===
            if not node.is_terminal():
                node, value = node.expand(env_copy, self.net)
            else:
                # Terminal nodes can also be evaluated
                value = self.evaluate_terminal(node)

            # === 3. BACKPROPAGATION ===
            node.backpropagate(value)

        # === 4. ACTION SELECTION ===
        # Pick child with the most visits (or highest value)
        best = max(root.children, key=lambda n: n.visits)
        return best.action_taken

    def evaluate_terminal(self, node):
        """
        Return value for a terminal state.
        You might define it as -1 * steps left or 0 if perfect solution.
        """
        # TODO: Customize this based on your task (e.g., reward based on remaining error)
        return 0.0  # placeholder