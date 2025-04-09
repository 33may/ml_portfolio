import numpy as np
import copy
from typing import Optional

from AIA.rl.AlphaTensor.env import mimic_step_fn
from AIA.rl.AlphaTensor.helpers import is_zero_tensor, get_random_action


class MCTSNode:
    def __init__(self, state, depth, parent=None, action_taken=None, is_terminal=False):
        self.state = state                  # Current tensor state
        self.parent = parent                # Parent node in the tree
        self.action_from_parent = action_taken    # Action that led from parent to this node
        self.is_terminal = is_terminal
        self.taken_actions = set()

        self.children = []                  # List of child nodes
        self.visits = 0                     # Number of visits to this node
        self.value_sum = 0.0                # Sum of all values backpropagated through this node

        self.is_expanded = False         # Will be set during expansion

        self.depth = depth

        self.limit_nodes_c = 10

    def get_limit_nodes(self):
        return self.limit_nodes_c * np.sqrt(self.visits)

    def expand(self, env, net):
        """
        Expand this node:
            1. add candidates from policy network (check for duplicates)
            2. check for already expanded nodes in children
            3. add new nodes
            4. fill the remaining with random actions

        """
        tensor_input, scalar_input = self.prepare_network_input()
        action_sequence, value = net(tensor_input, scalar_input)

        # //TODO update model estimate of the current node value


        # //TODO fix decode_action to properly work with step and mimic_step_fn (u,v,w)
        new_actions = self.decode_actions(action_sequence) # action sequence generates 4 actions

        new_unique_actions = list(set(new_actions) - self.taken_actions)
        self.taken_actions.update(new_unique_actions)

        # for all actions that are already expanded generate new random actions
        num_random_actions = 4 - len(new_unique_actions)

        for action in new_unique_actions:
            new_state, is_terminated = mimic_step_fn(self.state, action)
            child_node = MCTSNode(state=new_state,
                                  depth=self.depth + 1,
                                  parent=self,
                                  action_taken=action,
                                  is_terminal=is_terminated)
            self.children.append(child_node)

        for i in range(num_random_actions):
            action = get_random_action()

            if action not in self.taken_actions:
                new_state, is_terminated = mimic_step_fn(self.state, action)
                child_node = MCTSNode(state=new_state,
                                      depth=self.depth + 1,
                                      parent=self,
                                      action_taken=action,
                                      is_terminal=is_terminated)
                self.children.append(child_node)
                self.taken_actions.add(action)


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