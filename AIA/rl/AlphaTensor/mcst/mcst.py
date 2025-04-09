import random

import numpy as np
import copy
from typing import Optional

from AIA.rl.AlphaTensor.env import mimic_step_fn
from AIA.rl.AlphaTensor.helpers import is_zero_tensor, get_random_action, action_seq_to_actions


class MCTSNode:
    def __init__(self, state, depth, parent=None, action_taken=None, is_terminal=False):
        self.state = state                  # Current tensor state
        self.parent = parent                # Parent node in the tree
        self.action_from_parent = action_taken    # Action that led from parent to this node
        self.is_terminal = is_terminal
        self.taken_actions = set()

        self.children = []                  # List of child nodes
        self.cur_max_rank = 0
        self.total_rank = 0
        self.visits = 0                     # Number of visits to this node
        self.value_sum = 0.0                # Sum of all values backpropagated through this node

        self.is_expanded = False         # Will be set during expansion

        self.depth = depth

        self.limit_nodes_c = 10
        self.c_puct = 1.5

    def get_limit_nodes(self):
        return self.limit_nodes_c * np.sqrt(self.visits)

    def expand(self, net):
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
        new_actions = action_seq_to_actions(action_sequence) # action sequence generates 4 actions

        new_unique_actions = list(set(new_actions) - self.taken_actions)
        self.taken_actions.update(new_unique_actions)

        # for all actions that are already expanded generate new random actions
        num_random_actions = 4 - len(new_unique_actions)

        for idx, action in enumerate(new_unique_actions):
            new_state, is_terminated = mimic_step_fn(self.state, action)
            child_node = MCTSNode(state=new_state,
                                  depth=self.depth + 1,
                                  parent=self,
                                  action_taken=action,
                                  is_terminal=is_terminated)
            rank = self.cur_max_rank + len(new_unique_actions) - idx
            self.total_rank += rank
            self.children.append((child_node, rank))

        for i in range(num_random_actions):
            action = get_random_action()

            if action not in self.taken_actions:
                new_state, is_terminated = mimic_step_fn(self.state, action)
                child_node = MCTSNode(state=new_state,
                                      depth=self.depth + 1,
                                      parent=self,
                                      action_taken=action,
                                      is_terminal=is_terminated)
                self.children.append((child_node, 1))
                self.total_rank += 1
                self.taken_actions.add(action)


    def explore(self):
        return random.random() < 0.5

    def select_best_child(self, net):
        """
        When we come to the node, the select method is called.

        It either expands the node or not.

        Then process to select the best child node.
        """

        if len(self.children) < self.get_limit_nodes() or self.explore():
            self.expand(net)

        best_score = -float('inf')
        best_child = None

        for child_node, rank in self.children:
            # Q = value_sum / visits
            if child_node.visits == 0:
                q_value = 0  # You can also use child_node.value_estimate if available
            else:
                q_value = child_node.value_sum / child_node.visits

            prior = rank / self.total_rank if self.total_rank > 0 else 0

            u_value = self.c_puct * prior * (np.sqrt(self.visits) / (1 + child_node.visits))

            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_child = child_node

        return best_child


    def backpropagate(self, reward):
        """
        After simulation/evaluation, propagate the reward back up the tree.
        Each node in the path increases its visit count and adds reward to value_sum.
        """

    def prepare_network_input(self):
        """
        Prepares tensors and scalars for the network based on this node's state.
        Example: depth, previous actions, etc.
        """
        # TODO: Implement based on your model's expected input format
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