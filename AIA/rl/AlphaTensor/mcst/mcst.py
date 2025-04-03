class Node:
    def __init__(self,
                 state,
                 parent,
                 pre_action,
                 pre_action_idx,
                 is_terminal=False):
        self.state = state
        self.parent = parent
        self.pre_action = pre_action
        self.pre_action_idx = pre_action_idx
        self.is_terminal = is_terminal