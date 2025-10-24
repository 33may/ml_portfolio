import numpy as np
import torch
from torch import nn



# Torso
# ==========================
# input:
#     current_state: [batch x 4 x 4 x 4]
#     scalars: [batch x (depth)]
class Torso(nn.Module):
    def __init__(self, S=4, scalar_size=1, hidden_dim=512):
        super(Torso, self).__init__()
        input_dim = S * S * S + scalar_size  # 4*4*4 + 1 = 65

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.r2 = nn.ReLU()

    def forward(self, tensor_input
                # , scalar_input
                ):
        """
        tensor_input: [S, S, S] = [4, 4, 4]
        scalar_input: [1] (step)
        """
        x = tensor_input
        # s = scalar_input.reshape(1, -1)     # [1, 1]
        # x = torch.cat([x, s], dim=1)        # [1, 65]

        z1 = self.r1(self.l1(x))
        z2 = self.r2(self.l2(z1))
        return z2                           # state embedding [1, 512]


# Policy Head
class PolicyHead(nn.Module):
    def __init__(self, hidden_dim=512, vocab_size=4, max_len=52, pad_token_id=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, hidden_dim))
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tensor_emb):
        # tensor_state shape: [batch_size, hidden_dim], typically batch_size=1
        h = tensor_emb.unsqueeze(0)  # [1, batch_size, hidden_dim] (num_layers=1)
        input_token = torch.tensor([[self.pad_token_id]], device=tensor_emb.device)
        generated = []

        for step in range(self.max_len):
            # Token and positional embeddings
            tok_emb = self.token_embedding(input_token)
            pos_emb = self.pos_embedding[:, step:step+1, :]
            x = tok_emb + pos_emb

            # GRU step
            out, h = self.gru(x, h)
            logits = self.output(out.squeeze(1))

            # Ensure the 3 token is not generated except manual separator insertion
            logits[:, self.pad_token_id] = float('-inf')

            # Sample next token
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated.append(next_token.item())

            # Insert pad token every 12 steps (excluding step 0)
            if (step + 1) % 12 == 0 and (step + 1) < self.max_len:
                input_token = torch.tensor([[self.pad_token_id]], device=tensor_emb.device)
                generated.append(self.pad_token_id)
            else:
                input_token = next_token

            # Early stop if max_len reached
            if len(generated) >= self.max_len:
                break

        return torch.tensor(generated[:self.max_len], device=tensor_emb.device).unsqueeze(0)

class ValueHead(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.a1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.a2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, tensor_emb):

        z1 = self.a1(self.fc1(tensor_emb))
        z2 = self.a2(self.fc2(z1))
        out = self.fc3(z2)

        return out.squeeze(-1)


class Net:
    def __init__(self):
        self.torso = Torso()
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()
        self.last_torso = None

    def forward(self, tensor_input, scalar_input):
        tensor_embedding = self.torso(tensor_input, scalar_input)
        action_sequence = self.policy_head(tensor_embedding)
        value_sequence = self.value_head(tensor_embedding)
        return action_sequence[0, :], value_sequence


tensor_state = torch.randn((1, 512))
policy = PolicyHead()

a = policy(tensor_state)[0, :]

separator = 3

sep_indices = (a == separator).nonzero(as_tuple=True)[0]

boundaries = torch.cat([torch.tensor([-1]), sep_indices, torch.tensor([len(a)])])

segments = [a[boundaries[i]+1:boundaries[i+1]] for i in range(len(boundaries)-1)]

print(33)