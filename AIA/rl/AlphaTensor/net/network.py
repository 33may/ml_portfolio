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

    def forward(self, tensor_input, scalar_input):
        """
        tensor_input: [S, S, S] = [4, 4, 4]
        scalar_input: [1] (step)
        """
        x = tensor_input.reshape(1, -1)     # [1, 64]
        s = scalar_input.reshape(1, -1)     # [1, 1]
        x = torch.cat([x, s], dim=1)        # [1, 65]

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

    def forward(self, tensor_state):
        # tensor_state shape: [batch_size, hidden_dim], typically batch_size=1
        h = tensor_state.unsqueeze(0)  # [1, batch_size, hidden_dim] (num_layers=1)
        input_token = torch.tensor([[self.pad_token_id]], device=tensor_state.device)
        generated = []

        for step in range(self.max_len):
            # Token and positional embeddings
            tok_emb = self.token_embedding(input_token)
            pos_emb = self.pos_embedding[:, step:step+1, :]
            x = tok_emb + pos_emb

            # GRU step
            out, h = self.gru(x, h)
            logits = self.output(out.squeeze(1))
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated.append(next_token.item())

            # Insert pad token every 12 steps (excluding step 0)
            if (step + 1) % 12 == 0 and (step + 1) < self.max_len:
                input_token = torch.tensor([[self.pad_token_id]], device=tensor_state.device)
                generated.append(self.pad_token_id)
            else:
                input_token = next_token

            # Early stop if max_len reached
            if len(generated) >= self.max_len:
                break

        return torch.tensor(generated[:self.max_len], device=tensor_state.device).unsqueeze(0)



tensor_state = torch.randn((1, 512))


policy = PolicyHead()

a = policy(tensor_state)

# Value Head