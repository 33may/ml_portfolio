import torch
from torch import nn


def train_policy(policy_model, critic_model, input, optimizer, compute_device):
    """

    Args:
        policy_model: Model to update
        critic_model: Critic model to compute the value of the proposed action
        input: the input of shape [batch_size, 5] = (batch_size, observation_space.shape)
        optimizer: optimizer for the policy model

    Returns:

    """
    optimizer.zero_grad()

    input = input.to(compute_device)

    output = policy_model(input)

    critic_input = torch.cat((input, output), dim=1)

    score = critic_model(critic_input)

    loss = -score.mean()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()

def train_critic(critic_model, input, target, optimizer, compute_device):
    """
    Train function that run one update on the critic network using batch of inputs.

    Args:
        critic_model: Model to update
        input: the input of shape [batch_size, 7] = (batch_size, observation_space.shape + action_space.shape)
        target: Is the target reward we received from env and critic_estimate model
        optimizer: optimizer for the critic model

    Returns:

    """
    optimizer.zero_grad()
    criterion = nn.MSELoss()

    critic_model.train()

    input = input.to(compute_device)

    output = critic_model(input)

    loss = criterion(output, target)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(critic_model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )