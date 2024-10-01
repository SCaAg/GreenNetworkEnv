# Import necessary libraries
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt

# Import your custom environment
from test import CommunicationEnv

# Define the PPO agent
class PPOAgent(nn.Module):
    def __init__(self, num_micro_stations, lr=3e-4):
        super(PPOAgent, self).__init__()
        input_dim = 4 * num_micro_stations  # Observation dimension
        action_dim = 2 * num_micro_stations  # Total number of binary actions

        # Shared layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)

        # Policy head
        self.policy_head = nn.Linear(64, action_dim)

        # Value head
        self.value_head = nn.Linear(64, 1)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        # Shared layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Policy output (logits for Bernoulli distributions)
        logits = self.policy_head(x)

        # Value output
        state_value = self.value_head(x)

        return logits, state_value

    def select_action(self, state):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(state)
            # For Bernoulli distribution, logits correspond to log-odds
            probs = torch.sigmoid(logits)
            m = Bernoulli(probs)
            action = m.sample()
            return action.numpy(), m.log_prob(action).numpy()
    
    def evaluate_actions(self, state, action):
        logits, state_value = self.forward(state)
        probs = torch.sigmoid(logits)
        m = Bernoulli(probs)
        action_log_probs = m.log_prob(action)
        dist_entropy = m.entropy()
        return action_log_probs, torch.squeeze(state_value), dist_entropy

# Hyperparameters
NUM_EPISODES = 1000
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
CRITIC_DISCOUNT = 0.5
ENTROPY_BETA = 0.01
BATCH_SIZE = 64
PPO_EPOCHS = 10
MAX_GRAD_NORM = 0.5

# Create environment
env = gym.make('CommunicationEnv-v0',render_mode='human')
num_micro_stations = len(env.micro_stations)

# Initialize agent
agent = PPOAgent(num_micro_stations)

# Training loop
episode_rewards = []

for episode in range(NUM_EPISODES):
    state_dict, _ = env.reset()
    # Flatten and concatenate observation components
    state = np.concatenate([
        state_dict["consumed_energy"],
        state_dict["needed_energy"],
        state_dict["battery_level"],
        state_dict["sbs_status"]
    ])
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Shape: [1, input_dim]

    log_probs_list = []
    values_list = []
    states_list = []
    actions_list = []
    rewards_list = []
    masks_list = []

    episode_reward = 0

    for step in range(999999):  # Limit the number of steps per episode
        # Select action
        action, log_prob = agent.select_action(state)
        action_dict = {
            "energy_allocation": action[0, :num_micro_stations].astype(int),
            "sbs_status": action[0, num_micro_stations:].astype(int)
        }

        # Step the environment
        next_state_dict, reward, done, truncated, info = env.step(action_dict)

        # Process next state
        next_state = np.concatenate([
            next_state_dict["consumed_energy"],
            next_state_dict["needed_energy"],
            next_state_dict["battery_level"],
            next_state_dict["sbs_status"]
        ])
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        # Store transition
        log_probs_list.append(torch.tensor(log_prob))
        values_list.append(agent.forward(state)[1])
        rewards_list.append(torch.tensor([reward], dtype=torch.float32))
        masks_list.append(torch.tensor([1 - done], dtype=torch.float32))
        states_list.append(state)
        actions_list.append(torch.tensor(action))

        state = next_state
        episode_reward += reward

        if done or truncated:
            break

    # Compute returns and advantages
    next_value = agent.forward(state)[1].detach()
    returns = []
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards_list))):
        delta = rewards_list[i] + GAMMA * next_value * masks_list[i] - values_list[i]
        gae = delta + GAMMA * GAE_LAMBDA * masks_list[i] * gae
        advantages.insert(0, gae)
        next_value = values_list[i]

    returns = [advantage + value for advantage, value in zip(advantages, values_list)]

    # Flatten lists
    states = torch.cat(states_list)
    actions = torch.cat(actions_list)
    returns = torch.cat(returns).detach()
    advantages = torch.cat(advantages).detach()
    log_probs = torch.cat(log_probs_list).detach()

    # Optimize policy and value network
    for _ in range(PPO_EPOCHS):
        for idx in range(0, len(states), BATCH_SIZE):
            batch_slice = slice(idx, idx + BATCH_SIZE)
            batch_states = states[batch_slice]
            batch_actions = actions[batch_slice]
            batch_returns = returns[batch_slice]
            batch_advantages = advantages[batch_slice]
            batch_old_log_probs = log_probs[batch_slice]

            # Evaluate actions
            new_log_probs, state_values, dist_entropy = agent.evaluate_actions(batch_states, batch_actions)

            # Policy loss
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = CRITIC_DISCOUNT * (batch_returns - state_values).pow(2).mean()

            # Entropy loss
            entropy_loss = -ENTROPY_BETA * dist_entropy.mean()

            # Total loss
            loss = policy_loss + value_loss + entropy_loss

            # Backpropagation
            agent.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            agent.optimizer.step()

    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1}/{NUM_EPISODES}, Reward: {episode_reward}")

# Plot the rewards
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()

# Close the environment
env.close()
