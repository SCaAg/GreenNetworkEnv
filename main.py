# Import necessary libraries
import os
import re
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt

# Import your custom environment
from com_env import CommunicationEnv

# Define the PPO agent
class PPOAgent(nn.Module):
    def __init__(self, num_micro_stations, lr=1e-4):
        super(PPOAgent, self).__init__()
        input_dim = 4 * num_micro_stations  # Observation dimension
        action_dim = 2 * num_micro_stations  # Total number of binary actions

        # Shared layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

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
        x = torch.relu(self.fc3(x))

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
            return action.cpu().numpy(), m.log_prob(action).cpu().numpy()
    
    def evaluate_actions(self, state, action):
        logits, state_value = self.forward(state)
        probs = torch.sigmoid(logits)
        m = Bernoulli(probs)
        action_log_probs = m.log_prob(action)
        dist_entropy = m.entropy()
        return action_log_probs, torch.squeeze(state_value), dist_entropy

    def infer_action(self, state):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(state)
            probs = torch.sigmoid(logits)
            # Deterministic action selection based on probabilities
            action = (probs > 0.5).float()
            return action.cpu().numpy()

# Hyperparameters
NUM_EPISODES = 2000
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPSILON = 0.1
CRITIC_DISCOUNT = 0.5
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
PPO_EPOCHS = 4
MAX_GRAD_NORM = 0.5

# Function to find the latest model
def find_latest_model():
    pth_files = [f for f in os.listdir('.') if os.path.isfile(f) and f.startswith('ppo_agent_') and f.endswith('.pth')]
    if not pth_files:
        return None, 0  # No model found
    # Extract episode numbers
    pattern = re.compile(r'ppo_agent_(\d+)\.pth')
    episodes = []
    for f in pth_files:
        match = pattern.match(f)
        if match:
            episodes.append(int(match.group(1)))
    if not episodes:
        return None, 0  # No valid model files
    max_episode = max(episodes)
    latest_model = f'ppo_agent_{max_episode}.pth'
    return latest_model, max_episode

# Training function
def train():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    env = gym.make('CommunicationEnv-v0', render_mode='human')
    num_micro_stations = len(env.micro_stations)

    # Initialize agent and transfer to device (GPU or CPU)
    agent = PPOAgent(num_micro_stations, lr=1e-4).to(device)

    # Check for existing models and load the latest one
    latest_model_path, starting_episode = find_latest_model()
    if latest_model_path is not None:
        agent.load_state_dict(torch.load(latest_model_path))
        print(f"Loaded model from {latest_model_path}, starting from episode {starting_episode + 1}")
    else:
        print("No existing model found, starting training from scratch")
        starting_episode = 0  # Ensure starting_episode is zero if no model is found

    # Training loop
    episode_rewards = []

    total_episodes = starting_episode + NUM_EPISODES

    for episode in range(starting_episode, total_episodes):
        current_episode = episode - starting_episode + 1  # Episode count for printing
        if current_episode % 24 != 0:
            state_dict, _ = env.reset(options="continue")
        else:
            state_dict, _ = env.reset()
        # Flatten and concatenate observation components
        state = np.concatenate([
            state_dict["consumed_energy"],
            state_dict["needed_energy"],
            state_dict["battery_level"],
            state_dict["sbs_status"]
        ])
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # Transfer state to device

        log_probs_list = []
        values_list = []
        states_list = []
        actions_list = []
        rewards_list = []
        masks_list = []

        episode_reward = 0

        # Initialize losses for this episode
        episode_policy_loss = 0
        episode_value_loss = 0
        episode_entropy_loss = 0

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
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)  # Transfer next state to device

            # Store transition
            log_probs_list.append(torch.tensor(log_prob).to(device))
            values_list.append(agent.forward(state)[1])
            rewards_list.append(torch.tensor([reward], dtype=torch.float32).to(device))  # Transfer reward to device
            masks_list.append(torch.tensor([1 - done], dtype=torch.float32).to(device))  # Transfer mask to device
            states_list.append(state)
            actions_list.append(torch.tensor(action).to(device))  # Transfer action to device

            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        # Compute returns and advantages
        next_value = agent.forward(state)[1].detach()
        returns = []
        advantages = []
        gae = torch.tensor([0.0]).to(device)  # Initialize GAE on device
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

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

                # Accumulate losses
                episode_policy_loss += policy_loss.item()
                episode_value_loss += value_loss.item()
                episode_entropy_loss += entropy_loss.item()

                # Backpropagation
                agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                agent.optimizer.step()

        episode_rewards.append(episode_reward)
        print(f"Episode {current_episode}/{NUM_EPISODES}, Reward: {episode_reward}")

        # Save the model and losses every 100 episodes
        if (episode + 1) % 100 == 0:
            model_save_path = f'ppo_agent_{episode + 1}.pth'
            torch.save(agent.state_dict(), model_save_path)
            print(f"Model saved at episode {episode + 1}")

    # Plot the rewards
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.show()

# Usage example
if __name__ == "__main__":
    a=input("Press 1 to start training...")
    if a=='1':
        train()
    else:   
    # 检查是否有可用的GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # 加载已保存的模型
        model_path = 'ppo_agent_100.pth'
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            # 创建环境
            env = gym.make('CommunicationEnv-v0', render_mode='human')  # 使用human模式渲染
            num_micro_stations = len(env.micro_stations)
            
            # 初始化 agent 并移动到 device (GPU 或 CPU)
            agent = PPOAgent(num_micro_stations, lr=1e-4).to(device)
            agent.load_state_dict(torch.load(model_path, map_location=device))
            agent.eval()  # 设置模型为评估模式

            print("Model loaded successfully. Running inference for 1000 steps...")
            state_dict, _ = env.reset()
            state = np.concatenate([
                state_dict["consumed_energy"],
                state_dict["needed_energy"],
                state_dict["battery_level"],
                state_dict["sbs_status"]
            ])
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # 转换成 tensor 并移动到 GPU

            total_reward = 0
            for step in range(1000):
                # 使用CUDA进行推理
                action = agent.infer_action(state)
                action_dict = {
                    "energy_allocation": action[0, :num_micro_stations].astype(int),
                    "sbs_status": action[0, num_micro_stations:].astype(int)
                }
                
                # 与环境交互
                next_state_dict, reward, done, truncated, info = env.step(action_dict)
                
                # 更新状态并移动到GPU
                next_state = np.concatenate([
                    next_state_dict["consumed_energy"],
                    next_state_dict["needed_energy"],
                    next_state_dict["battery_level"],
                    next_state_dict["sbs_status"]
                ])
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
                state = next_state

                total_reward += reward

                if done or truncated:
                    print(f"Episode ended at step {step} with reward {total_reward}")

            print(f"Total reward after 1000 steps: {total_reward}")
        else:
            print(f"Model {model_path} not found!")

