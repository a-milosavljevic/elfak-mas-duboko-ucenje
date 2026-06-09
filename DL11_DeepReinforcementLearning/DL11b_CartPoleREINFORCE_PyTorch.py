import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import imageio
from IPython.display import Image, display

print("PyTorch version: " + torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Koristi se uređaj: {device}")


########################################################################################################################
# PODEŠAVANJE PUTANJA
########################################################################################################################

results_folder = os.getcwd()
agent_path = os.path.join(results_folder, "CartPoleREINFORCE_agent.pth")
fig_path = os.path.join(results_folder, "CartPoleREINFORCE_training.png")
gif_path = os.path.join(results_folder, "CartPoleREINFORCE_demo.gif")


########################################################################################################################
# POLICY MREŽA
########################################################################################################################

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Mreža direktno izbacuje distribuciju verovatnoća za akcije
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs


########################################################################################################################
# REINFORCE AGENT KLASA (sa statističkim baseline-om)
########################################################################################################################

class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99
        self.learning_rate = 0.005 

        self.model = PolicyNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Liste za pamćenje istorije unutar jedne epizode
        self.saved_log_probs = []
        self.rewards = []
        
        # Statistički baseline (pokretni prosek)
        self.baseline = None
        self.baseline_alpha = 0.1 # Faktor osvežavanja (koliko brzo zaboravljamo stare epizode)

    def act(self, state):
        state_tensor = torch.FloatTensor(state).to(device)
        action_probs = self.model(state_tensor)
        
        m = Categorical(action_probs)
        action = m.sample()
        
        # Pamtimo logaritam verovatnoće za kasniji backpropagation
        self.saved_log_probs.append(m.log_prob(action))
        
        return action.item()

    def learn_episode(self):
        if not self.rewards:
            return

        # 1. Računamo diskontovane nagrade (Returns) unazad
        returns = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns_tensor = torch.tensor(returns).to(device)
        
        # 2. Ažuriranje baseline-a na bazi trenutne epizode
        episode_mean_return = returns_tensor.mean().item()
        
        if self.baseline is None:
            self.baseline = episode_mean_return
        else:
            # Eksponencijalni pokretni prosek (Exponential Moving Average)
            self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * episode_mean_return

        # 3. Računanje Advantage-a (Return - Baseline)
        policy_losses = []
        
        for log_prob, R in zip(self.saved_log_probs, returns_tensor):
            # Koliko je ovaj potez bio bolji/gori od istorijskog proseka
            advantage = R.item() - self.baseline
            
            # Policy Gradient: loss = -log_prob * advantage
            policy_losses.append(-log_prob * advantage)
            
        # Zbir grešaka za celu epizodu
        loss = torch.stack(policy_losses).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Čišćenje lista za novu epizodu
        self.saved_log_probs.clear()
        self.rewards.clear()


########################################################################################################################
# GLAVNA PETLJA ZA OBUČAVANJE
########################################################################################################################

env = gym.make('CartPole-v1')
state_size = int(env.observation_space.shape[0])
action_size = int(env.action_space.n)

agent = REINFORCEAgent(state_size, action_size)
episodes = 500

scores = []

print("Započinjem obučavanje...")
for e in range(episodes):
    state, info = env.reset()
    state = np.reshape(state, [1, state_size])

    time_steps = 0
    done = False

    while not done:
        action = agent.act(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Kažnjavamo ga za pad
        if terminated:
            reward = -10

        agent.rewards.append(reward)

        state = np.reshape(next_state, [1, state_size])
        time_steps += 1

        if done:
            # Učimo na kraju epizode koristeći REINFORCE sa baseline-om
            agent.learn_episode()
            
            print(f"Epizoda: {e+1}/{episodes} | Rezultat: {time_steps} | Trenutni Baseline: {agent.baseline:.2f}")
            scores.append(time_steps)
            break

torch.save(agent.model.state_dict(), agent_path)
print(f"Težine modela su uspešno sačuvane u fajl '{agent_path}'!")

plt.plot(scores)
plt.ylabel('Preživljeni koraci (Score)')
plt.xlabel('Epizoda')
plt.title('REINFORCE (Policy Gradients) na CartPole okruženju')
plt.savefig(fig_path)


########################################################################################################################
# VIZUELIZACIJA RADA OBUČENOG AGENTA
########################################################################################################################

loaded_agent = REINFORCEAgent(state_size, action_size)
loaded_agent.model.load_state_dict(torch.load(agent_path, weights_only=True))
loaded_agent.model.eval()

print("Obučeni model je uspešno učitan sa diska.")

env_render = gym.make('CartPole-v1', render_mode='rgb_array')
state, info = env_render.reset()
state = np.reshape(state, [1, state_size])

frames = []
done = False
time_steps = 0

print("Pokrećem obučenog agenta i snimam frejmove...")

while not done:
    frames.append(env_render.render())

    state_tensor = torch.FloatTensor(state).to(device)
    with torch.no_grad():
        action_probs = loaded_agent.model(state_tensor)
        action = torch.argmax(action_probs).item()

    next_state, reward, terminated, truncated, info = env_render.step(action)
    done = terminated or truncated

    state = np.reshape(next_state, [1, state_size])
    time_steps += 1

env_render.close()

print(f"Epizoda završena nakon {time_steps} koraka. Generišem GIF animaciju...")
imageio.mimsave(gif_path, frames, duration=1000/30, loop=0)