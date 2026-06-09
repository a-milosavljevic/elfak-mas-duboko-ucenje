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
agent_path = os.path.join(results_folder, "CartPoleActorCritic_agent.pth")
fig_path = os.path.join(results_folder, "CartPoleActorCritic_training.png")
gif_path = os.path.join(results_folder, "CartPoleActorCritic_demo.gif")


########################################################################################################################
# ACTOR-CRITIC MREŽA
########################################################################################################################

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 128)
        self.actor_fc = nn.Linear(128, action_size)
        self.critic_fc = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.actor_fc(x), dim=-1)
        state_value = self.critic_fc(x)
        return action_probs, state_value


########################################################################################################################
# ACTOR-CRITIC AGENT KLASA
########################################################################################################################

class ActorCriticAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99
        self.learning_rate = 0.005 # Malo viša stopa učenja je standard za A2C

        self.model = ActorCriticNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Liste za pamćenje istorije unutar jedne epizode
        self.saved_log_probs = []
        self.state_values = []
        self.rewards = []

    def act(self, state):
        state_tensor = torch.FloatTensor(state).to(device)
        action_probs, state_value = self.model(state_tensor)
        
        m = Categorical(action_probs)
        action = m.sample()
        
        # Interno pamtimo podatke za učenje na kraju epizode
        self.saved_log_probs.append(m.log_prob(action))
        self.state_values.append(state_value)
        
        # Vraćamo samo akciju
        return action.item()

    def learn_episode(self):
        if not self.rewards:
            return

        # Računamo diskontovane nagrade unazad
        returns = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(device)
        
        # Normalizacija nagrada - ključno za stabilnost Kritičara!
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        policy_losses = []
        critic_losses = []
        
        for log_prob, value, R in zip(self.saved_log_probs, self.state_values, returns):
            advantage = R - value.item()
            
            policy_losses.append(-log_prob * advantage)
            critic_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(device)))
            
        # Zbir grešaka (Kritičara množimo sa 0.5)
        loss = torch.stack(policy_losses).sum() + 0.5 * torch.stack(critic_losses).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Čišćenje lista za novu epizodu
        self.saved_log_probs.clear()
        self.state_values.clear()
        self.rewards.clear()


########################################################################################################################
# GLAVNA PETLJA ZA OBUČAVANJE
########################################################################################################################

env = gym.make('CartPole-v1')
state_size = int(env.observation_space.shape[0])
action_size = int(env.action_space.n)

agent = ActorCriticAgent(state_size, action_size)
episodes = 500

scores = []

print("Započinjem obučavanje...")
for e in range(episodes):
    state, info = env.reset()
    state = np.reshape(state, [1, state_size])

    time_steps = 0
    done = False

    while not done:
        # Sada dobijamo samo akciju
        action = agent.act(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Kažnjavamo ga jako za pad
        if terminated:
            reward = -10

        # Dodajemo nagradu u internu memoriju agenta
        agent.rewards.append(reward)

        state = np.reshape(next_state, [1, state_size])
        time_steps += 1

        if done:
            # Učimo tek na kraju cele epizode!
            agent.learn_episode()
            
            print(f"Epizoda: {e+1}/{episodes} | Rezultat: {time_steps}")
            scores.append(time_steps)
            break

torch.save(agent.model.state_dict(), agent_path)
print(f"Težine modela su uspešno sačuvane u fajl '{agent_path}'!")

plt.plot(scores)
plt.ylabel('Preživljeni koraci (Score)')
plt.xlabel('Epizoda')
plt.title('Actor-Critic Treniranje na CartPole okruženju')
plt.savefig(fig_path)


########################################################################################################################
# VIZUELIZACIJA RADA OBUČENOG AGENTA
########################################################################################################################

loaded_agent = ActorCriticAgent(state_size, action_size)
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
        action_probs, _ = loaded_agent.model(state_tensor)
        # Za testiranje biramo akciju sa najvećom verovatnoćom (bez semplovanja)
        action = torch.argmax(action_probs).item()

    next_state, reward, terminated, truncated, info = env_render.step(action)
    done = terminated or truncated

    state = np.reshape(next_state, [1, state_size])
    time_steps += 1

env_render.close()

print(f"Epizoda završena nakon {time_steps} koraka. Generišem GIF animaciju...")
imageio.mimsave(gif_path, frames, duration=1000/30, loop=0)