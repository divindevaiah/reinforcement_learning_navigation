import sys
from os.path import join
import time
import signal
import random
import numpy as np

import gym
import rospkg
from termcolor import colored

from gym_ropod.envs.ropod_nav_env import RopodNavActions

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

RUNNING = False

class Network(nn.Module):
    
    def __init__(self, ):
        
        super(Network, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.fc1 = nn.Linear(507, 256)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.act3 = nn.ReLU()
        self.out = nn.Linear(64, 3)
        
    def forward(self, x):

        x = self.fc1(torch.tensor(x).float())
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.out(x)
        return x
    
    def update(self, state, y, model):
        
        self.optimizer = optim.RMSprop(model.parameters(), lr=0.0025, alpha=0.90, eps=0.01)
        y_pred = model(state)
        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class DQNAgent:
    
    def __init__(self):

        rospack = rospkg.RosPack()
        ropod_sim_pkg_path = rospack.get_path('ropod_sim_model')
        launch_file = join(ropod_sim_pkg_path, 'launch/simulator/ropod_origin_gazebo_simulator.launch')
        signal.signal(signal.SIGINT, sigint_handler)
        
        self.env = gym.make('ropod-nav-discrete-v0', launch_file_path=launch_file, env_type='square')
        self.env.render(mode='human')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 300
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99994
        self.batch_size = 600
        self.train_start = 1000
        
        self.model = Network()
        self.model = self.model.float()
        
    def remember(self, state, action, reward, next_state, done):
        
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
    def act(self, state):
        
        if np.random.uniform(0.0, 1.0) <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return torch.argmax(self.model(state))
        
    def replay(self):
        
        if len(self.memory) < self.train_start:
            return
        
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done, target = [], [], [], [] 

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        target = self.model(state)
        target_next = self.model(next_state)
            
        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (torch.max(target_next[i]))
            
        self.model.update(state, target, self.model)
        
    def load(self, name):
        
        device = torch.device('cpu')
        self.model = Network()
        self.model.load_state_dict(torch.load(name))
        self.model.to(device)
        
    def save(self, name):
        
        torch.save(self.model.state_dict(), name)
    
    def test(self, name):

        self.load(name)

        try:
            for e in range(self.EPISODES):
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                done = False
                i = 0
                RUNNING = True

                while not done:

                    if not RUNNING:
                        break

                    action = torch.argmax(self.model(state))
                    next_state, _, done, _ = self.env.step(torch.tensor(action).item())

                    print(colored('Step {0}: "{1}"'.format(i, RopodNavActions.action_num_to_str[torch.tensor(action).item()]), 'green'))

                    next_state = np.reshape(next_state, [1, self.state_size])

                    state = next_state
                    
                    i += 1
                    if done:
                        print(colored("Episode {0} finished after {1} timesteps. Epsilon: {2}".format(e, i, self.epsilon), 'yellow'))
                        print(colored('Resetting environment', 'yellow'))
                        self.env.reset()
                    else:
                        time.sleep(0.05)
                    
        except Exception as e:

            print(colored('Simulation interupted because of following error', 'red'))
            print(str(e))
            RUNNING = False

        finally:
            # close the simulation cleanly
            RUNNING = False
            print(colored('Closing simulator', 'green'))
            self.env.close()
            sys.exit(0)


    def run(self):

        # self.load('ropod-DQN07_600.pt')

        global RUNNING

        try:
            for e in range(self.EPISODES):
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                done = False
                i = 0
                RUNNING = True

                while not done:

                    if not RUNNING:
                        break
                                        
                    action = self.act(state)
                    next_state, reward, done, _ = self.env.step(torch.tensor(action).item())
                    next_state = np.reshape(next_state, [1, self.state_size])

                    if i < 1500:
                        reward = reward
                    else:
                        reward = reward - 100
                        done = True
                        print('MAX ITERATION REACHED!!!')

                    print(colored('Step {0}: "{1}" -> reward {2}, epsilon: {3}'.format(i, RopodNavActions.action_num_to_str[torch.tensor(action).item()],
                                                        reward, self.epsilon), 'green'))

                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    
                    i += 1
                    if done:
                        print(colored("Episode {0} finished after {1} timesteps. Epsilon: {2}".format(e, i, self.epsilon), 'yellow'))
                        print(colored('Resetting environment', 'yellow'))
                        self.env.reset()
                    else:
                        time.sleep(0.05)
                    
                    self.replay()

                if e == 300:
                    self.save('ropod-DQN07_300.pt')
                elif e == 600:
                    self.save('ropod-DQN07_600.pt')
                elif e == 900:
                    self.save('ropod-DQN07_900.pt')
                elif e == 1200:
                    self.save('ropod-DQN07_1200.pt')
                elif e == 1500:
                    self.save('ropod-DQN07_1500.pt')
                elif e == 1800:
                    self.save('ropod-DQN07_1800.pt')
                    
            self.save('ropod-DQN07.pt')

        except Exception as e:
            print(colored('Simulation interupted because of following error', 'red'))
            print(str(e))
            RUNNING = False

        finally:
            # close the simulation cleanly
            RUNNING = False
            print(colored('Closing simulator', 'green'))
            self.env.close()
            sys.exit(0)


def sigint_handler(signum, frame):
    global RUNNING
    print(colored('Simulation interupted', 'red'))
    RUNNING = False

def main():

    agent = DQNAgent()
    time.sleep(5)
    agent.run()
    # agent.test('ropod-DQN07_100.pt')

if __name__ == "__main__":
    main()

