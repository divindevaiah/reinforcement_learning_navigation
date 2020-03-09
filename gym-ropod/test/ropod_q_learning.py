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
import torch.nn.functional as F
import torch.optim as optim

RUNNING = False

class Network(nn.Module):
    def __init__(self, ):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(503, 200, bias = True)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(200, 75, bias = True)
        self.act2 = nn.Tanh()
        self.out = nn.Linear(75, 7, bias = True)
        
    def forward(self, x):
        x = self.fc1(torch.tensor(x))
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.out(x)
        return x

def sigint_handler(signum, frame):
    global RUNNING
    print(colored('Simulation interupted', 'red'))
    RUNNING = False

def main():

    net = Network()
    optimizer = optim.Adamax(net.parameters(), lr=0.02)

    gamma = .80
    rAll = 0
    epsilon = 0.1
    max_number_exploration = 6000
    allQ = None

    global RUNNING
    rospack = rospkg.RosPack()
    ropod_sim_pkg_path = rospack.get_path('ropod_sim_model')
    launch_file = join(ropod_sim_pkg_path, 'launch/simulator/ropod_origin_gazebo_simulator.launch')
    number_of_steps = 8000

    env = gym.make('ropod-nav-discrete-v0', launch_file_path=launch_file, env_type='square')
    env.render(mode='human')
    time.sleep(5)

    # data = env.reset()
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        for i_episode in range(2000):
            observation = env.reset()
            print(colored('Running simulation of episode {0} for {1} steps'.format(i_episode, number_of_steps), 'green'))
            episode_step_count = 0
            RUNNING = True
            for i in range(number_of_steps):
                
                if not RUNNING:
                    break
                
                net.zero_grad()
                allQ = net(observation)

                if random.uniform(0, 1) > epsilon and i_episode < max_number_exploration:
                    action = env.action_space.sample()  # Explore action space
                else:
                    action = torch.argmax(allQ)
                    action = action.item()  # Exploit learned values

                (next_observation, reward, done, info) = env.step(action)

                print(colored('Step {0}: "{1}" -> reward {2}'.format(i, RopodNavActions.action_num_to_str[action],
                                                        reward), 'green'))
                episode_step_count += 1

                Q1 = net(next_observation)

                maxQ1 = torch.max(Q1)
                targetQ = allQ
                targetQ[action] = targetQ[action] + reward + (gamma * maxQ1)

                loss = F.mse_loss(allQ, targetQ)
                loss.backward()
                optimizer.step()

                rAll += reward
                observation = next_observation

                if done:
                    epsilon = 1 - 1./(1.0015**(i_episode) + .1)
                    print(colored('Episode done after {0} steps'.format(episode_step_count), 'yellow'))
                    print(colored('Resetting environment', 'yellow'))
                    env.reset()
                    episode_step_count = 0
                else:
                    time.sleep(0.05)

            print(allQ)
            torch.save(net.state_dict(), 'param_05032020.pt')

    except Exception as e:
        print(colored('Simulation interupted because of following error', 'red'))
        print(str(e))
        RUNNING = False

    finally:
        # close the simulation cleanly
        RUNNING = False
        print(colored('Closing simulator', 'green'))
        env.close()
        sys.exit(0)

if __name__ == "__main__":
    main()
