import sys
from os.path import join
import time
import signal
import random
import numpy as np
import q_learning
from q_learning import discretize

import gym
import rospkg
from termcolor import colored

from gym_ropod.envs.ropod_nav_env import RopodNavActions

RUNNING = False

def sigint_handler(signum, frame):
    global RUNNING
    print(colored('Simulation interupted', 'red'))
    RUNNING = False

def main():
    global RUNNING
    rospack = rospkg.RosPack()
    ropod_sim_pkg_path = rospack.get_path('ropod_sim_model')
    launch_file = join(ropod_sim_pkg_path, 'launch/simulator/ropod_origin_gazebo_simulator.launch')
    number_of_steps = 5000

    env = gym.make('ropod-nav-discrete-v0', launch_file_path=launch_file, env_type='square')
    env.render(mode='human')
    time.sleep(5)

    # data = env.reset()
    signal.signal(signal.SIGINT, sigint_handler)

    q_obj = q_learning.q_learning(500, 7)
    q_table = q_obj.create_q_table()

    try:
        for i_episode in range(2):
            observation = env.reset()
            print(colored('Running simulation of episode {0} for {1} steps'.format(i_episode, number_of_steps), 'green'))
            episode_step_count = 0
            RUNNING = True
            for i in range(number_of_steps):
                
                if not RUNNING:
                    break

                discrete_observation = np.zeros(503)
                for idx, i in enumerate(observation):
                    value, bucket = ([], 0)
                    if idx == 0:
                        value = q_obj.robot_pose_x
                        bucket = q_obj.robot_pose_x_bucket
                    elif idx == 1:
                        value = q_obj.robot_pose_y
                        bucket = q_obj.robot_pose_y_bucket
                    elif idx == 2:
                        value = q_obj.robot_pose_theta
                        bucket = q_obj.robot_pose_theta_bucket
                    else:
                        value = q_obj.lidar
                        bucket = q_obj.lidar_bucket
                    
                    discrete_observation[idx] = discretize(i, value, bucket)

                if random.uniform(0, 1) > q_obj.epsilon and q_obj.i_episode < max_number_exploration:
                    action = env.action_space.sample()  # Explore action space
                else:
                    action = np.argmax(obj.q_table[tuple(discrete_observation)])  # Exploit learned values

                (next_observation, reward, done, info) = env.step(action)

                print(colored('Step {0}: "{1}" -> reward {2}'.format(i, RopodNavActions.action_num_to_str[action],
                                                        reward), 'green'))
                episode_step_count += 1

                next_discrete_observation = np.zeros(503)
                for idx, i in enumerate(observation):
                    next_value, next_bucket = ([], 0)
                    if idx == 0:
                        next_value = q_obj.robot_pose_x
                        next_bucket = q_obj.robot_pose_x_bucket
                    elif idx == 1:
                        next_value = q_obj.robot_pose_y
                        next_bucket = q_obj.robot_pose_y_bucket
                    elif idx == 2:
                        next_value = q_obj.robot_pose_theta
                        next_bucket = q_obj.robot_pose_theta_bucket
                    else:
                        next_value = q_obj.lidar
                        next_bucket = q_obj.lidar_bucket
                    
                    next_discrete_observation[idx] = discretize(i, next_value, next_bucket)


                old_value = obj.q_table[tuple(discrete_observation)][action]
                # print('Old_value: ', old_value)

                next_max = np.max(obj.q_table[tuple(next_discrete_observation)])
                # print('Next_Max: ', next_max)

                # Q formula and updating q_table
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                obj.q_table[discrete_observation][action] = new_value
                observation = next_observation

                if done:
                    print(colored('Episode done after {0} steps'.format(episode_step_count), 'yellow'))
                    print(colored('Resetting environment', 'yellow'))
                    env.reset()
                    episode_step_count = 0
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
        env.close()
        sys.exit(0)

if __name__ == "__main__":
    main()