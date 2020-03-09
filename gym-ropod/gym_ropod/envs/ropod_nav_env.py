from typing import Tuple, Sequence

import os
import numpy as np
import math
from gym import spaces

from gym_ropod.envs.ropod_env import RopodEnv, RopodEnvConfig
from gym_ropod.utils.model import PrimitiveModel
from gym_ropod.utils.geometry import GeometryUtils

class RopodNavActions(object):
    '''Defines the following navigation action mappings:
    action_num_to_str: Dict[int, str] -- maps integers describing actions
                                         (belonging to the action space
                                          gym.spaces.Discrete(5) to descriptive
                                          action names)
    action_to_vel: Dict[str, List[float]] -- maps action names to 2D velocity commands
                                             of the form [x, y, theta]

    @author Alex mitrevski
    @contact aleksandar.mitrevski@h-brs.de

    '''
    action_num_to_str = {
        0: 'forward',
        # 1: 'left',
        # 2: 'right',
        1: 'left_turn',
        2: 'right_turn'
        # 5: 'backward',
        # 6: 'do_nothing'
    }

    action_to_vel = {
        'forward': [0.8, 0.0, 0.0],
        # 'left': [0.0, 0.8, 0.0],
        # 'right': [0.0, -0.8, 0.0],
        'left_turn': [0.8, 0.0, 0.5],
        'right_turn': [0.8, 0.0, -0.5]
        # 'backward': [-0.8, 0.0, 0.0],
        # 'do_nothing': [0.0, 0.0, 0.0]
    }


class RopodNavDiscreteEnv(RopodEnv):
    '''A navigation environment for a ROPOD robot with a discrete action space.

    @author Alex Mitrevski
    @contact aleksandar.mitrevski@h-brs.de

    '''
    def __init__(self, launch_file_path: str,
                 env_type: str='square',
                 number_of_obstacles: int=0):
        '''Throws an AssertionError if "env_name" is not in RopodEnvConfig.env_to_config or
        the environment variable "ROPOD_GYM_MODEL_PATH" is not set.

        Keyword arguments:

        launch_file_path: str -- absolute path to a launch file that starts the ROPOD simulation
        cmd_vel_topic: str -- name of a ROS topic for base velocity commands
                              (default "/ropod/cmd_vel")
        env_type: str -- type of an environment (default "square")
        number_of_obstacles: int -- number of obstacles to add to the environment (default 0)

        '''
        super(RopodNavDiscreteEnv, self).__init__(launch_file_path)

        if env_type not in RopodEnvConfig.env_to_config:
            raise AssertionError('Unknown environment "{0}" specified'.format(env_type))

        if 'ROPOD_GYM_MODEL_PATH' not in os.environ:
            raise AssertionError('The ROPOD_GYM_MODEL_PATH environment variable is not set')

        self.model_path = os.environ['ROPOD_GYM_MODEL_PATH']
        self.env_config = RopodEnvConfig.env_to_config[env_type]
        self.number_of_obstacles = number_of_obstacles

        self.action_space = spaces.Discrete(len(RopodNavActions.action_num_to_str))
        # self.observation_space = spaces.Box(0., 5., (503,))
        self.observation_space = spaces.Box(0., 5., (507,))

        self.robot_reached_goal = False

        self.collision_punishment = -100.
        self.goal_reached_reward = 300.
        self.direction_change_punishment = -5.
        self.observation_reward = 10
        self.__inf = float('inf')

        self.area = 20.0
        self.rel_pose = None
        self.goal_pose = None
        self.previous_action = None

    def step(self, action: int) -> Tuple[Tuple[float, float, float],
                                         Sequence[float], float, bool]:
        '''Publishes a velocity command message based on the given action.
        Returns:
        * a list in which
            * the first three elements represent the current goal the robot is pursuing
              (pose in the form (x, y, theta))
            * the subsequent elements represent the current laser scan measurements
        * obtained reward after performing the action
        * an indicator about whether the episode is done
        * an info dictionary containing a single key - "goal" -
          with the goal pose in the format (x, y, theta) as its value

        Keyword arguments:
        action: int -- a navigation action to execute

        '''
        # applying the action
        vels = RopodNavActions.action_to_vel[RopodNavActions.action_num_to_str[action]]
        self.vel_msg.linear.x = vels[0]
        self.vel_msg.linear.y = vels[1]
        self.vel_msg.angular.z = vels[2]
        self.vel_pub.publish(self.vel_msg)

        # preparing the result
        observation = [x if x != self.__inf else self.laser_scan_msg.range_max
                       for x in self.laser_scan_msg.ranges]

        goal_found = GeometryUtils.poses_equal(self.robot_pose, self.goal_pose, 
                                               position_tolerance = 4.0, orientation_tolerance = (2 * np.pi))

        if goal_found:
            self.robot_reached_goal = True
            print('GOAL FOUND!!!')
        else:
            self.robot_reached_goal = False

        if self.robot_under_collision:
            print('COLLISION DETECTED!!!')


        done = self.robot_under_collision or goal_found

        self.previous_action = action

        self.rel_pose = np.array(self.goal_pose) - np.array(self.robot_pose)

        observation = self.nonLin_observation(observation)

        reward = self.get_reward(action)

        return (list(observation) + list([self.rel_pose[0], self.rel_pose[1]]) + list(self.robot_pose) + list([self.goal_pose[0], self.goal_pose[1]]), reward, done, {'goal': self.goal_pose})

    def get_reward(self, action: int) -> float:
        '''Calculates the reward obtained by applying the given action
        using the following equation:

        R_t = \frac{1}{d} + c_1\mathbf{1}_{c_t=1} + c_2\mathbf{a_{t-1} \neq a_t}

        where
        * d is the distance from the robot to the goal
        * c_t indicates whether the robot has collided
        * a_t is the action at time t
        * c_1 is the value of self.collision_punishment
        * c_2 is the vlaue of self.direction_change_punishment

        Keyword arguments:
        action: int -- an executed action

        '''
        goal_dist = GeometryUtils.distance(self.robot_pose, self.goal_pose)
        collision = 1 if self.robot_under_collision else 0
        goal_reached = 1 if self.robot_reached_goal else 0
        direction_change = 1 if action != self.previous_action else 0

        # self.rel_pose

        reward = (self.area * 4.0 * math.log(goal_dist + .1, .1)) + \
                 collision * self.collision_punishment + \
                 direction_change * self.direction_change_punishment + \
                 goal_reached * self.goal_reached_reward + self.observation_reward

        # reward = (np.exp(.02 * reward) - np.exp(-.02 * reward)) / (np.exp(.02 * reward) + np.exp(-.02 * reward))

        return reward

    def reset(self):
        '''Resets the simulation environment. The first three elements of the
        returned observation represent the current goal the robot is pursuing
        (pose in the form (x, y, theta)); the subsequent elements represent
        the current laser measurements.
        '''
        super().reset()

        self.robot_reached_goal = False

        # we add the static environment models
        for model in self.env_config.models:
            self.insert_env_model(model)

        # we add obstacles to the environment
        for i in range(self.number_of_obstacles):
            pose, collision_size, visual_size = self.sample_model_parameters()
            model_name = 'box_' + str(i+1)
            model = PrimitiveModel(name=model_name,
                                   sdf_path=os.path.join(self.model_path, 'models/box.sdf'),
                                   model_type='box', pose=pose,
                                   collision_size=collision_size,
                                   visual_size=visual_size)
            self.insert_dynamic_model(model)

        # we generate a goal pose for the robot
        self.goal_pose = self.generate_goal_pose()
        
        goal_model = PrimitiveModel(name='goal',
                                    sdf_path=os.path.join(self.model_path, 'models/box.sdf'),
                                    model_type='box', pose=((self.goal_pose[0], self.goal_pose[1], 0.), (0., 0., self.goal_pose[2])),
                                    collision_size=(0.02, 0.02, 0.02),
                                    visual_size=(6.5, 6.5, 0.2))
        self.insert_dynamic_model(goal_model)

        # preparing the result
        observation = [x if x != self.__inf else self.laser_scan_msg.range_max
                       for x in self.laser_scan_msg.ranges]
        
        self.rel_pose = np.array(self.robot_pose) - np.array(self.goal_pose)

        observation = self.nonLin_observation(observation)

        return list(observation) + list([self.rel_pose[0], self.rel_pose[1]])  + list(self.robot_pose) + list([self.goal_pose[0], self.goal_pose[1]])
    
    def state_reduction(self, state):
        '''Reduces the observation space from 500 to 50'''
        avg = len(state) / float(50)
        out = []
        last = 0.0

        while last < len(state):
            out.append(np.average(state[int(last):int(last + avg)]))
            last += avg

        return out

    def nonLin_observation(self, observation):
        
        obs = np.log(np.array(observation) + 0.1) / np.log(1.1)

        self.observation_reward = np.average(obs)

        return obs

    def generate_goal_pose(self) -> Tuple[float, float, float]:
        '''Randomly generates a goal pose in the environment, ensuring that
        the pose does not overlap any of the existing objects.
        '''
        goal_pose_found = False
        pose = None
        while not goal_pose_found:
            position_x = np.random.uniform(self.env_config.boundaries[0][0],
                                           self.env_config.boundaries[0][1])
            # position_y = np.random.uniform(self.env_config.boundaries[1][0],
            #                                self.env_config.boundaries[1][1])
            # orientation_z = np.random.uniform(-np.pi, np.pi)

            # position_x = np.random.choice([np.random.uniform(-10, -6), 
            #                             np.random.uniform(6, 10)])
            position_y = np.random.choice([np.random.uniform(-10, -6), 
                                        np.random.uniform(6, 10)])
            orientation_z = np.random.uniform(-np.pi, np.pi)

            pose = (position_x, position_y, orientation_z)
            if not self.__pose_overlapping_models(pose):
                goal_pose_found = True

        self.goal_position.publish(str(pose))

        return pose

    def sample_model_parameters(self) -> Tuple[Tuple, Tuple, Tuple]:
        '''Generates a random pose as well as collision and visual sizes
        for a dynamic model. The parameters are generated as follows:
        * for the pose, only the position and z-orientation are set;
          the x and y positions are sampled from the environment boundaries specified in
          self.env_config, the orientation is sampled between -pi and pi radians, and
          the z position is set to half the z collision size in order for the model
          to be on top of the ground
        * the collision sizes are sampled between 0.2 and 1.0 in all three directions
        * the visual size is the same as the collision size
        '''
        collision_size_x = np.random.uniform(0.2, 1.0)
        collision_size_y = np.random.uniform(0.2, 1.0)
        collision_size_z = np.random.uniform(0.2, 1.0)
        collision_size = (collision_size_x, collision_size_y, collision_size_z)
        visual_size = collision_size

        position_x = np.random.uniform(self.env_config.boundaries[0][0],
                                       self.env_config.boundaries[0][1])
        position_y = np.random.uniform(self.env_config.boundaries[1][0],
                                       self.env_config.boundaries[1][1])
        position_z = collision_size_z / 2.
        orientation_z = np.random.uniform(-np.pi, np.pi)
        pose = ((position_x, position_y, position_z), (0., 0., orientation_z))

        return (pose, visual_size, collision_size)

    def __pose_overlapping_models(self, pose: Tuple[float, float, float]):
        '''Returns True if the given pose overlaps with any of the existing
        objects in the environment; returns False otherwise.

        Keyword arguments:
        pose: Tuple[float, float, float]: a 2D pose in the format (x, y, theta)

        '''
        for model in self.models:
            if GeometryUtils.pose_inside_model(pose, model):
                return True
        return False