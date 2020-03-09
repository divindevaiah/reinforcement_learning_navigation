import numpy as np
import random
import time

def discretize(float_value, values, num_buckets):
    increment = (abs(values[0])+values[1])/num_buckets
    for i in range(0, num_buckets):
        if float_value < increment*(i+1) + values[0]:
            return i

class q_learning(object):
    def __init__(self, obs_dim, act_dims):
        self.obs_dim = obs_dim
        self.q_table = None
        self.q_table_dim = []
        
        self.lidar = [0., 4.]
        self.robot_pose_x = [-10, 10]
        self.robot_pose_y = [-10, 10]
        self.robot_pose_theta = [-np.pi, np.pi]

        self.robot_pose_x_bucket = 40
        self.robot_pose_y_bucket = 40
        self.robot_pose_theta_bucket = 40
        self.lidar_bucket = 40
        self.action_num = act_dims

        # Hyper-parameters
        self.alpha = 0.4
        self.gamma = 0.2
        self.epsilon = 0.8
        self.max_number_exploration = 1000

    def create_q_table(self):
        
        self.q_table_dim = [self.robot_pose_x_bucket, self.robot_pose_y_bucket, self.robot_pose_theta_bucket]

        for i in range(self.obs_dim):
            self.q_table_dim.append(self.lidar_bucket)
        
        self.q_table_dim.append(self.action_num)
        self.q_table = np.zeros(self.q_table_dim)
        
        return self.q_table