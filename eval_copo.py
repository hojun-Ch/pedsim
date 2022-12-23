import cv2
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import random
import json
from arguments import parser

from mlagents_envs.environment import UnityEnvironment as UE
from gym_unity.envs import UnityToGymWrapper

import time
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os

from utils import obs_to_img_feature, obs_to_global_reward

from agents.copo import COPO

def eval_bypass(args, agent):
    home_path = args.home_path
    env_path = home_path + args.env_path + "eval_bypass.x86_64"
    
    channel = EngineConfigurationChannel()
    unity_env = UE(file_name = env_path, seed=1, side_channels=[channel], no_graphics=not args.rendering, worker_id=args.worker)
    
    env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=True)
    channel.set_configuration_parameters(time_scale = 1.0)
    
    goal = [0.0,0.0]
    np.random.seed(0)
    
    for i in range(1, args.num_ped):
        if i // 2 == 0:
            x = np.random.uniform(0.85, 0.95, 1)
            z = np.random.uniform(-0.1, -0.95, 1)
            goal.append(x[0])
            goal.append(z[0])
            
        else:
            x = np.random.uniform(0.85, 0.95, 1)
            z = np.random.uniform(0.1, 0.95, 1)
            goal.append(x[0])
            goal.append(z[0])        
    
    goal = np.array(goal)
    
    obs = env.reset()
    
    episode_return = 0
    episode_coll = 0
    
    for i in range(args.rollout_length):
        prev_obs = obs[0]
        img, feature = obs_to_img_feature(obs, goal, args.map_length, args.map_width, args.scale, args.img_size, args.num_ped, args.feature_dim, args.neighbor_distance)
        action, log_prob, value, n_value, g_value = agent.act(img, feature)
        obs, __, __, __ = env.step(action.reshape(-1))
        g_reward, g_coll = obs_to_global_reward(obs, prev_obs, goal, args.map_length, args.map_width, args.num_ped, args.coll_penalty, args.neighbor_distance)
        episode_return += g_reward
        episode_coll += g_coll
    
    env.close()
    return episode_return, episode_coll

def eval_crossing(args, agent):
    home_path = args.home_path
    env_path = home_path + args.env_path + "eval_crossing.x86_64"
    
    channel = EngineConfigurationChannel()
    unity_env = UE(file_name = env_path, seed=1, side_channels=[channel], no_graphics=not args.eval_rendering, worker_id=args.worker)
    
    env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=True)
    channel.set_configuration_parameters(time_scale = 1.0)
    
    goal = [0.0,0.0]
    np.random.seed(0)
    
    for i in range(1, args.num_ped):
        if i // 2 == 0:
            x = np.random.uniform(-0.55, -0.45, 1)
            z = np.random.uniform(-0.95, 0.95, 1)
            goal.append(x[0])
            goal.append(z[0])
            
        else:
            x = np.random.uniform(0.45, 0.55, 1)
            z = np.random.uniform(-0.95, 0.95, 1)
            goal.append(x[0])
            goal.append(z[0])        
    
    goal = np.array(goal)
    
    obs = env.reset()
    
    episode_return = 0
    episode_coll = 0
    
    for i in range(args.rollout_length):
        prev_obs = obs[0]
        img, feature = obs_to_img_feature(obs, goal, args.map_length, args.map_width, args.scale, args.img_size, args.num_ped, args.feature_dim, args.neighbor_distance)
        action, log_prob, value, n_value, g_value = agent.act(img, feature)
        obs, __, __, __ = env.step(action.reshape(-1))
        g_reward, g_coll = obs_to_global_reward(obs, prev_obs, goal, args.map_length, args.map_width, args.num_ped, args.coll_penalty, args.neighbor_distance)
        episode_return += g_reward
        episode_coll += g_coll
    env.close()
    return episode_return, episode_coll

def eval_spread(args, agent):
    home_path = args.home_path
    env_path = home_path + args.env_path + "eval_spread.x86_64"
    
    channel = EngineConfigurationChannel()
    unity_env = UE(file_name = env_path, seed=1, side_channels=[channel], no_graphics=not args.eval_rendering, worker_id=args.worker)
    
    env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=True)
    channel.set_configuration_parameters(time_scale = 1.0)
    
    goal = [0.0,0.0]
    np.random.seed(0)
    
    for i in range(1, args.num_ped):
        if i // 4 == 0:
            x = np.random.uniform(0.85, 0.95, 1)
            z = np.random.uniform(-0.95, 0.95, 1)
            goal.append(x[0])
            goal.append(z[0])
            
        elif i // 4 == 1:
            x = np.random.uniform(-0.85, -0.95, 1)
            z = np.random.uniform(-0.95, 0.95, 1)
            goal.append(x[0])
            goal.append(z[0])        
            
        elif i // 4 == 2:
            x = np.random.uniform(-0.95, 0.95, 1)
            z = np.random.uniform(-0.95, -0.85, 1)
            goal.append(x[0])
            goal.append(z[0]) 
            
        else:
            x = np.random.uniform(-0.95, 0.95, 1)
            z = np.random.uniform(0.85, 0.95, 1)
            goal.append(x[0])
            goal.append(z[0]) 
    
    goal = np.array(goal)
    
    obs = env.reset()
    
    episode_return = 0
    episode_coll = 0
    
    for i in range(args.rollout_length):
        prev_obs = obs[0]
        img, feature = obs_to_img_feature(obs, goal, args.map_length, args.map_width, args.scale, args.img_size, args.num_ped, args.feature_dim, args.neighbor_distance)
        action, log_prob, value, n_value, g_value = agent.act(img, feature)
        obs, __, __, __ = env.step(action.reshape(-1))
        g_reward, g_coll = obs_to_global_reward(obs, prev_obs, goal, args.map_length, args.map_width, args.num_ped, args.coll_penalty, args.neighbor_distance)
        episode_return += g_reward
        episode_coll += g_coll
    env.close()
    return episode_return, episode_coll

if __name__ == '__main__':
    
    args = parser.parse_args()
    
    with open(args.name+'_args.txt', 'r') as f:
        args.__dict__ = json.load(f)
    
    print(args)
    
    # args = 
    agent = COPO(args)
    
    agent.load_ckpt(args.model_path + args.name + "/model_00000.pt")
    
    reward = eval_bypass(args, agent)
    print(reward)
    # home_path = os.path.expanduser('~')

    # channel = EngineConfigurationChannel()
    
    # print(home_path + "/Unity/first pedsim/eval_bypass.x86_64")
    
    # unity_env = UE(file_name = home_path + "/Unity/first pedsim/eval_bypass.x86_64", seed=1, side_channels=[channel], no_graphics=False)

    # env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=True)
    # channel.set_configuration_parameters(time_scale = 1.0)
    
    # for i in range(5000):
    #     obs, __, __, __ = env.step(np.array([1,0] * 301))
