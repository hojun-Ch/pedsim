import cv2
import numpy as np
import torch
import math
import matplotlib.pyplot as plt

from mlagents_envs.environment import UnityEnvironment as UE
from gym_unity.envs import UnityToGymWrapper

import time
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os
from arguments import parser
from agents.copo import COPO


red_color = (1.0,0.0,0.0)
green_color = (0.0,1.0,0.0)
blue_color = (0.0,0.0,1.0)
white_color = (1.0,1.0,1.0)
black_color = (0.0,0.0,0.0)

def dist_to_reward(distance, max_distance):
    
    return (max_distance/2 - distance) / (max_distance/2)

def pos_to_pixel(x, z, scale = 4, l=200, w=100, pad=20):
    u = round((x + l) * scale) + pad
    v = round((-z + w) * scale) + pad
    
    return u, v

def obs_to_img_feature(obs, goal, l, w, scale, img_size, num_ped, feature_dim, distance):
    
    #  create entire map
    img = np.zeros((scale * w + img_size, scale * l + img_size,3), np.float32)

    # mark red on obstacles(wall)
    img[:img_size//2, :] = red_color
    img[-img_size//2:, :] = red_color
    img[:, :img_size//2] = red_color
    img[:, -img_size//2:] = red_color

    # mark blue on pedestrians
    for i in range(num_ped):
        
        x = obs[0][5*i]
        z = obs[0][5*i + 1]
        theta = obs[0][5*i + 2]
        vel = obs[0][5*i + 3]
        vel_quantized = 1
        
        if 1.66 <= vel < 3.32:
            vel_quantized = 2
        elif vel >= 3.32:
            vel_quantized = 3
            
        
        u, v = pos_to_pixel(x,z, scale=scale, l=l//2, w=w//2, pad=img_size//2)
        
        img = cv2.ellipse(img, (u,v), (2 + vel_quantized,2), theta, -90, 90, green_color, -1)
        img = cv2.circle(img, (u,v), 2, blue_color, -1)

    image_batch = np.zeros((num_ped, img_size, img_size, 3), dtype=np.float32)
    feature = np.zeros((num_ped, feature_dim), dtype=np.float32)
    for i in range(num_ped):
        
        x = obs[0][5*i]
        z = obs[0][5*i + 1]
        theta = obs[0][5*i + 2]
        vel = obs[0][5*i + 3]
        
        u, v = pos_to_pixel(x,z, scale=scale, l=l//2, w=w//2, pad=img_size//2)
        
        pixel_distance = scale * distance
        image_batch[i] = img[v-pixel_distance:v+pixel_distance, u-pixel_distance:u+pixel_distance]
        
        feature[i][0] = x / (l//2)
        feature[i][1] = z / (w//2)
        feature[i][2] = (theta - 180) / 360
        feature[i][3] = goal[2*i] 
        feature[i][4] = goal[2*i+1] 
    return image_batch.reshape((args.num_ped, 3, args.img_size, args.img_size)), feature 

def obs_to_reward(obs, goal,  l, w, num_ped, coll_penalty, neighbor_distance):
    reward = np.zeros(num_ped)
    n_reward = np.zeros(num_ped)
    g_reward = np.ones(num_ped)
    neighbor_num = np.zeros(num_ped)
    max_distance = math.sqrt(l**2 + w**2)
    
    for i in range(num_ped):
    
        x = obs[0][5*i] 
        z = obs[0][5*i + 1] 
        coll = obs[0][5*i + 4]
        
        goal_x = goal[2*i] * (l//2)
        goal_z = goal[2*i + 1] * (w//2)
        
        distance_to_goal = math.sqrt((goal_x - x)**2 + (goal_z - z) ** 2)        
        
        reward[i] = dist_to_reward(distance_to_goal, max_distance) - coll * coll_penalty
        
    for i in range(num_ped):
        for j in range(i+1, num_ped):
            x1 = obs[0][5*i]
            z1 = obs[0][5*i + 1] 
            x2 = obs[0][5*j]
            z2 = obs[0][5*j + 1]
            
            if abs(x1 - x2) < neighbor_distance and abs(z1 - z2) < neighbor_distance:
                n_reward[i] += reward[j]
                n_reward[j] += reward[i]
                neighbor_num[i] += 1
                neighbor_num[j] += 1
    
    n_reward = n_reward / (neighbor_num +1e-8)
    
    g_reward *= reward.sum() / num_ped
    
    return reward, n_reward, g_reward

if __name__ == '__main__':
    
    args = parser.parse_args()
    
    home_path = os.path.expanduser('~')

    channel = EngineConfigurationChannel()

    unity_env = UE(file_name = home_path + "/Unity/first pedsim/pedsim_demo.x86_64", seed=1, side_channels=[channel], no_graphics=True)

    env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=True)
    channel.set_configuration_parameters(time_scale = 16.0)
    agent = COPO(args)
    # if args.algo == 'copo':
    #     agent = COPO(args)
    for j in range(args.max_step // args.rollout_length):
        obs = env.reset()
        goal = np.random.uniform(-0.9, 0.9, args.num_ped * 2)
        
        for i in range(args.rollout_length):
            img, feature = obs_to_img_feature(obs, goal, args.map_length, args.map_width, args.scale, args.img_size, args.num_ped, args.feature_dim, args.neighbor_distance)
            action, log_prob, value, n_value, g_value = agent.act(img, feature)
            obs, __, __, __ = env.step(action.reshape(-1))
            reward, n_reward, g_reward = obs_to_reward(obs, goal, args.map_length, args.map_width, args.num_ped, args.coll_penalty, args.neighbor_distance)
            if i == 0:
                agent.rollout_buffer.add(feature, img, action, reward, n_reward, g_reward, np.array([1] * args.num_ped), value, n_value, g_value, log_prob)
            else:
                agent.rollout_buffer.add(feature, img, action, reward, n_reward, g_reward, np.array([0] * args.num_ped), value, n_value, g_value, log_prob)

        img, feature = obs_to_img_feature(obs, goal, args.map_length, args.map_width, args.scale, args.img_size, args.num_ped, args.feature_dim, args.neighbor_distance) 
        last_value, last_n_value, last_g_value = agent.get_values(img,feature)
        agent.rollout_buffer.compute_returns_and_advantage(last_value, last_n_value, last_g_value, 0)

            
        lcf, pg_losses, clip_fractions, g_value_losses, n_value_losses, g_value_losses = agent.update_policy()
        
        losses = agent.update_lcf(lcf)
        
        print("pg_losses: ", pg_losses)
        
        agent.rollout_buffer.reset()

    env.close()



