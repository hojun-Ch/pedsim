import cv2
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import json

from mlagents_envs.environment import UnityEnvironment as UE
from gym_unity.envs import UnityToGymWrapper

import time
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os
from arguments import parser
from agents.ccpo import CCPO
from eval_ccpo import eval_bypass, eval_crossing, eval_spread
from utils import obs_to_img_feature, obs_to_reward, obs_to_reward_coll_smoothed

import wandb

if __name__ == '__main__':
    
    # get and save args
    args = parser.parse_args()
    
    wandb.init(project="pedsim_demo", reinit=True, entity="hojun-chung")
    wandb.run.name = args.name
    
    wandb.config.update(args)
    with open(args.name+'_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    # make model directory
    if not os.path.isdir(args.model_path + args.name):
        os.mkdir(args.model_path + args.name)
        
    env_path = args.home_path + args.env_path
    
    agent = CCPO(args)
    # if args.algo == 'copo':
    #     agent = COPO(args)    
    seed = range(args.max_step // args.rollout_length)
    print("total episode:", args.max_step // args.rollout_length)
    
    best_return = -1000

    for j in range(args.max_step // args.rollout_length):
        
        agent.train_mode()
        
        channel = EngineConfigurationChannel()

        unity_env = UE(file_name = env_path + "pedsim_demo.x86_64", seed=seed[j], side_channels=[channel], no_graphics=not args.rendering, worker_id = args.worker)

        env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=True)
        channel.set_configuration_parameters(time_scale = 1.0)
        
        obs = env.reset()
        goal = np.random.uniform(-0.9, 0.9, args.num_ped * 2)
        
        print("episode num: ", j)
        print("collecting rollout.....")

        # collect rollout
        train_global_return = 0
        train_collision = 0
        lcf = np.random.uniform(-math.pi / 2, math.pi / 2, args.num_ped)

        
        for i in range(args.rollout_length):
            prev_obs = obs[0]
            img, feature = obs_to_img_feature(obs, goal, args.map_length, args.map_width, args.scale, args.img_size, args.num_ped, args.feature_dim, args.neighbor_distance)
            action, log_prob, value, n_value, g_value = agent.act(img, feature, lcf)
            obs, __, __, __ = env.step(action.reshape(-1))
            if args.smooth_cost:
                reward, n_reward, g_reward, global_reward_wo_coll, global_coll = obs_to_reward_coll_smoothed(obs, prev_obs, goal, args.map_length, args.map_width, args.num_ped, args.coll_penalty, args.neighbor_distance)
            else:
                reward, n_reward, g_reward, global_reward_wo_coll, global_coll = obs_to_reward(obs, prev_obs, goal, args.map_length, args.map_width, args.num_ped, args.coll_penalty, args.neighbor_distance)
            train_global_return += global_reward_wo_coll
            train_collision +=  global_coll
            if i == 0:
                agent.rollout_buffer.add(feature, img, action, reward, n_reward, g_reward, np.array([1] * args.num_ped), value, n_value, g_value, log_prob, lcf)
            else:
                agent.rollout_buffer.add(feature, img, action, reward, n_reward, g_reward, np.array([0] * args.num_ped), value, n_value, g_value, log_prob, lcf)
        env.close()
        print("episode end!")
        
        #  compute advantage and return
        img, feature = obs_to_img_feature(obs, goal, args.map_length, args.map_width, args.scale, args.img_size, args.num_ped, args.feature_dim, args.neighbor_distance) 
        last_value, last_n_value, last_g_value = agent.get_values(img,feature)
        agent.rollout_buffer.compute_returns_and_advantage(last_value, last_n_value, last_g_value, 0)

        print("updating....")
        
        # update policy 
        pg_losses, clip_fraction, i_value_loss, n_value_loss, g_value_loss = agent.update_policy()
        
        # reset buffer 
        agent.rollout_buffer.reset()
        
        # evaluation
        if j % args.eval_frequency == 0:
            print("now on evaluation")
            
            agent.eval_mode()
            
            bypass_return, bypass_coll = eval_bypass(args, agent)
            crossing_return, crossing_coll = eval_crossing(args, agent)
            spread_return, spread_coll = eval_spread(args, agent)
            
            
            if bypass_return + crossing_return + spread_return > best_return:
                best_return = bypass_return + crossing_return + spread_return
                agent.save_model(args.model_path + args.name + "/model_best.pt")
        
        # logging
        
        if j % args.archive_frequency == 0:
            agent.save_model(args.model_path + args.name + "/model_%05d.pt"%j)
        
        for i in range(len(pg_losses)):
            wandb.log({
                'policy losses': pg_losses[i],
                'clip fractions': clip_fraction[i],
                'independent value losses': i_value_loss[i],
                'neighbor value losses':n_value_loss[i],
                'global value losses': g_value_loss[i],
                'bypass eval returns': bypass_return,
                'bypass_eval_collisions': bypass_coll,
                'crossing eval returns': crossing_return,
                "crossing_eval_collisions": crossing_coll,
                'spread eval returns': spread_return,
                'spread_eval_collisions': spread_coll,
                'train_return': train_global_return,
                'train_collision': train_collision
            })

        

    env.close()



