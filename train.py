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
from agents.copo import COPO
from eval import eval_bypass, eval_crossing, eval_spread
from utils import obs_to_img_feature, obs_to_reward

import wandb

if __name__ == '__main__':
    
    # get and save args
    args = parser.parse_args()
    
    wandb.init(project="pedsim_"+args.name, reinit=True)
    
    wandb.config.update(args)
    with open(args.name+'_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    # make model directory
    if not os.path.isdir(args.model_path + args.name):
        os.mkdir(args.model_path + args.name)
        
    env_path = args.home_path + args.env_path
    
    agent = COPO(args)
    # if args.algo == 'copo':
    #     agent = COPO(args)    
    seed = range(args.max_step // args.rollout_length)
    print("total episode:", args.max_step // args.rollout_length)
    
    best_return = -1000

    for j in range(args.max_step // args.rollout_length):
        
        agent.train_mode()
        
        channel = EngineConfigurationChannel()

        unity_env = UE(file_name = env_path + "pedsim_demo.x86_64", seed=seed[j], side_channels=[channel], no_graphics=not args.rendering)

        env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=True)
        channel.set_configuration_parameters(time_scale = 16.0)
        
        obs = env.reset()
        goal = np.random.uniform(-0.9, 0.9, args.num_ped * 2)
        
        print("episode num: ", j)
        print("collecting rollout.....")

        # collect rollout
        for i in range(args.rollout_length):
            prev_obs = obs[0]
            img, feature = obs_to_img_feature(obs, goal, args.map_length, args.map_width, args.scale, args.img_size, args.num_ped, args.feature_dim, args.neighbor_distance)
            action, log_prob, value, n_value, g_value = agent.act(img, feature)
            obs, __, __, __ = env.step(action.reshape(-1))
            reward, n_reward, g_reward = obs_to_reward(obs, prev_obs, goal, args.map_length, args.map_width, args.num_ped, args.coll_penalty, args.neighbor_distance)

            if i == 0:
                agent.rollout_buffer.add(feature, img, action, reward, n_reward, g_reward, np.array([1] * args.num_ped), value, n_value, g_value, log_prob)
            elif i == args.rollout_length - 1:
                agent.rollout_buffer.add(feature, img, action, reward, n_reward, g_reward, np.array([1] * args.num_ped), value, n_value, g_value, log_prob)
            else:
                agent.rollout_buffer.add(feature, img, action, reward, n_reward, g_reward, np.array([0] * args.num_ped), value, n_value, g_value, log_prob)
        env.close()
        print("episode end!")
        
        #  compute advantage and return
        img, feature = obs_to_img_feature(obs, goal, args.map_length, args.map_width, args.scale, args.img_size, args.num_ped, args.feature_dim, args.neighbor_distance) 
        last_value, last_n_value, last_g_value = agent.get_values(img,feature)
        agent.rollout_buffer.compute_returns_and_advantage(last_value, last_n_value, last_g_value, 0)

        print("updating....")
        
        # update policy 
        lcf, pg_losses, clip_fraction, i_value_loss, n_value_loss, g_value_loss = agent.update_policy()
        
        #  update lcf
        lcf_loss = agent.update_lcf(lcf)
        
        # reset buffer 
        agent.rollout_buffer.reset()
        
        # evaluation
        if j % args.eval_frequency == 0:
            print("now on evaluation")
            
            agent.eval_mode()
            
            bypass_return = eval_bypass(args, agent)
            print("bypass_return:", bypass_return)
            crossing_return = eval_crossing(args, agent)
            print("crossing_return:", bypass_return)
            spread_return = eval_spread(args, agent)
            print("spread_return:", bypass_return)
            
            agent.save_model(args.model_path + args.name + "/model_%05d.pt"%j)
            
            if bypass_return + crossing_return + spread_return > best_return:
                agent.save_model(args.model_path + args.name + "/model_best.pt")
        
        # logging
        
        for i in range(len(pg_losses)):
            wandb.log({
                'policy losses': pg_losses[i],
                'clip fractions': clip_fraction[i],
                'independent value losses': i_value_loss[i],
                'neighbor value losses':n_value_loss[i],
                'global value losses': g_value_loss[i],
                'LCF losses': sum(lcf_loss) / len(lcf_loss),
                'bypass eval returns': bypass_return,
                'crossing eval returns': crossing_return,
                'spread eval returns': spread_return
            })

        

    env.close()



