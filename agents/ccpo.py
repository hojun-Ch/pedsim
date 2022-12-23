import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import copy

from mlagents_envs.environment import UnityEnvironment as UE
from gym_unity.envs import UnityToGymWrapper

from gym.spaces import Dict, Box
import time
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal
from models.model import Encoder, PolicyNetwork_cond, ValueNetwork, lcfNetwork
from buffers.buffers import RolloutBuffer


class CCPO():
    
    def __init__(self, args):
        
        self.state_dim = args.state_dim
        self.feature_dim = args.feature_dim
        self.action_dim = args.action_dim
        self.n_envs = args.num_ped
        self.learning_rate = args.learning_rate
        self.gae_lambda = args.gae_lambda
        self.gamma = args.gamma
        self.buffer_size = args.buffer_size
        self.ppo_epoch = args.ppo_epoch
        self.batch_size = args.batch_size
        self.n_updates = 0
        self.clip_range = args.ppo_clip_range
        self.normalize_advantage = args.normalize_advantages
        self.ent_coef = args.ent_coef
        self.vf_coef = args.vf_coef
        self.max_grad_norm = args.max_grad_norm
        self.img_size = args.img_size
        
        # rollout buffer
        self.rollout_buffer = RolloutBuffer(self.buffer_size, self.feature_dim, self.img_size, self.action_dim, device=args.device, gae_lambda=self.gae_lambda, gamma=self.gamma, n_envs=self.n_envs)
        
        # networks
        self.state_encoder = Encoder(args)
        self.policy_network = PolicyNetwork_cond(args)
        self.i_value_network = ValueNetwork(args)
        self.n_value_network = ValueNetwork(args)
        self.g_value_network = ValueNetwork(args)
        
        self.device = torch.device(args.device)
        self.state_encoder.to(self.device)
        self.policy_network.to(self.device)
        self.i_value_network.to(self.device)
        self.n_value_network.to(self.device)
        self.g_value_network.to(self.device)


                
        # optimizers
        entire_parameters = list(self.policy_network.parameters()) + list(self.i_value_network.parameters())\
                            + list(self.n_value_network.parameters()) + list(self.g_value_network.parameters()) + list(self.state_encoder.parameters())
        self.optimizer = torch.optim.Adam(entire_parameters, lr=self.learning_rate)
        # self.encoder_optimizer = torch.optim.Adam(self.state_encoder.parameters(), lr=self.learning_rate)
        # self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        # self.i_value_optimizer = torch.optim.Adam(self.i_value_network.parameters(), lr=self.learning_rate)
        # self.n_value_optimizer = torch.optim.Adam(self.n_value_network.parameters(), lr=self.learning_rate)
        # self.g_value_optimizer = torch.optim.Adam(self.g_value_network.parameters(), lr=self.learning_rate)
        

    def evaluate_actions(self, feature, img, actions, lcf):
        
        # img= torch.from_numpy(img.astype(np.float32))
        # feature  = torch.from_numpy(feature.astype(np.float32))
        obs = self.state_encoder(img, feature)
            
        i_value = self.i_value_network(obs)
        n_value = self.n_value_network(obs)
        g_value = self.g_value_network(obs)

        log_prob, entropy = self.policy_network.evaluate_action(obs, actions, lcf)
            
        return i_value, n_value, g_value, log_prob, entropy
    
        
        
    def act(self, img, feature, lcf):
        """
        """
        with torch.no_grad():
            img= torch.from_numpy(img.astype(np.float32))
            feature  = torch.from_numpy(feature.astype(np.float32))
            lcf = torch.from_numpy(lcf.reshape((-1,1)).astype(np.float32))
            
            img = img.to(self.device)
            feature = feature.to(self.device)
            lcf = lcf.to(self.device)

            obs = self.state_encoder(img, feature)
            
            action, log_prob = self.policy_network(obs, lcf)
            
            value = self.i_value_network(obs)
            n_value = self.n_value_network(obs)
            g_value = self.g_value_network(obs)
            
        
        return action.clone().cpu().numpy(), log_prob, value, n_value, g_value
    
    def get_values(self, img, feature):
        """
        """
        with torch.no_grad():
            img= torch.from_numpy(img.astype(np.float32))
            feature  = torch.from_numpy(feature.astype(np.float32))
            
            img = img.to(self.device)
            feature = feature.to(self.device)
            
            obs = self.state_encoder(img, feature)
                        
            value = self.i_value_network(obs)
            n_value = self.n_value_network(obs)
            g_value = self.g_value_network(obs)
        
        return value, n_value, g_value

    def update_policy(self):
        clip_range = self.clip_range
        
        pg_losses = []
        clip_fractions = []
        entropy_losses = []
        value_losses = []
        n_value_losses = []
        g_value_losses = []
        
        for epoch in range(self.ppo_epoch):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                lcf = rollout_data.lcf

                values, n_values, g_values, log_prob, entropy = self.evaluate_actions(rollout_data.observations, rollout_data.surroundings, actions, lcf)
                values = values.flatten()
                n_values = n_values.flatten()
                g_values = g_values.flatten()
                
                # Normalize advantage
                i_advantages = rollout_data.i_advantages
                n_advantages = rollout_data.n_advantages
                
                if self.normalize_advantage:
                    i_advantages = (i_advantages - i_advantages.mean()) / (i_advantages.std() + 1e-8)
                    n_advantages = (n_advantages - n_advantages.mean()) / (n_advantages.std() + 1e-8)
                    
                coord_advantages = torch.cos(lcf) * i_advantages + torch.sin(lcf) * n_advantages
                # ratio between old and new policy, should be one at the first iterations
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = coord_advantages * ratio
                policy_loss_2 = coord_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                values_pred = values
                n_values_pred = n_values
                g_values_pred = g_values
                    
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.i_returns, values_pred)
                value_losses.append(value_loss.item())
                
                n_value_loss = F.mse_loss(rollout_data.n_returns, n_values_pred)
                n_value_losses.append(n_value_loss.item())
                
                g_value_loss = F.mse_loss(rollout_data.g_returns, g_values_pred)
                g_value_losses.append(g_value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * (value_loss + n_value_loss + g_value_loss)

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()

                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                                
        self.n_updates += self.ppo_epoch
        return pg_losses, clip_fractions, value_losses, n_value_losses, g_value_losses
                
    def train_mode(self):
        self.state_encoder.train()
        self.policy_network.train()
        self.i_value_network.train()
        self.n_value_network.train()
        self.g_value_network.train()

        
    def eval_mode(self):
        self.state_encoder.eval()
        self.policy_network.eval()
        self.i_value_network.eval()
        self.n_value_network.eval()
        self.g_value_network.eval()
        
    def save_model(self, path):
        torch.save({
            'encoder_state_dict':self.state_encoder.state_dict(),
            'policy_state_dict':self.policy_network.state_dict(),
            'i_value_state_dict':self.i_value_network.state_dict(),
            'n_value_state_dict':self.n_value_network.state_dict(),
            'g_value_state_dict':self.g_value_network.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
        }, path)
        
    def load_ckpt(self, model_path):
        
        ckpt = torch.load(model_path)
        self.state_encoder.load_state_dict(ckpt['encoder_state_dict'])
        self.policy_network.load_state_dict(ckpt['policy_state_dict'])
        self.i_value_network.load_state_dict(ckpt['i_value_state_dict'])
        self.n_value_network.load_state_dict(ckpt['n_value_state_dict'])
        self.g_value_network.load_state_dict(ckpt['g_value_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        