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
from models.model import Encoder, PolicyNetwork, ValueNetwork, lcfNetwork
from buffers.buffers import RolloutBuffer


class COPO():
    
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
        self.lcf_learning_rate = args.lcf_learning_rate
        self.lcf_epochs = args.lcf_epochs
        self.ent_coef = args.ent_coef
        self.vf_coef = args.vf_coef
        self.max_grad_norm = args.max_grad_norm
        self.img_size = args.img_size
        self.train_lcf = args.train_lcf
        
        # rollout buffer
        self.rollout_buffer = RolloutBuffer(self.buffer_size, self.feature_dim, self.img_size, self.action_dim, device=args.device, gae_lambda=self.gae_lambda, gamma=self.gamma, n_envs=self.n_envs)
        
        # networks
        self.state_encoder = Encoder(args)
        self.policy_network = PolicyNetwork(args)
        self.i_value_network = ValueNetwork(args)
        self.n_value_network = ValueNetwork(args)
        self.g_value_network = ValueNetwork(args)
        
        self.LCF_dist = lcfNetwork(args)
        self.device = torch.device(args.device)
        self.state_encoder.to(self.device)
        self.policy_network.to(self.device)
        self.i_value_network.to(self.device)
        self.n_value_network.to(self.device)
        self.g_value_network.to(self.device)
        self.LCF_dist.to(self.device)
        self.LCF_dist.eval()

                
        # optimizers
        entire_parameters = list(self.policy_network.parameters()) + list(self.i_value_network.parameters())\
                            + list(self.n_value_network.parameters()) + list(self.g_value_network.parameters()) + list(self.state_encoder.parameters())
        self.optimizer = torch.optim.Adam(entire_parameters, lr=self.learning_rate)
        # self.encoder_optimizer = torch.optim.Adam(self.state_encoder.parameters(), lr=self.learning_rate)
        # self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        # self.i_value_optimizer = torch.optim.Adam(self.i_value_network.parameters(), lr=self.learning_rate)
        # self.n_value_optimizer = torch.optim.Adam(self.n_value_network.parameters(), lr=self.learning_rate)
        # self.g_value_optimizer = torch.optim.Adam(self.g_value_network.parameters(), lr=self.learning_rate)
        
        self.lcf_optimizer = torch.optim.Adam(self.LCF_dist.parameters(), lr=self.lcf_learning_rate)

    def evaluate_actions(self, feature, img, actions):
        
        # img= torch.from_numpy(img.astype(np.float32))
        # feature  = torch.from_numpy(feature.astype(np.float32))
        obs = self.state_encoder(img, feature)
            
        i_value = self.i_value_network(obs)
        n_value = self.n_value_network(obs)
        g_value = self.g_value_network(obs)
        
        log_prob, entropy = self.policy_network.evaluate_action(obs, actions)
            
        return i_value, n_value, g_value, log_prob, entropy
    
    def evaluate_actions_with_old(self, feature, img, actions):
        # img= torch.from_numpy(img.astype(np.float32))
        # feature  = torch.from_numpy(feature.astype(np.float32))
        
        obs = self.old_state_encoder(img, feature)
        
        log_prob, __ = self.old_policy_network.evaluate_action(obs, actions)
        
        return log_prob
        
        
    def act(self, img, feature):
        """
        """
        with torch.no_grad():
            img= torch.from_numpy(img.astype(np.float32))
            feature  = torch.from_numpy(feature.astype(np.float32))
            
            img = img.to(self.device)
            feature = feature.to(self.device)

            obs = self.state_encoder(img, feature)
            
            action, log_prob = self.policy_network(obs)
            
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
        self.old_policy_network = copy.deepcopy(self.policy_network)
        self.old_state_encoder = copy.deepcopy(self.state_encoder)
        self.old_optimizer = torch.optim.Adam(list(self.old_policy_network.parameters()) + list(self.old_state_encoder.parameters()), lr=self.learning_rate)
        
        pg_losses = []
        clip_fractions = []
        entropy_losses = []
        value_losses = []
        n_value_losses = []
        g_value_losses = []
        
        for epoch in range(self.ppo_epoch):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                values, n_values, g_values, log_prob, entropy = self.evaluate_actions(rollout_data.observations, rollout_data.surroundings, actions)
                values = values.flatten()
                n_values = n_values.flatten()
                g_values = g_values.flatten()
                
                # sample lcf
                lcf = self.LCF_dist(torch.tensor([0.0]).to(self.device), n_values.shape[0])
                lcf = lcf.view(-1)
                lcf = torch.clamp(lcf, -math.pi / 2, math.pi / 2)
                lcf_clone = lcf.detach()
                
                # Normalize advantage
                i_advantages = rollout_data.i_advantages
                n_advantages = rollout_data.n_advantages
                
                if self.normalize_advantage:
                    i_advantages = (i_advantages - i_advantages.mean()) / (i_advantages.std() + 1e-8)
                    n_advantages = (n_advantages - n_advantages.mean()) / (n_advantages.std() + 1e-8)
                    
                coord_advantages = torch.cos(lcf_clone) * i_advantages + torch.sin(lcf_clone) * n_advantages
                # ratio between old and new policy, should be one at the first iteration
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
    
    def update_lcf(self):
        losses = []
        clip_range = self.clip_range
        for epoch in range(self.lcf_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                
                actions = rollout_data.actions

                __, __, __, log_prob, __ = self.evaluate_actions(rollout_data.observations, rollout_data.surroundings, actions)
                
                
                
                # Normalize advantage
                i_advantages = rollout_data.i_advantages
                n_advantages = rollout_data.n_advantages
                g_advantages = rollout_data.g_advantages
                
                if self.normalize_advantage:
                    i_advantages = (i_advantages - i_advantages.mean()) / (i_advantages.std() + 1e-8)
                    n_advantages = (n_advantages - n_advantages.mean()) / (n_advantages.std() + 1e-8)
                    g_advantages = (g_advantages - g_advantages.mean()) / (g_advantages.std() + 1e-8)
                
                # sample lcf
                lcf = self.LCF_dist(torch.tensor([0.0]).to(self.device), i_advantages.shape[0])
                lcf = lcf.view(-1)
                lcf = torch.clamp(lcf, -math.pi / 2, math.pi / 2)
                    
                coord_advantages = torch.cos(lcf) * i_advantages + torch.sin(lcf) * n_advantages
                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = g_advantages * ratio
                
                policy_loss_2 = g_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                log_prob_from_old_policy = self.evaluate_actions_with_old(rollout_data.observations, rollout_data.surroundings, actions)
                
                old_policy_loss = -log_prob_from_old_policy.mean()
                
                self.optimizer.zero_grad()
                self.old_optimizer.zero_grad()
                
                policy_loss.backward()
                old_policy_loss.backward()
                
                # 1st term 
                grad = torch.tensor([0.0]).to(self.device)
                
                for param in self.state_encoder.parameters():
                    grad = torch.cat((grad, param.grad.view(-1)), 0)
                for param in self.policy_network.parameters():
                    grad = torch.cat((grad, param.grad.view(-1)), 0)
                    
                
                # 2nd term
                old_grad = torch.tensor([0.0]).to(self.device)
                
                for param in self.old_state_encoder.parameters():
                    old_grad = torch.cat((old_grad, param.grad.view(-1)), 0)
                for param in self.old_policy_network.parameters():
                    old_grad = torch.cat((old_grad, param.grad.view(-1)), 0)
                
                # lcf probability
                # lcf_log_prob = self.LCF_dist.get_log_prob(lcf)
                
                # lcf loss
                lcf_loss = - torch.dot(grad, old_grad) * coord_advantages
                lcf_loss = lcf_loss.mean()
                lcf_loss.requires_grad_(True)
                losses.append(lcf_loss.item())
                
                # Optimization step
                self.lcf_optimizer.zero_grad()
                lcf_loss.backward()
                self.lcf_optimizer.step()
                
                self.optimizer.zero_grad()
                self.old_optimizer.zero_grad()

                
        return losses
                
    def train_mode(self):
        self.state_encoder.train()
        self.policy_network.train()
        self.i_value_network.train()
        self.n_value_network.train()
        self.g_value_network.train()
        if self.train_lcf:
            self.LCF_dist.train()
        
    def eval_mode(self):
        self.state_encoder.eval()
        self.policy_network.eval()
        self.i_value_network.eval()
        self.n_value_network.eval()
        self.g_value_network.eval()
        self.LCF_dist.eval()
        
    def save_model(self, path):
        torch.save({
            'encoder_state_dict':self.state_encoder.state_dict(),
            'policy_state_dict':self.policy_network.state_dict(),
            'i_value_state_dict':self.i_value_network.state_dict(),
            'n_value_state_dict':self.n_value_network.state_dict(),
            'g_value_state_dict':self.g_value_network.state_dict(),
            'lcf_state_dict':self.LCF_dist.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'lcf_optimizer_state_dict':self.lcf_optimizer.state_dict()
        }, path)
        
    def load_ckpt(self, model_path):
        
        ckpt = torch.load(model_path)
        self.state_encoder.load_state_dict(ckpt['encoder_state_dict'])
        self.policy_network.load_state_dict(ckpt['policy_state_dict'])
        self.i_value_network.load_state_dict(ckpt['i_value_state_dict'])
        self.n_value_network.load_state_dict(ckpt['n_value_state_dict'])
        self.g_value_network.load_state_dict(ckpt['g_value_state_dict'])
        
        self.LCF_dist.load_state_dict(ckpt['lcf_state_dict'])
        
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        self.lcf_optimizer.load_state_dict(ckpt['lcf_optimizer_state_dict'])

        return
        