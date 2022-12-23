
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Encoder(nn.Module):

    def __init__(
        self,
        args,
        conv_layer=3,
        mlp_layer=3,
        img_channels=3,
        
    ):
        super().__init__()
        
        dropout = args.dropout
        num_hidden = args.encoder_num_hidden
        if conv_layer == 3:
            self.image_encoder = nn.Sequential(
                    nn.Conv2d(img_channels, num_hidden // 4, 4, 2, 1),
                    nn.Dropout2d(p=dropout, inplace=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_hidden // 4, num_hidden // 2, 4, 2, 1),
                    nn.Dropout2d(p=dropout, inplace=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_hidden // 2, num_hidden, 4, 2, 1),
                    nn.Dropout2d(p=dropout, inplace=True),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            
        self.image_flatten_layer = nn.Linear((args.img_size // (2 ** conv_layer))**2, 1)
        
        if mlp_layer == 3:
            self.feature_encoder = nn.Sequential(
                    nn.Linear(args.feature_dim, num_hidden //4),
                    nn.Dropout(p=dropout, inplace=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(num_hidden//4, num_hidden // 2),
                    nn.Dropout(p=dropout, inplace=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(num_hidden // 2, num_hidden)
                )

       

    def forward(self, image, feature):
        """

        """
        image_embedding = self.image_encoder(image)
        image_embedding = image_embedding.reshape(image_embedding.shape[0], image_embedding.shape[1], -1)
        image_embedding = self.image_flatten_layer(image_embedding)
        feature_embedding = self.feature_encoder(feature)

        embedding = image_embedding + feature_embedding.unsqueeze(dim=2)
        
        return embedding.reshape(embedding.shape[0], -1)

class PolicyNetwork(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        num_hidden = args.policy_num_hidden
        obs_dimension = args.encoder_num_hidden
        action_dim = args.action_dim
        
        self.layer = nn.Sequential(
            nn.Linear(obs_dimension, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, action_dim)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim, requires_grad=True))
        
    def forward(self, obs):
        
        print(obs.shape)

        mu = self.layer(obs)
        
        normal_layer = Normal(mu, torch.exp(self.log_std))
        
        action = normal_layer.sample()
        
        log_prob = normal_layer.log_prob(action)
                
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=1)
        else:
            log_prob = log_prob.sum()
        
        return action, log_prob

    def evaluate_action(self, obs, action):
        
        mu = self.layer(obs)
        
        normal_layer = Normal(mu, torch.exp(self.log_std))
        
        log_prob = normal_layer.log_prob(action)
        
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=1)
        else:
            log_prob = log_prob.sum()
            
        entropy = normal_layer.entropy()
        
        if len(entropy.shape) > 1:
            entropy = entropy.sum(dim=1)
        else:
            entropy = entropy.sum()
        
        return log_prob, entropy
    
class PolicyNetwork_cond(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        num_hidden = args.policy_num_hidden
        obs_dimension = args.encoder_num_hidden
        action_dim = args.action_dim
        
        self.layer = nn.Sequential(
            nn.Linear(obs_dimension + 1, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, action_dim)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim, requires_grad=True))
        
    def forward(self, obs, lcf):
        
        obs = torch.cat((obs,lcf), dim=1)
        
        mu = self.layer(obs)
        
        normal_layer = Normal(mu, torch.exp(self.log_std))
        
        action = normal_layer.sample()
        
        log_prob = normal_layer.log_prob(action)
                
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=1)
        else:
            log_prob = log_prob.sum()
        
        return action, log_prob

    def evaluate_action(self, obs, action, lcf):
        
        obs = torch.cat((obs,lcf.reshape(-1,1)), dim=1)
        
        mu = self.layer(obs)
        
        normal_layer = Normal(mu, torch.exp(self.log_std))
        
        log_prob = normal_layer.log_prob(action)
        
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=1)
        else:
            log_prob = log_prob.sum()
            
        entropy = normal_layer.entropy()
        
        if len(entropy.shape) > 1:
            entropy = entropy.sum(dim=1)
        else:
            entropy = entropy.sum()
        
        return log_prob, entropy
    
class ValueNetwork(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        num_hidden = args.policy_num_hidden
        obs_dimension = args.encoder_num_hidden
        
        self.layer = nn.Sequential(
            nn.Linear(obs_dimension, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1)
        )
        
    def forward(self, obs):

        value = self.layer(obs)
        
        return value

class lcfNetwork(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor([0.0], requires_grad=True))
        self.log_std = nn.Parameter(torch.tensor([-2.3]), requires_grad=True)
        
    def forward(self, x, batch_size):

        normal_layer = Normal(self.mean, torch.exp(self.log_std))
        
        lcf = normal_layer.rsample(torch.Size([batch_size]))
                
        return lcf
    
    def get_log_prob(self, lcf):
        
        normal_layer = Normal(self.mean, torch.exp(self.log_std))
        
        log_prob = normal_layer.log_prob(lcf)
        
        return log_prob