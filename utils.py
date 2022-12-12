import cv2
import numpy as np
import torch
import math

red_color = (1.0,0.0,0.0)
green_color = (0.0,1.0,0.0)
blue_color = (0.0,0.0,1.0)
white_color = (1.0,1.0,1.0)
black_color = (0.0,0.0,0.0)

def dist_to_reward(distance, prev_distance, max_distance, neighbor_distance):
    reward = (prev_distance - distance)
    
    if distance < neighbor_distance:
        reward += (neighbor_distance - distance) / neighbor_distance
        
    if distance < 0.5:
        reward = 2.0
        
    return reward

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
    return image_batch.reshape((num_ped, 3, img_size, img_size)), feature 

def obs_to_reward(obs, prev_obs, goal,  l, w, num_ped, coll_penalty, neighbor_distance):
    reward = np.zeros(num_ped)
    g_reward = np.ones(num_ped)
    n_reward = np.zeros(num_ped)
    neighbor_num = np.ones(num_ped)
    no_coll_reward = np.ones(num_ped)
    collision = np.zeros(num_ped)
    
    max_distance = math.sqrt(l**2 + w**2)
    
    for i in range(num_ped):
        
        x = obs[0][5*i] 
        z = obs[0][5*i + 1] 
        coll = obs[0][5*i + 4]
        
        x_0 = prev_obs[5*i]
        z_0 = prev_obs[5*i + 1]
        
        goal_x = goal[2*i] * (l//2)
        goal_z = goal[2*i + 1] * (w//2)
        
        prev_distance = math.sqrt((goal_x - x_0)**2 + (goal_z - z_0) ** 2) 
        distance_to_goal = math.sqrt((goal_x - x)**2 + (goal_z - z) ** 2)        
        
        reward[i] = dist_to_reward(distance_to_goal, prev_distance, max_distance, neighbor_distance) - coll * coll_penalty
        no_coll_reward[i] = dist_to_reward(distance_to_goal, prev_distance, max_distance, neighbor_distance)
        collision[i] = coll
        
    for i in range(num_ped):
        n_reward[i] += reward[i]
        
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
    
    global_reward_wo_coll = no_coll_reward.sum() / num_ped
    g_coll = collision.sum() / num_ped
    return reward, n_reward, g_reward, global_reward_wo_coll, g_coll

def obs_to_global_reward(obs, prev_obs, goal,  l, w, num_ped, coll_penalty, neighbor_distance):
    reward = np.zeros(num_ped)
    collision = np.zeros(num_ped)
    max_distance = math.sqrt(l**2 + w**2)
    
    for i in range(num_ped):
        
        x = obs[0][5*i] 
        z = obs[0][5*i + 1] 
        coll = obs[0][5*i + 4]
        
        x_0 = prev_obs[5*i]
        z_0 = prev_obs[5*i + 1]
        
        goal_x = goal[2*i] * (l//2)
        goal_z = goal[2*i + 1] * (w//2)
        
        prev_distance = math.sqrt((goal_x - x_0)**2 + (goal_z - z_0) ** 2) 
        distance_to_goal = math.sqrt((goal_x - x)**2 + (goal_z - z) ** 2)        
        
        reward[i] = dist_to_reward(distance_to_goal, prev_distance, max_distance, neighbor_distance)
        collision[i] = coll
        
    
    g_reward = reward.sum() / num_ped
    g_coll = collision.sum() / num_ped
    
    return g_reward, g_coll