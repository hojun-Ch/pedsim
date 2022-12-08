import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='pedsim')

parser.add_argument(
    '--algo',
    type=str,
    default='copo',
    help='policy algo'
)

parser.add_argument(
    '--name',
    type=str,
    default='demo',
    help='name of each trial'
)

# env path
parser.add_argument(
    '--home_path',
    type=str,
    default=os.path.expanduser('~'),
    help='path to home')

parser.add_argument(
    '--env_path',
    type=str,
    default="/Unity/first pedsim/",
    help='path from home to env folder')
# for COPO training
parser.add_argument(
    '--rendering',
    type=str2bool,
    default=False,
    help='render while training')
parser.add_argument(
    '--state_dim',
    type=int,
    default=4805,
    help='dim of state')
parser.add_argument(
    '--feature_dim',
    type=int,
    default=5,
    help='dim of feature state (x, z, theta, goal_x, goal_z)'
)
parser.add_argument(
    '--img_size',
    type=int,
    default=40,
    help='size of cropped image which represents neighborhoods'
)

parser.add_argument(
    '--action_dim',
    type=int,
    default=2,
    help='dim of action')

parser.add_argument(
    '--num_ped',
    type=int,
    default=301,
    help='# of pedestrians')

parser.add_argument(
    '--learning_rate',
    type=float,
    default=3e-4,
    help='learning rate of policy')

parser.add_argument(
    '--gae_lambda',
    type=float,
    default=0.95,
    help='lambda for calculating Generalized Advantage Estimation')

parser.add_argument(
    '--gamma',
    type=float,
    default=0.999,
    help='discounting factor')

parser.add_argument(
    '--buffer_size',
    type=int,
    default=4000,
    help='size of buffers (must same with rollout_length)')

parser.add_argument(
    '--ppo_epoch',
    type=int,
    default=5,
    help='ppo epoch for one training loop')

parser.add_argument(
    '--batch_size',
    type=int,
    default=16384,
    help='batch size for training')
parser.add_argument(
    '--ppo_clip_range',
    type=float,
    default=0.2,
    help='batch size for training')
parser.add_argument(
    '--normalize_advantages',
    type=str2bool,
    default=True,
    help='normalize advantage for stable training')
parser.add_argument(
    '--lcf_learning_rate',
    type=float,
    default=1e-4,
    help='learning rate of local coordinate factor')
parser.add_argument(
    '--lcf_epochs',
    type=int,
    default=5,
    help='lcf epoch for one training loop')
parser.add_argument(
    '--ent_coef',
    type=float,
    default=0.01,
    help='ent coef for entropy loss')
parser.add_argument(
    '--vf_coef',
    type=float,
    default=1.0,
    help='vf coef for value loss')
parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=0.5,
    help='max grad norm for gradient clipping')
parser.add_argument(
    '--device',
    type=str,
    default='cuda',
    help='learning device(gpu or cpu)'
)
parser.add_argument(
    '--rollout_length',
    type=int,
    default=4000
)
parser.add_argument(
    '--max_step',
    type=int,
    default=1000000,
    help='total env step'
)

# for network architecture

parser.add_argument(
    '--dropout',
    type=float,
    default=0.1,
    help='dropout prob')
parser.add_argument(
    '--encoder_num_hidden',
    type=int,
    default=64,
    help='dim of state encoding'
)
parser.add_argument(
    '--policy_num_hidden',
    type=int,
    default=64,
    help='hidden layer dim inside the actor&critic network'
)

# for environment
parser.add_argument(
    '--map_length',
    type=int,
    default=400,
    help='length of the entire map'
)

parser.add_argument(
    '--map_width',
    type=int,
    default=200,
    help='width of the entire map'
)

parser.add_argument(
    '--scale',
    type=int,
    default=4,
    help='scale for building occupancy map'
)

parser.add_argument(
    '--neighbor_distance',
    type=int,
    default=5,
    help='radius to define neighbor'
)

parser.add_argument(
    '--coll_penalty',
    type=float,
    default=1.0,
    help='collision penalty'
)

# for evaluation
parser.add_argument(
    '--eval_frequency',
    type=int,
    default=10,
    help='evaluation frequency'
)

parser.add_argument(
    '--eval_rendering',
    type=str2bool,
    default=False,
    help='render evaluation'
)

parser.add_argument(
    '--model_path',
    type=str,
    default='./ckpts/',
    help='path to saved model'
)
