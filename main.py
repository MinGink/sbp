from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from algos.procgen_torch_layers import ProcgenCNN

import wandb
from wandb.integration.sb3 import WandbCallback

import numpy as np
import random
import os
import argparse
import time
from os.path import dirname, abspath
import torch
import datetime
import gym
# import gymnasium as gym

from stable_baselines3 import A2C, PPO
from algos import ACH


def make_env(config):
    env = gym.make(config["env"])
    env = Monitor(env)  # record stats such as returns
    return env


def make_procgen_env(config):
    from procgen import ProcgenEnv
    from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor, VecNormalize

    env = ProcgenEnv(num_envs = 64,
                     env_name=config["env"],
                     num_levels=0,
                     start_level=0,
                     distribution_mode="easy",
                     num_threads=config["n_threads"],
                     )

    env = VecExtractDictObs(env, "rgb")
    env = VecMonitor(venv=env, filename=None)
    env = VecNormalize(venv=env)

    return env



def main(args):

    if args.seed != None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)



    if args.gpu is not None and args.gpu >= 0:
        assert torch.cuda.is_available()
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device('cpu')


        #======================================wandb==============================================
    log_path = os.path.join(dirname(abspath(__file__)), "logs")
    args.exp_name = f"{args.env}_{args.algo}_{args.memo}" # _{datetime.datetime.now().strftime('%d_%H_%M')}
    tags_list = [args.env, args.algo]

    #change args to config dict:
    config_dict = vars(args)


    if args.offline_wandb:
        os.environ['WANDB_MODE'] = 'dryrun'


    logger = wandb.init(project="ICLR",
                     name=args.exp_name,
                     config=config_dict,
                     tags=tags_list,
                     dir=log_path,
                     entity="mingatum",
                     sync_tensorboard=True,  # auto-upload stable_baselines3's tensorboard metrics
                     monitor_gym=False,  # auto-upload the videos of agents playing the game
                     save_code=True, )



    if args.env_type == "atari":
        vec_env = make_atari_env(env_id= config_dict['env'], n_envs=config_dict['n_threads'], seed=config_dict['seed'])
        vec_env = VecFrameStack(vec_env, n_stack=4)
        policy_kwargs = dict()
    elif args.env_type == "procgen":
        vec_env = make_procgen_env(config_dict)

        policy_kwargs = dict(features_extractor_class=ProcgenCNN,
                            normalize_images=False)

    else:
        raise NotImplementedError (f"env_type {args.env_type} not implemented")



    if args.algo == "ACH":
        model = ACH("CnnPolicy",
                        vec_env,
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        gamma=config_dict['gamma'],
                        tensorboard_log=f"logs/{logger.id}",
                        device=args.device,
                        seed=config_dict['seed'],
                     )

    elif args.algo == "PPO":
        model = PPO("CnnPolicy",
                        vec_env,
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        gamma=config_dict['gamma'],
                        tensorboard_log=f"logs/{logger.id}",
                        device=args.device,
                        seed=config_dict['seed']
                     )
    else:
        raise NotImplementedError(f"algo {args.algo} not implemented")


    # Add WandBCallback to the training algorithm
    model.learn(total_timesteps=config_dict['total_timesteps'],
                callback=WandbCallback(model_save_path=f"models/{logger.id}",
                                       gradient_save_freq=0,
                                       verbose=2,
                                       ),
                )


    logger.finish()
    print("=====Done!!!=====")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AAAI')

    parser.add_argument('--memo', type=str, default="debug", help='memo name')
    parser.add_argument('--env_type', type=str, default="procgen", help='environment type',choices=['atari','procgen'])
    parser.add_argument('--env', type=str, default="starpilot", help='environment name')
    parser.add_argument("--gpu", type=int, default=-1, help='use cpu set -1')

    parser.add_argument('--algo', type=str, default="PPO", help='algorithm name',choices=['PPO','ACH'])
    parser.add_argument('--variant', type=str, default="random", help='variants name',choices=['random','decrease','increase','fixed','default'])

    parser.add_argument('--seed', type=int, default=None, help='random seed, if seed < 0, dont use torch.manual_seed')
    parser.add_argument('--offline_wandb', action='store_true', help='use offline wandb')
    parser.add_argument('--n_threads', type=int, default=4, help='number of envs for multiprocessing')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='gae_lambda')
    parser.add_argument('--total_timesteps', type=int, default=2000000, help='total number of training timesteps')


    args = parser.parse_args()


    training_begin_time = time.time()
    main(args)
    training_time = time.time() - training_begin_time
    print('Training time: {} h'.format(training_time/3600))