"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

def main():
    train_env = gym.make('CustomHopper-source-v0')

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

    """
    # PPO model
    train_env = DummyVecEnv([lambda: train_env]) # Wrap the environment
    model = PPO("MlpPolicy", train_env, verbose=1) # Initialize the PPO model
    model.learn(total_timesteps=100000) # Train the model
    ppo_path = os.path.join("models", "CustomHopper_source_v0_PPO_100k") # Path to save the model
    model.save(ppo_path) # Save the model

    evaluate_policy(model, train_env, n_eval_episodes=10, render=True) # Evaluate the model
    """

    """
    train_env = DummyVecEnv([lambda: train_env]) # Wrap the environment
    ppo_path = os.path.join("models", "CustomHopper_source_v0_PPO_100k") # Path to save the model
    model = PPO.load(ppo_path, train_env) # Load the model
    evaluate_policy(model, train_env, n_eval_episodes=30, render=True)
    """

    """
    # SAC model
    train_env = DummyVecEnv([lambda: train_env])
    model = SAC("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=200000)

    sac_path = os.path.join("models", "CustomHopper_source_v0_SAC_200k")
    model.save(sac_path)

    evaluate_policy(model, train_env, n_eval_episodes=30, render=True)
    """

    train_env = DummyVecEnv([lambda: train_env])
    sac_path = os.path.join("models", "CustomHopper_source_v0_SAC_200k")
    model = SAC.load(sac_path, train_env)
    evaluate_policy(model, train_env, n_eval_episodes=30, render=True)


if __name__ == '__main__':
    main()