import gymnax
import argparse
import os
from utils.helpers import  load_config
from utils.run import rollout_episode, load_neural_network
from animate import animate

ROOT_DIR = os.getcwd()

def run(env_name):
    base = f"envs/{env_name}/ppo"
    configs = load_config(base + ".yaml") # load the config file for the specified environment
    # if not random, load the trained model from the .pkl file
    model, model_params = load_neural_network(
        configs.train_config, base + ".pkl"
    )

    # create environment and parameters using the config
    env, env_params = gymnax.make(
        configs.train_config.env_name,
        **configs.train_config.env_kwargs,
    )

    # update the environment parameters with the ones specified in the config
    env_params.replace(**configs.train_config.env_params)
    
    state_seq, cum_rewards, reward_seq = rollout_episode(# call rollout_episode function to simulate an episode
        env, env_params, model, model_params
    )

    print(f'rewards {cum_rewards} duration {len(reward_seq)} state_seq {state_seq}')

def run_animate(env_name):

    base = f"envs/{env_name}/ppo"
    configs = load_config(base + ".yaml") # load the config file for the specified environment
    # if not random, load the trained model from the .pkl file
    model, model_params = load_neural_network(
        configs.train_config, base + ".pkl"
    )

    # create environment and parameters using the config
    env, env_params = gymnax.make(
        configs.train_config.env_name,
        **configs.train_config.env_kwargs,
    )

    # update the environment parameters with the ones specified in the config
    env_params.replace(**configs.train_config.env_params)
    
    state_seq, cum_rewards, reward_seq = rollout_episode(# call rollout_episode function to simulate an episode
        env, env_params, model, model_params
    )

    animate(state_seq, f'output/{env_name}_test.mp4')

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    # define command line arguments
    parser.add_argument( # specify the environment name
        "-env",
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Environment name.",
    )
    parser.add_argument( # specify the output directory
        "-a",
        "--animate",
        type=bool,
        default=False,
        help="True: run and animate, False: just run.",
    )
    args, _ = parser.parse_known_args()
    if args.animate:
        run_animate(args.env_name)
    else:
        run(args.env_name)
