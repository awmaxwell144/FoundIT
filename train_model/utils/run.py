import numpy as np
import jax
import os
from utils.models import get_model_ready
from utils.helpers import load_pkl_object
import matplotlib.pyplot as plt


# initializes the model using a random key 'rng'
# model is set up using the 'get_model_ready' function
# model parameters are loaded from the pkl path specified by 'agent_path'
def load_neural_network(config, agent_path): 
    rng = jax.random.PRNGKey(0)
    model, _ = get_model_ready(rng, config)

    params = load_pkl_object(agent_path)["network"]
    return model, params

# episode rollout refers to the process of simulating one complete episode of an agent interacting with an environment
# environment is reset, and the initial st
def rollout_episode(env, env_params, model, model_params, max_frames=200):
    state_seq = [] # list to store the sequence of states encountered during the episode
    rng = jax.random.PRNGKey(0)

    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params) # initial observation from the environment, initial state of the environment

    if model is not None:
        if model.model_name == "LSTM":
            hidden = model.initialize_carry()

    t_counter = 0 # count the number of steps taken
    reward_seq = [] # list to store the rewards recieved at each step
    while True:
        state_seq.append(env_state) # append the current state
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        if model is not None:
            if model.model_name == "LSTM": # if the model is an LSTM, apply the model to get the next hidden state and action
                hidden, action = model.apply(model_params, obs, hidden, rng_act)
            else:
                if model.model_name.startswith("separate"):
                    # if the model name starts with 'separate', get the value and policy, then sample an action from the policy
                    v, pi = model.apply(model_params, obs, rng_act)
                    action = pi.sample(seed=rng_act)
                else:
                    # otherwise, apply the model to get the action
                    action = model.apply(model_params, obs, rng_act)
        else:
            # if there's no models, take the default action
            action = 0  # env.action_space(env_params).sample(rng_act)

        # step the environment by executing the chosen action
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )

        reward_seq.append(reward) # append the reward to the reward sequence
        print(t_counter, reward, action, done) 
        print(10 * "=")
        t_counter += 1
        if done or t_counter == max_frames:
            break
        else:
            env_state = next_env_state
            obs = next_obs
    print(f"{env.name} - Steps: {t_counter}, Return: {np.sum(reward_seq)}")
    return state_seq, np.cumsum(reward_seq)


def graph_save(plot_metric, data, env_name, path):
    fig,ax = plt.subplots()
    ax.plot(data)
    plt.title(f'{plot_metric} over Episode Rollout')
    plt.xlabel("Episode Number")
    plt.ylabel(f'{plot_metric}')
    plt.savefig(f'{path}/{env_name}_{plot_metric}.png')