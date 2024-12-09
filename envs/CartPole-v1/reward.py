import numpy as np
import jax.numpy as jnp
def compute_reward(state):
    # Reward for keeping pole upright and close to vertical position (theta)
    theta_penalty = -abs(state.theta)  # Negative reward if pole is tilted
    # Reward for keeping pole at rest (theta_dot)
    velocity_penalty = -state.theta_dot**2  # Squared term to penalize faster movements

    # Additive penalty for going over the limits on x and theta (as before)
    x_limit_penalty = -abs(state.x)  # Negative reward if cart goes out of bounds
    theta_limit_penalty = -abs(state.theta)  # Negative reward if pole is tilted beyond threshold

    # Combine all penalties with a positive weight to obtain final reward
    reward = (
        theta_penalty + velocity_penalty 
        + x_limit_penalty * 0.1 
        + theta_limit_penalty * 0.05
    )
    
    return reward