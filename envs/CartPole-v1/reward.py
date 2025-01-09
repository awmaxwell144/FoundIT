import numpy as np
import jax.numpy as jnp

def compute_reward(state):
    x, x_dot, theta, theta_dot = state.x, state.x_dot, state.theta, state.theta_dot

    # define the threshold levels
    theta_threshold_radians = 12 * 2 * jnp.pi / 360
    x_threshold = 2.4

    failure_conditions = jnp.logical_or(jnp.logical_or(x < -x_threshold, x > x_threshold), 
                                        jnp.logical_or(theta < -theta_threshold_radians, theta > theta_threshold_radians))

    # Reward for staying alive
    alive_reward = 1.0

    # Quadratic penalties
    theta_penalty = -2.0 * jnp.square(theta)  # Increased from 1.0 to 2.0 to increase penalty
    x_penalty = -0.5 * jnp.square(x)  # Increased from 0.1 to 0.5 to increase penalty
    
    # Penalty for velocity - we want the cart to stay still, not just upright
    theta_dot_penalty = -0.2 * jnp.square(theta_dot)
    x_dot_penalty = -0.1 * jnp.square(x_dot)

    reward = jnp.where(failure_conditions, 0.0, alive_reward + theta_penalty + x_penalty + theta_dot_penalty + x_dot_penalty)

    return reward