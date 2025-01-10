import numpy as np
import jax.numpy as jnp

def compute_reward(state) -> float:
    theta, theta_dot = state.theta, state.theta_dot
    
    # Normalize theta to [-pi, pi]
    theta = ((theta + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    
    # It is beneficial to minimize theta (keeping pendulum upright), 
    # theta_dot (reducing swing) and time (doing it faster)
    # We can slightly increase the weights for the theta and theta_dot to push the agent to learn faster
    costs = theta**2 + 0.2*theta_dot**2 + 0.001*state.time**2

    return -costs