import numpy as np
import jax.numpy as jnp

def compute_reward(state) -> float:
    # Defining reward for the agent
    reward = 1.0 - 0.1*(state.theta**2) - 0.01*(abs(state.x_dot)) - 0.01*(abs(state.x)) - 0.1*(state.theta_dot**2)
    
    return reward