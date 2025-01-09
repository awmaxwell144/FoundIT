import numpy as np
import jax.numpy as jnp

def compute_reward(state) -> float:
    # define the target position
    target_position = 0.0
    # compute the absolute difference between current position and target
    distance = jnp.abs(state.position - target_position)
    # define the reward as negative distance plus velocity
    reward = -distance + state.velocity - 0.01*state.time
    return reward