import numpy as np
import jax.numpy as jnp

def compute_reward(state) -> float:
    
    # Reward reaching high velocities
    reward = 10 * (jnp.abs(state.velocity) / jax.numpy.float32(1))
    
    # Add a bonus for moving forward
    reward += 5 * ((state.position >= 0) - (state.position < 0))
    
    # Penalize being far away from the target position
    reward -= 2 * abs(state.position)
    
    return jnp.clip(reward, -10.0, 10.0)