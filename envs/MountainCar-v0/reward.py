import numpy as np
import jax.numpy as jnp

def compute_reward(state) -> float:
    # Calculate the bonus based on the distance to the goal
    goal_bonus = jnp.where(state.position <= 0.5, jnp.exp(-10 * abs(state.position)) + 1, 1)
    
    # Reward velocity as well
    velocity_bonus = jnp.clip(state.velocity / 0.07, 0, 1)  # clip to [0, 1] range
    
    # Combine the bonuses into a single reward value
    reward = goal_bonus * velocity_bonus
    
    return reward