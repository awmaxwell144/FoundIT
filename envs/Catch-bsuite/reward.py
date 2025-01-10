import numpy as np
import jax.numpy as jnp

def compute_reward(state):
    # Calculate distance between P and B in x and y directions
    dx = state.paddle_x - state.ball_x
    dy = (state.paddle_y - state.ball_y)

    # Reward for positioning P right below B, normalized to [-1, 0, 1]
    reward = -jnp.abs(dx) + jnp.sign(dy) * jnp.abs(dy)
    
    # Clip reward to range [-1, 1] to prevent large rewards
    return jnp.clip(reward, -1.0, 1.0)