import numpy as np
import jax.numpy as jnp

def compute_reward(state) -> float:
    # define position goal
    goal_position = 0.0
    distance_to_goal = jnp.abs(state.position - goal_position) # absolute distance to goal
    speed = jnp.abs(state.velocity) # absolute speed
  
    # higher speed and lesser distance to goal gets higher reward
    reward = speed - distance_to_goal

    return reward