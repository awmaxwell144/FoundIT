import numpy as np
import jax.numpy as jnp

def compute_reward(state) -> float:
    # Calculate the Manhattan distance to the goal
    dist_to_goal = jnp.sum(jnp.abs(state.pos - state.goal))
    
    # Calculate a reward inversely proportional to the distance
    # Increase the scaling factor to 0.5 to provide a stronger incentive for shorter paths 
    reward_dist = -0.5 * dist_to_goal

    # Check if the goal is reached
    is_goal_reached = jnp.logical_and(state.pos[0] == state.goal[0], state.pos[1] == state.goal[1])
  
    # If the goal is reached, we give a large positive reward
    # Increase the reward for reaching the goal to 500 to strongly incentivize goal attainment
    reward_goal = jnp.where(is_goal_reached, 500., 0.)
  
    # Time penalty for each step to encourage faster completion
    # Increase the penalty to -5 to add more pressure to reach the goal quickly
    reward_time = -5.0
  
    # Total reward is a combination of the three components
    reward = reward_dist + reward_goal + reward_time

    return reward