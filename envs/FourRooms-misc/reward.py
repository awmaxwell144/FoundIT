import numpy as np
import jax.numpy as jnp

def compute_reward(state) -> float:
    # Euclidean distance between current position and goal
    distance = jnp.linalg.norm(state.pos - state.goal)
    
    # Reward for reaching the goal is 100.0 else a penalty proportional to the distance and time elapsed
    reward = jnp.where(jnp.all(state.pos == state.goal, axis=-1), 100.0, -0.02*distance - 0.01*state.time)

    return reward