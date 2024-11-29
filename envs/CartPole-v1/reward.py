import jax.numpy as jnp

def compute_reward(self, state):
    # Reward is 1 if the pole is still upright and -1 otherwise
    theta = state.theta
    theta_threshold_radians = self.params.theta_threshold_radians
    
    # We use a sigmoid function to interpolate between rewards of -1 and 0
    # The temperature parameter controls how quickly we reach 0 reward from -1
    temp = 10.0  # Temperature parameter
    reward = jnp.where(
        jnp.abs(theta) <= theta_threshold_radians, 
        1.0 / (1 + jnp.exp(-(jnp.abs(theta)) / temp)), 
        -1.0)
    
    return reward