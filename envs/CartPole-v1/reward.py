import jax.numpy as jnp

def compute_reward(self, state):
    theta = state.theta
    theta_threshold = self.params.theta_threshold_radians
    
    # Penalize if the pole is pointing down or has a large angle
    reward_down = -jnp.abs(theta)
    
    # Penalize if the pole is too far away from the cart
    x = state.x
    x_threshold = 2 * self.params.x_threshold
    reward_away = -jnp.abs(x) / x_threshold
    
    # Reward for keeping the pole upright and close to the cart
    reward_upright = jnp.cos(theta)
    
    # Combine rewards with different weights
    weight_down = 0.5
    weight_away = 0.2
    weight_upright = 0.3
    
    return weight_down * reward_down + weight_away * reward_away + weight_upright * reward_upright

class CartPoleEnvironment:
    def __init__(self, params):
        self.params = params
        
    # ... other methods ...