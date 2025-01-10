import numpy as np
import jax.numpy as jnp

def compute_reward(state) -> float:
    x, x_dot, theta, theta_dot = state.x, state.x_dot, state.theta, state.theta_dot

    # Encourage the pole to stay upright and the cart to stay in the center
    theta_cost = (jnp.abs(theta) / jnp.pi) ** 2
    x_cost = (jnp.abs(x) / 2.4) ** 2

    # Penalize extreme velocities which may make the system unstable
    x_dot_cost = jnp.clip(jnp.abs(x_dot) / 10., 0., 1.) 
    theta_dot_cost = jnp.clip(jnp.abs(theta_dot) / 10., 0., 1.)

    # Combine individual cost terms with respective weights (pole being upright has the highest priority)
    reward = 1.0 - (0.5 * theta_cost + 0.3 * x_cost + 0.1 * x_dot_cost + 0.1 * theta_dot_cost)

    return reward