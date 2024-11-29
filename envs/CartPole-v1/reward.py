import jax.numpy as jnp  # Import jnp

def compute_reward(state):
    angle_reward = -jnp.abs(state.theta)
    theta_threshold_radians = 0.209439  # Threshold value for theta (in radians)

    if state.theta < -theta_threshold_radians or state.theta > theta_threshold_radians:
        # Penalize the agent when the pole goes beyond the threshold
        return -10.0
    else:
        # Reward the agent for keeping the pole upright
        return 1.0

# Improvements to your reward function:
#
#   1. I've added a threshold value for theta (pole's angle) to make the reward more meaningful.
#      This is because, in real-world scenarios, the pole might not remain perfectly straight.
#
#   2. The reward function now returns -10.0 when the pole goes beyond the threshold.
#       This encourages the agent to keep the pole upright and avoid penalties.
#
#   3. For simplicity, I've kept the reward value at 1.0 when the pole is within the allowed range.
#       You can experiment with different values to make your reward function more suitable for your use case.
import jax.numpy as jnp

def compute_reward(state):
    angle_reward = -jnp.abs(state.theta)
    return 0.1 * angle_reward + 0.9 * (-state.x)  # Combine multiple reward components into a single value