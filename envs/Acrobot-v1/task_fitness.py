import math
def tff(reward_info):
    final_state = reward_info["final_state"]
    theta1 = float(final_state["joint_angle1"])
    theta2 = float(final_state["joint_angle2"])
    return -math.cos(theta1) - math.cos(theta2 + theta1)