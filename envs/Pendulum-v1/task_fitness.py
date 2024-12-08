def tff(reward_info):
    final_state = reward_info["final_state"]
    final_theta = float(final_state["theta"])
    return abs(final_theta)