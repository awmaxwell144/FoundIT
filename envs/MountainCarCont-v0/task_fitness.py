def tff(reward_info):
    final_state = reward_info["final_state"]
    final_position = final_state["position"]
    return float(final_position)