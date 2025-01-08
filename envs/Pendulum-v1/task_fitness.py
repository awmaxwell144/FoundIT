import re
def tff(reward_info):
    final_state = reward_info["final_state"]
    match = re.search(r"theta:\s*([-+]?\d*\.\d+|\d+)", final_state)
    final_theta = float(match.group(1))
    return abs(final_theta)