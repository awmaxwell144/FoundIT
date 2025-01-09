import re
def tff(reward_info):
    final_state = reward_info["final_state"]
    match = re.search(r"time:\s*([-+]?\d*\.\d+|\d+)", final_state)
    time = float(match.group(1))
    return time