import re 
def tff(reward_info):
    final_state = reward_info["final_state"]
    match = re.search(r"position:\s*([-+]?\d*\.\d+|\d+)", final_state)
    final_position = float(match.group(1))
    return final_position