import os
import yaml
import re
import ast

ROOT_DIR = os.getcwd()

def read_config(task_name):
    config_path = f'{ROOT_DIR}/envs/{task_name}/{task_name}.yaml'

    # Check if the file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Check if cfg is None
    if cfg is None:
        raise ValueError(f"The configuration file '{config_path}' is empty or not properly formatted.")  
    
    env_name = cfg["env_name"]
    task_description = cfg["description"]
    task = cfg["task"]
    iterations = cfg["iterations"]
    samples = cfg["samples"]
    return env_name, task_description, task, iterations, samples

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()
    
def write_to_py(output_file, code):
    with open(output_file, 'w') as py_file:
            py_file.writelines(code)

def process_run(input_string):
     # Regular expression to find the list between 'rewards' and 'duration'
    rewards_pattern = r"rewards\s*\[\s*(.*?)\s*\]\s*duration"
    duration_pattern = r"duration\s*(\d+)\s*"
    

    # Extract rewards list
    rewards_match = re.search(rewards_pattern, input_string, flags=re.DOTALL)
    if rewards_match:
        rewards_str = rewards_match.group(1)
        # Convert to a list of floats
        reward_seq = [float(x.strip()) for x in rewards_str.split()]
    else:
        reward_seq = []

    # Extract duration integer
    duration_match = re.search(duration_pattern, input_string)
    if duration_match:
        duration = int(duration_match.group(1))
    else:
        duration = None

    state_seq = process_state_seq(input_string, duration)

    return reward_seq, duration, state_seq

def process_error(input_string):
    if (input_string == None): return input_string

    keyword = "File"
    # Find all occurrences of "File"
    occurrences = [index for index in range(len(input_string)) if input_string.startswith(keyword, index)]

    # Check if there are at least three occurrences
    if len(occurrences) >= 5:
        # Get the start index of the third-to-last occurrence
        start_index = occurrences[-5]
        # Return the substring from that point to the end
        return input_string[start_index:]
    else:
        # If there are fewer than three occurrences, return an empty string
        return input_string


def process_state_seq(input_string, duration):
    state_seq_pattern = r"state_seq\s*(.*)"
    state_seq_match = re.search(state_seq_pattern, input_string, flags=re.DOTALL)
    if state_seq_match:
        state_seq_str = state_seq_match.group(1).strip()  # Extract and strip leading/trailing whitespace
        # using eval because I directly producing and passing in the input
        state_seq_all = eval(state_seq_str)
        state_seq = reformat_state_seq(state_seq_all, duration)

    else:
        state_seq = ""
    return state_seq
    
def reformat_state_seq(state_seq, duration):
    counter = 0
    if (duration <= 15): spread = 2
    elif (duration < 50): spread = 5
    else: spread = 10

    output = f'\nEvery {spread} state sequence(s): '
    for state in state_seq:
        if (counter % spread == 0):
            output= output + "\n\nTime: " + str(state.time)
            output= output + "\nx: " + str(state.x)
            output= output + "\nx_dot: " + str(state.x_dot)
            output= output + "\ntheta: " + str(state.theta)
            output= output + "\ntheta_dot: " + str(state.theta_dot)
        counter+=1
    output+= "\n\n"
    return output
# Define placeholder functions/classes
class EnvState:
    def __init__(self, time, x, x_dot, theta, theta_dot):
        self.time = time
        self.x = x
        self.x_dot = x_dot
        self.theta = theta
        self.theta_dot = theta_dot

    def __repr__(self):
        return f"EnvState(time={self.time}, x={self.x}, x_dot={self.x_dot}, theta={self.theta}, theta_dot={self.theta_dot})"

def Array(value, dtype=None, weak_type=None):
    return value  # Return the value directly or create a wrapper if needed

# Define placeholders for `int32` and other similar types
int32 = "int32"  # Or any placeholder, if needed
float32 = "float32"

if __name__ == "__main__":
    with open('test.txt','r') as file:
        input = file.read()
    process_run(input)