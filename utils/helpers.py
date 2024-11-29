import os
import yaml
import re

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
    duration_pattern = r"duration\s*(\d+)"

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

    return reward_seq, duration

def process_error(input_string):
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

