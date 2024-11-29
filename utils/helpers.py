import os
import yaml

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