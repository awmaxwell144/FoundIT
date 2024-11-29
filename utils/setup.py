import yaml
import os

ROOT_DIR = os.getcwd()

from utils.logs import clear_logs
from utils.ollama_checks import is_ollama_running, check_and_pull_model

def setup(env_name):
    copy_task_fitness(env_name)
    clear_logs()
    
    # run checks on ollama
    is_ollama_running()
    check_and_pull_model()


def copy_task_fitness(env_name):
    env_tff = f'{ROOT_DIR}/envs/{env_name}/task_fitness.py'
    output_tff = f'{ROOT_DIR}/evaluate/tff.py'
    # Open the input file in read mode
    with open(env_tff, 'r') as input_file:
        tff = input_file.read()  # Read the content
    
    # Open the output file in write mode and write the content
    with open(output_tff, 'w') as output_file:
        output_file.write(tff)


