import os
import subprocess
import logging
import httpx

ROOT_DIR = os.getcwd()


def setup(env_name):
    copy_task_fitness(env_name)
    copy_animate(env_name)
    clear_logs()
    # Suppress httpx logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
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

def copy_animate(env_name):
    env_animate = f'{ROOT_DIR}/envs/{env_name}/animate.py'
    output_animate = f'{ROOT_DIR}/run_visualize/animate.py'
    # Open the input file in read mode
    with open(env_animate, 'r') as input_file:
        animate = input_file.read()  # Read the content
    
    # Open the output file in write mode and write the content
    with open(output_animate, 'w') as output_file:
        output_file.write(animate)


def clear_logs():
    reward_log = 'output/rewards_log.txt'
    all_log = 'output/all_logs.txt'
    with open(reward_log, 'w') as file: pass
    with open(all_log, 'w') as file: pass



def is_ollama_running(host="http://127.0.0.1", port=11434):
    """Check if the Ollama server is running."""
    try:
        response = httpx.get(f"{host}:{port}/health")
        return response.status_code == 200
    except Exception as e:
        logging.debug(f'Ollama server not running: {e}')
        start_ollama_server()

def start_ollama_server(): 
    try:
        logging.info("Starting ollama server")
        # Start Ollama server
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.info("Ollama server started.")
    except Exception as e:
        logging.warning(f"Failed to start Ollama server: {e}")

def check_and_pull_model(model_name="llama3.1"):
    """
    Check if a specific model (e.g., llama3) is downloaded, and pull it if not.
    """
    try:
        # Get the list of available models
        models = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        if model_name in models.stdout:
            logging.info(f"Model '{model_name}' is already downloaded.")
        else:
            logging.info(f"Model '{model_name}' is not available locally. Pulling it now...")
            subprocess.run(["ollama", "pull", model_name], check=True)
            logging.info(f"Model '{model_name}' has been successfully downloaded.")
    except subprocess.CalledProcessError as e:
        logging.warning(f"Failed to check or pull the model '{model_name}': {e}")
