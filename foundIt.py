import os
import sys
import logging
import argparse
from utils.setup import setup
from utils.helpers import file_to_string, read_config
from utils.logs import all_log, reward_log


ROOT_DIR = os.getcwd()

# Main
def main(env_name):
    logging.basicConfig(level = logging.DEBUG)
    
    # load config info
    logging.debug("Load config")
    env_name, task_description, task, iterations, samples = read_config(env_name)
    reward_location = f'envs/{env_name}/reward.py'

    # setup
    setup(env_name)

    # load all text prompts
    logging.debug('Loading prompts')
    prompt_dir = f'{ROOT_DIR}/generate_reward/prompts'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    reward_signature = file_to_string(f'{ROOT_DIR}/envs/{env_name}/reward_signature.txt')
    task_file = f'{ROOT_DIR}/envs/{env_name}/{env_name}.py'
    task_obs_code_string  = file_to_string(task_file)
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    executing_error_feedback = file_to_string(f'{prompt_dir}/executing_error_feedback.txt')

    # concatenate prompts into initial messages
    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    # define variables for loop
    responses = []

    # for number of iterations
    for iter in range(iterations):
        # define loop internal vars
        responses.clear()
        cur_response = ""
        num_samples = 0

        # logging
        logging.info(f'\n\n Iteration: {iter}')
        reward_log(f"Iteration: {iter}")
        all_log(f"Iteration: {iter}")

        # while true (for each sample in the iteration)
            # if you have enough samples, break
            # try:
                # generate response given messages
            # except Exception as e:
                # log the error
                # exit()
            # add generated response to whatever is storing responses

        # for each reward function
            # process generated response
            # append proccessed reward function to some way of storing
            # add generated reward function to environment code 
            # train model (it shouldnt break if there's an error here)
            # run model (it shouldnt break if there's an error here)
            # store model information

        # Reward reflection
            # if all generate execution errors
                # break forward to the next iteration with the same messages
            # if its not all of them
                # evaluate the ones that don't generate execution errors
                # choose the *best* one
                # record stats about *best* reward
                # record general stats abt every reward
                # update the messages

    # write something to note if none of the iterations generate something executable

    # evaluate performance over iterations





# call main
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument( # specify the environment name
        "-env",
        "--env_name",
        type=str,
        default="cartpole",
        help="Environment name.",
    )
    args, _ = parser.parse_known_args()
    main(args.env_name)