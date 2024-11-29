import os
import sys
import ollama
import logging
import argparse
import subprocess
from utils.helpers import read_config, file_to_string, write_to_py, process_run, process_error
from utils.setup import setup
from utils.logs import all_log, reward_log
from generate_reward.tools.rewards import format_reward


ROOT_DIR = os.getcwd()

# Main
def main(env_name):
    logging.basicConfig(level = logging.INFO)
    
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
        logging.info(f'\n\n Iteration: {iter+1}')
        reward_log(f"Iteration: {iter+1}")
        all_log(f"Iteration: {iter+1}")

        # while true (for each sample in the iteration)
        # while true (for each sample in the iteration)
        while (True):
            # if you have enough samples, break
            if (num_samples >= samples): break

            try:
                logging.info(f'Generating reward number {num_samples+1}')
                # generate response given messages
                cur_response = ollama.chat(model = "llama3.1",
                               messages = messages)['message']['content']
                
                num_samples+=1
            except Exception as e:
                # log the error
                logging.warning(f'Reward generation attempt failed with error: {e}')
                # exit()
                exit()
            if (cur_response == None): 
                logging.warning('Code terminated due to failed attempt')
                exit()

            # format generated response
            reward = format_reward(cur_response)
            reward_log(reward, f'Reward Number {num_samples+1}:')
            # add generated response to whatever is storing responses
            responses.append(reward)



        reward_info_all = []
        exceptions = []
        reward_location = f'envs/{env_name}/reward.py'
        reward_num = 1 # for log purposes
        from evaluate.tff import tff
        # for each reward function
        for r in responses:
            logging.info(f'Evaluate {reward_num}')
            encountered_exception = False
            exception = "None"
            reward_seq = []
            state_seq = ""
            duration = 0
            # add generated reward function to environment code 
            write_to_py(reward_location, r)

            # train model (it shouldnt break if there's an error here)
            try:
                logging.info(f'Training model')
                train_output = subprocess.run(['python3', 'train_model/train.py', '-env',f'{env_name}'], 
                                          capture_output=True, check=True, text=True)
                logging.info("Training complete")
                 # run model (it shouldnt break if there's an error here)
                try:
                    logging.info('Running model')
                    run_output = subprocess.run(['python3', 'run_visualize/run_visualize.py', '-env',f'{env_name}'], 
                                            capture_output=True, check=True, text=True)
                    logging.info("Run Complete")
                    reward_seq, duration, state_seq = process_run(run_output.stdout)
                except subprocess.CalledProcessError as e:
                    logging.info("Exception occurred when running the model, moving to next reward")
                    encountered_exception = True
                    exception = process_error(e.stderr)
                
            except subprocess.CalledProcessError as e:
                logging.info("Exception occurred when training model, moving to next reward")
                encountered_exception = True
                exception = process_error(e.stderr)
           
            # evaluate with provided task fitness function
            if not encountered_exception:
                eval = tff(reward_seq, duration)
            else:
                eval = 0

            
            # store run and model information
            reward_info = {
                "reward_function": r,
                "reward_seq": reward_seq,
                "duration": duration,
                "eval": eval,
                "state_seq": state_seq,
                "exception": exception
            }
            reward_log(reward_info["reward_function"])
            all_log(reward_info, f'Reward {reward_num} Information: ', type = "reward_info")
            reward_info_all.append(reward_info)
            exceptions.append(encountered_exception)
            reward_num+=1


        # if it's not the last iteration 
        if (iter < (iterations - 1)):
            # Reward reflection
            # if all generate execution errors
            if all(exceptions):
                # choose the first reward to move forward with
                reward_info = reward_info_all[0]
                executing_error = executing_error_feedback.format(reward_func = reward_info["reward_function"], initial_user = initial_user, traceback_msg = reward_info["exception"])
                messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": executing_error}]
                continue #should skip to next iteration of outer loop

            # if some execute
            else:
                best_eval = 0
                best_reward = {}
                # choose the *best* reward function based on task fitness function output
                for reward_info in reward_info_all:
                    if (reward_info["eval"] > best_eval):
                        best_eval = reward_info["eval"]
                        # record stats about *best* reward
                        best_reward = reward_info
                
                # update the messages
                feedback = policy_feedback.format(reward_function = best_reward["reward_function"],
                                            reward = best_reward["reward_seq"],
                                            duration = best_reward["duration"])
                user = feedback + code_feedback + initial_user
                messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": user}]


    # choose the best reward function, write it to final_reward, run, and animate
    # if all generate execution errors
    if all(exceptions):
        # choose the first reward to move forward with
        best_reward = reward_info_all[0]
        write_to_py(f'output/final_reward.py', best_reward["reward_function"])
        logging.warning("The final reward function does not execute.")
        all_log(best_reward, "Final Reward Function (does not execute)", type = "reward_info")
    else:
        best_eval = 0
        best_reward = {}
        # choose the *best* reward function based on task fitness function output
        for reward_info in reward_info_all:
            if (reward_info["eval"] > best_eval):
                best_eval = reward_info["eval"]
                best_reward = reward_info

        # record stats about *best* reward
        all_log("Final Reward Information")
        all_log(best_reward, type = "reward_info")
        reward_log(best_reward["reward_function"], "Final Reward Function")
        write_to_py(f'output/final_reward.py', best_reward["reward_function"])
        try:
            logging.info('Run and animate final')
            run_output = subprocess.run(['python3', 'run_visualize/run_visualize.py', '-env',f'{env_name}', '-a', 'True'], 
                                    capture_output=True, check=True, text=True)
            logging.info("Run and animate complete")
        except subprocess.CalledProcessError as e:
            logging.info("Exception occurred when running and animating the model")





# call main
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument( # specify the environment name
        "-env",
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Environment name.",
    )
    args, _ = parser.parse_known_args()
    main(args.env_name)