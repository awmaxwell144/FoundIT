

def reward_log(text, intro=""):
    output_file = 'output/rewards_log.txt'
    with open(output_file, 'a') as file:
            file.write(intro + '\n')
            file.writelines(text)
            file.write('\n\n')

def all_log(text, intro="", type = ""):
    output_file = 'output/all_logs.txt'
    if (type == "reward_info"):
        log_reward_info(text)
    elif(type == "messages"):
        log_messages(text)
    else:
        with open(output_file, 'a') as file:
                file.write(intro + '\n')
                file.writelines(text)
                file.write('\n\n')
    

def log_messages(messages):
    output_file = 'output/all_logs.txt'
    with open(output_file, 'a') as file:
        file.write("Messages: \n")
        file.write("System: ")
        file.write(messages[0]["content"])
        file.write("\n")
        file.write("User: \n")
        file.write(messages[1]["content"])
        file.write("\n\n")


def log_reward_info(reward_info):
    output_file = 'output/all_logs.txt'
    with open(output_file, 'a') as file:
        file.write("Reward Function: \n")
        file.write(reward_info["reward_function"])
        file.write("\n\n\nCumulative Reward: ")
        file.write(str(reward_info["reward_seq"]))
        file.write("\nTask Fitness Function Output: ")
        file.write(str(reward_info["eval"]))
        file.write("\n" + str(reward_info["state_seq"]))
        file.write("\nException: ")
        file.write(str(reward_info["exception"]))
        file.write("\n\n\n\n")
