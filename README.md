# FoundIT
## Contrastive Language-Image Pre-training to Automate Reward Function Evaluation in Large Language Model-Generated Reward Functions
Repository **without** Contrastive Language-Image Pre-training (CLIP) for reward evaluation

### Instructions:
Clone the repository:
``` git clone https://github.com/awmaxwell144/FoundIT_VLM.git ```

#### Container:
It is reccomended to run this repository within the provided container.
There are two options,
- build, push, and launch the container from the code in the `/container` directory, 
- pull and launch the pre-built container 
```
dockerhub:
    username: am8792
    image: gen_reward
    tag: v1 
```

#### OpenAI Key:
The process can be run with GPT-4 or Llama3.
If you wish to run it with GPT-4, you must specify your OpenAI API Key as follows:
```export OPENAI_API_KEY=YOUR_OPENAI_API_KEY```

#### Run:
To run the process, the basic command is
```python3 foundIT_VLM.py```
This will, by default, run the process on the CartPole-v1 environment with Llama3

The results of the run will be stored in the `/outputs` folder.

Run Flags:
`found_IT_VLM.py` has two flags: `-env` and `-c`

`-env`:
The environment flag specifies which environment you would like to run the process on. 
There are seven built in environments: 
- Acrobot-v1
- CartPole-v1
- Catch-bsuite
- FourRooms-misc
- MountainCar-v0
- MountainCarCont-v0
- Pendulum-v1
Information about these environments can be found in the [gymnax](https://github.com/RobertTLange/gymnax/tree/main) and [gymnax baselines](https://github.com/RobertTLange/gymnax-blines/tree/main) repositories

`-c`:
The Chat-GPT flag specifies which LLM to use. If the `-c` flag is present, the process will use GPT-4. If it is not present, it will use Llama3


Editing parameters:

The number of iterations and the number of samples per iteration can be edited on a by-environment basis in the environment's config file.
For example, the config file for the CartPole-v1 environment can be found at the fullowing path: 
```
envs/CartPole-v1/CartPole-v1.yaml
```

Sections of code or inspiration for this project come from the following sources:
https://github.com/RobertTLange/gymnax/tree/main
https://github.com/RobertTLange/gymnax-blines/tree/main
https://github.com/eureka-research/Eureka