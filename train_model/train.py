import os
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import jax
from utils.models import get_model_ready
from utils.helpers import load_config, save_pkl_object
from utils.ppo import train_ppo as train_fn
from datetime import datetime
from mle_toolbox import MLExperiment


def main(config, mle_log):
    """Run training with ES or PPO. Store logs and agent ckpt."""
    rng = jax.random.PRNGKey(config.seed_id)
    # Setup the model architecture
    rng, rng_init = jax.random.split(rng)
    model, params = get_model_ready(rng_init, config)



    # Log and store the results.
    log_steps, log_return, network_ckpt = train_fn(
         rng, config, model, params, mle_log
    )

    data_to_store = {
        "log_steps": log_steps,
        "log_return": log_return,
        "network": network_ckpt,
        "train_config": config,
    }

    save_pkl_object(
        data_to_store,
        f"envs/{config.env_name}/ppo.pkl",
    )


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-env",
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Environment name",
    )
    args, _ = parser.parse_known_args()


    log_config = {"use_wandb": True,
                "wandb_config": {
                "key": "85e5e85bc8e81793e548faaa04f6347bb40ba04c",  # Only needed if not logged in
                "entity": "awmaxwell144-princeton-university",  # Only needed if not logged in
                "project": "Eureka_Jax",
               "group": "Eureka_Jax",
               "name": datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
                }
        
    mle = MLExperiment(config_fname=f'envs/{args.env_name}/ppo.yaml', log_config=log_config)
    main(mle.train_config, mle_log=mle.log)
    print("Completed")



    