import ray
from ray import tune
from abides_markets.configs import rmsc04

from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLoggerCallback
import wandb
from abides_gym.envs.markets_execution_custom_metrics import MyCallbacks


api_key = wandb.api.api_key


# Import to register environments
import abides_gym

ray.shutdown()
ray.init(num_cpus=10)

"""
PPO's default:
sample_batch_size=200, batch_mode=truncate_episodes, training_batch_size=4000 -> workers collect batches of 200 ts (per worker), then the policy network gets updated using the 4000 last ts (in n minibatch-chunks of 128 (sgd_minibatch_size)).

DQN's default:
train_batch_size=32, sample_batch_size=4, timesteps_per_iteration=1000 -> workers collect chunks of 4 ts and add these to the replay buffer (of size buffer_size ts), then at each train call, at least 1000 ts are pulled altogether from the buffer (in batches of 32) to update the network.
"""

name_xp = "dqn_execution_v1_11"
tune.run(
    "DQN",
    name=name_xp,
    resume=False,
    stop={"training_iteration": 100},  # "episode_reward_mean": 2e6,
    checkpoint_at_end=True,
    checkpoint_freq=20,
    config={
        "callbacks": MyCallbacks,
        # Environment stuff
        "env": "markets-execution-v0",  # "CartPole-v1"markets-execution-v0
        "env_config": {
            "mkt_close": "16:00:00",
            "timestep_duration": tune.grid_search(["30S"]),
            #'order_fixed_size': 10,
            "execution_window": tune.grid_search(
                ["01:00:00", "00:30:00"]
            ),  # , "00:10:00"
            # 'timestep_duration':'30S',
            #     'debug_mode':False,
            #     'mkt_close':"10:30:00",
            "parent_order_size": tune.grid_search([20000]),
            "order_fixed_size": tune.grid_search([500, 1000]),
            #     'execution_window':tune.grid_search(["01:00:00", "00:30:00"]),
            "not_enough_reward_update": tune.grid_search([0, -1, -10, -100]),
            #
        },
        "seed": tune.grid_search(
            [1, 2, 3]
        ),  # 0,# tune.grid_search([1, 2]),#tune.grid_search([1, 2, 3, 4, 5]),
        # parallelism stuff
        "num_gpus": 0,
        "num_workers": 0,
        # Learning algo stuff
        "hiddens": [50, 20],  # tune.grid_search([[50, 20]]),
        "gamma": 1,
        "lr": 0.0001,  # tune.grid_search([0.01, .001, 0.0001])
        "framework": "torch",
        "observation_filter": "MeanStdFilter",
    },
    callbacks=[
        WandbLoggerCallback(
            project="abides_markets_execution_abm",
            group=name_xp,  # if not specified will be trainable name (here DQN)
            api_key=api_key,
            log_config=False,
        )
    ],
)
