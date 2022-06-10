import os
import numpy as np
import gym
from gym.envs.registration import register
import ray
from ray import tune
from ray.tune.registry import register_env
from abides_gym.envs.marl_environment_v0 import SubGymMultiAgentRLEnv_v0
import argparse
from abides_core.utils import str_to_ns
from scripts.marl_utils import multi_agent_init, multi_agent_policies

parser = argparse.ArgumentParser()
parser.add_argument('-vm',
                    '--mm_add_volume',
                    default=0,
                    type=int,
                    help='Whether to add volume info to MM state; 0: Dont add; 1: Add')
parser.add_argument('-vp',
                    '--pt_add_volume',
                    default=0,
                    type=int,
                    help='Whether to add volume info to PT state; 0: Dont add; 1: Add')
parser.add_argument('-l',
                    '--quote_history',
                    default=5,
                    type=int,
                    help='Length of history for quotes L: [t-L,...,t]')
parser.add_argument('-d',
                    '--delay_in_volume_reporting',
                    default=0,
                    type=int,
                    help='Volume reported at time t: [t-L-d,...,t-d]')
parser.add_argument('-m',
                    '--trade_history',
                    default=None,
                    type=int,
                    help='Length of history for volumes M: [t-d-M,...,t-d]')
parser.add_argument('-tf',
                    '--tune_flag',
                    default=1,
                    type=int,
                    help='0: Use env.step with fixed actions; 1: Use ray.tune for running expts')
args, remaining_args = parser.parse_known_args()
mm_add_volume = args.mm_add_volume
pt_add_volume = args.pt_add_volume
pt_add_momentum = 1
num_pts = 1
L = args.quote_history
M = args.trade_history if args.trade_history is not None else L
d = args.delay_in_volume_reporting
tune_flag = True if args.tune_flag else False
a2c_flag = False
timestep_duration = "60S"
register_env(
    "marl-v0",
    lambda config: SubGymMultiAgentRLEnv_v0(**config),
)
ray.shutdown()
ray.init() 
# #Causes IOError: [RayletClient] Unable to register worker with raylet. No such file or directory!
if a2c_flag:
    name_xp = "a2c_"
else:
    name_xp = 'ppo_'
name_xp += f"marl_vol{mm_add_volume}{pt_add_volume}_L{L}_d{d}_M{M}_pts{num_pts}"
base_log_dir = os.getcwd() + '/results/' + name_xp

env_config = multi_agent_init(num_pts,mm_add_volume,pt_add_volume,L,d,M,
                base_log_dir,pt_add_momentum=pt_add_momentum,
                timestep_duration=timestep_duration)
# env_config["linear_oracle"] = True
multiagent_policies = multi_agent_policies(env_config['learning_agent_ids'],
                        env_config['observation_space'],env_config['action_space'])

if tune_flag:
    config = {
            "env": "marl-v0",
            "env_config": env_config,
            "seed": 0,
            "num_gpus": 0,
            "num_workers": 0,
            "multiagent": multiagent_policies,
            # "framework": "tf",
            "horizon": int((str_to_ns(env_config["mkt_close"]) - \
                        str_to_ns(env_config["mkt_open"]))/str_to_ns(env_config["timestep_duration"]))
        }

    stop = {
        "training_iteration": 1000
    }

    tune.run(
        "A2C" if a2c_flag else "PPO",
        name=name_xp,
        resume=False,
        stop=stop,
        checkpoint_at_end=True,
        checkpoint_freq=20,
        config=config,
        verbose=1,
    )
else:
    register(
    id="marl-v0",
    entry_point=SubGymMultiAgentRLEnv_v0,
    )
    env = gym.make(
    "marl-v0",
    background_config="rmsc04",
    **env_config,
    )
    random_action = {'MM': np.array([0,2,1]),'PT1': np.array([0,0]),'PT2': np.array([2,2]),'PT3': np.array([0,2])}
    env.seed(0)
    initial_state = env.reset()
    for i in range(500):
        print(f'\niter {i}')
        state, reward, done, info = env.step(random_action)
        # print(state,random_action,reward)