from email import policy
import os
import time
from matplotlib import tri
import ray
from ray.tune.registry import register_env
from gym.envs.registration import register
from abides_gym.envs.marl_environment_v0 import SubGymMultiAgentRLEnv_v0
import ray.rllib.agents.ppo as ppo
import argparse
from abides_core.utils import str_to_ns
from scripts.marl_utils import multi_agent_init, multi_agent_policies, \
    multi_agent_policy_MARL, run_episode, log_results

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
                    help='Length of history for state L: [t-L,...,t]')
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
momentum_agent_freq = "5min"
lambda_a = 5.7e-12

register(
    id="marl-v0",
    entry_point=SubGymMultiAgentRLEnv_v0,
    )
register_env(
    "marl-v0",
    lambda config: SubGymMultiAgentRLEnv_v0(**config),
)
ray.shutdown()
ray.init()

log_flag = False #For the marl env
env_config = multi_agent_init(num_pts,mm_add_volume,pt_add_volume,L,d,M,'',log_flag,pt_add_momentum=pt_add_momentum)
# env_config["linear_oracle"] = True
multiagent_policies = multi_agent_policies(env_config['learning_agent_ids'],
                        env_config['observation_space'],env_config['action_space'])
horizon = int((str_to_ns(env_config["mkt_close"]) - \
                str_to_ns(env_config["mkt_open"]))/str_to_ns(env_config["timestep_duration"]))

num_test_ep = int(50)
policy_folder = f"~/ray_results/ppo_marl_vol{mm_add_volume}{pt_add_volume}_L{L}_d{d}_M{M}_pts{num_pts}"
# print(policy_folder)
config = {}
config = ppo.DEFAULT_CONFIG.copy()
config["env_config"] = env_config
config["multiagent"] = multiagent_policies
config["framework"] = "tf"
config["horizon"] = horizon
trained_policy = multi_agent_policy_MARL(policy_folder,config,'marl-v0')

name_xp = f"ppo_marl_vol{mm_add_volume}{pt_add_volume}_L{L}_d{d}_M{M}_pts{num_pts}_momentum_agent_freq{momentum_agent_freq}_lambda_a{lambda_a}"
test_log_dir = os.getcwd() + '/results/test/' + name_xp + '/'
start_time = time.time()
for i in range(num_test_ep):
    print(f'\nTest Episode: {i+1}/{num_test_ep}\n')
    obs_dict, matching_agents, matched_value_agent_orders, agent_pnls = run_episode(num_pts,env_config,
                                                                            trained_policy,horizon,
                                                                            i,env_config["add_volume"],
                                                                            L,M,pt_add_momentum)
    log_results(i,obs_dict,test_log_dir,num_pts,matching_agents,matched_value_agent_orders,agent_pnls)
    if i == 0:
        print(f'Time for first episode test: {(time.time() - start_time)/60} minutes')
print(f'Time to test learnt policies: {(time.time() - start_time)/60} minutes')
