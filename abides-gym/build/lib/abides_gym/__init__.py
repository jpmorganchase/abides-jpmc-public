from gym.envs.registration import register
from ray.tune.registry import register_env

from .envs import *


# REGISTER ENVS FOR GYM USE

register(
    id="markets-daily_investor-v0",
    entry_point=SubGymMarketsDailyInvestorEnv_v0,
)

register(
    id="markets-execution-v0",
    entry_point=SubGymMarketsExecutionEnv_v0,
)


# REGISTER ENVS FOR RAY/RLLIB USE

register_env(
    "markets-daily_investor-v0",
    lambda config: SubGymMarketsDailyInvestorEnv_v0(**config),
)

register_env(
    "markets-execution-v0",
    lambda config: SubGymMarketsExecutionEnv_v0(**config),
)
