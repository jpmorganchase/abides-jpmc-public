from copy import deepcopy
from abc import abstractmethod, ABC
from typing import Any, Callable, Dict, List, Optional, Tuple

import gym
import numpy as np
from gym.utils import seeding

import abides_markets.agents.utils as markets_agent_utils
from abides_core import Kernel, NanosecondTime
from abides_core.generators import InterArrivalTimeGenerator
from abides_core.utils import subdict
from abides_markets.utils import config_add_agents
from .core_environment import AbidesGymCoreEnv

from ..experimental_agents.financial_gym_agent import FinancialGymAgent


class AbidesGymMarketsEnv(AbidesGymCoreEnv, ABC):
    """
    Abstract class for markets gym to inherit from to create usable specific ABIDES Gyms

    Arguments:
        - background_config_pair: tuple consisting in the background builder function and the inputs to use
        - wakeup_interval_generator: generator used to compute delta time wakeup for the gym experimental agent
        - starting_cash: cash of the agents at the beginning of the simulation
        - state_history_length: length of the raw state buffer
        - market_data_buffer_length: length of the market data buffer
        - first_interval: how long the simulation is run before the first wake up of the gym experimental agent
        - raw_state_pre_process: decorator used to pre-process raw_state

    """

    raw_state_pre_process = markets_agent_utils.identity_decorator

    def __init__(
        self,
        background_config_pair: Tuple[Callable, Optional[Dict[str, Any]]],
        wakeup_interval_generator: InterArrivalTimeGenerator,
        starting_cash: int,
        state_buffer_length: int,
        market_data_buffer_length: int,
        first_interval: Optional[NanosecondTime] = None,
        raw_state_pre_process=markets_agent_utils.identity_decorator,
    ) -> None:
        super().__init__(
            background_config_pair,
            wakeup_interval_generator,
            state_buffer_length,
            first_interval=first_interval,
            gymAgentConstructor=FinancialGymAgent,
        )
        self.starting_cash: int = starting_cash
        self.market_data_buffer_length: int = market_data_buffer_length
        self.extra_gym_agent_kvargs = {
            "starting_cash": self.starting_cash,
            "market_data_buffer_length": self.market_data_buffer_length,
        }
        self.extra_background_config_kvargs = {
            "exchange_log_orders": False,
            "book_logging": False,  # may need to set to True if wants to return OB in terminal state when episode ends (gym2)
            "log_orders": None,
        }
