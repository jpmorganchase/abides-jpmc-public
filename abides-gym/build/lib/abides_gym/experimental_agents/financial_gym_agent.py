from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from abides_core.generators import ConstantTimeGenerator, InterArrivalTimeGenerator
from abides_markets.agents.background_v2.core_background_agent import (
    CoreBackgroundAgent,
)
from abides_markets.orders import Order

from .core_gym_agent import CoreGymAgent


class FinancialGymAgent(CoreBackgroundAgent, CoreGymAgent):
    """
    Gym experimental agent class. This agent is the interface between the ABIDES simulation and the ABIDES Gym environments.

    Arguments:
        - id: agents id in the simulation
        - symbol: ticker of the traded asset
        - starting_cash: agent's cash at the beginning of the simulation
        - subscribe_freq: frequency the agents receives market data from the exchange
        - subscribe: flag if the agent subscribe or not to market data
        - subscribe_num_levels: number of level depth in the OB the agent subscribes to
        - wakeup_interval_generator: inter-wakeup generator for agents next wakeup generation
        - state_buffer_length: length of the buffer of the agent raw_states
        _ market_data_buffer_length: length of the buffer for the received market data


    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        subscribe_freq: int = int(1e8),
        subscribe: float = True,
        subscribe_num_levels: int = 10,
        wakeup_interval_generator: InterArrivalTimeGenerator = ConstantTimeGenerator(
            step_duration=str_to_ns("1min")
        ),
        state_buffer_length: int = 2,
        market_data_buffer_length: int = 5,
        first_interval: Optional[NanosecondTime] = None,
        log_orders: bool = False,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__(
            id,
            symbol=symbol,
            starting_cash=starting_cash,
            log_orders=log_orders,
            name=name,
            type=type,
            random_state=random_state,
            wakeup_interval_generator=wakeup_interval_generator,
            state_buffer_length=state_buffer_length,
            market_data_buffer_length=market_data_buffer_length,
            first_interval=first_interval,
            subscribe=subscribe,
            subscribe_num_levels=subscribe_num_levels,
            subscribe_freq=subscribe_freq,
        )
        self.symbol: str = symbol
        # Frequency of agent data subscription up in ns-1
        self.subscribe_freq: int = subscribe_freq
        self.subscribe: bool = subscribe
        self.subscribe_num_levels: int = subscribe_num_levels

        self.wakeup_interval_generator: InterArrivalTimeGenerator = (
            wakeup_interval_generator
        )
        self.lookback_period: NanosecondTime = self.wakeup_interval_generator.mean()

        if hasattr(self.wakeup_interval_generator, "random_generator"):
            self.wakeup_interval_generator.random_generator = self.random_state

        self.state_buffer_length: int = state_buffer_length
        self.market_data_buffer_length: int = market_data_buffer_length
        self.first_interval: Optional[NanosecondTime] = first_interval
        # internal variables
        self.has_subscribed: bool = False
        self.episode_executed_orders: List[
            Order
        ] = []  # list of executed orders during full episode

        # list of executed orders between steps - is reset at every step
        self.inter_wakeup_executed_orders: List[Order] = []
        self.parsed_episode_executed_orders: List[Tuple[int, int]] = []  # (price, qty)
        self.parsed_inter_wakeup_executed_orders: List[
            Tuple[int, int]
        ] = []  # (price, qty)
        self.parsed_mkt_data: Dict[str, Any] = {}
        self.parsed_mkt_data_buffer = deque(maxlen=self.market_data_buffer_length)
        self.parsed_volume_data = {}
        self.parsed_volume_data_buffer = deque(maxlen=self.market_data_buffer_length)
        self.raw_state = deque(maxlen=self.state_buffer_length)
        # dictionary to track order status:
        # - keys = order_id
        # - value = dictionary {'active'|'cancelled'|'executed', Order, 'active_qty','executed_qty', 'cancelled_qty }
        self.order_status: Dict[int, Dict[str, Any]] = {}

    def act_on_wakeup(self) -> Dict:
        """
        Computes next wakeup time, computes the new raw_state and clears the internal step buffers.
        Returns the raw_state to the abides gym environnement (outside of the abides simulation) where the next action will be selected.


        Returns:
            - the raw_state dictionnary that will be processed in the abides gym subenvironment
        """
        # compute the state (returned to the Gym Env)
        # wakeup logic
        wake_time = (
            self.current_time + self.wakeup_interval_generator.next()
        )  # generates next wakeup time
        self.set_wakeup(wake_time)
        self.update_raw_state()
        raw_state = deepcopy(self.get_raw_state())
        self.new_step_reset()
        # return non None value so the kernel catches it and stops
        return raw_state
