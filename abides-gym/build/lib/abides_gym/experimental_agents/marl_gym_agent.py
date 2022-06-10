from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Deque

import numpy as np

from abides_core import NanosecondTime, Message
from abides_core.utils import str_to_ns, fmt_ts
from abides_core.generators import ConstantTimeGenerator, InterArrivalTimeGenerator
from abides_markets.agents.background_v2.core_background_agent import (
    CoreBackgroundAgent,
)
from abides_markets.messages.marketdata import (
    L2DataMsg,
    L2SubReqMsg,
    TransactedVolSubReqMsg,
)
from abides_markets.orders import Order

from abides_markets.agents.background_v2.core_background_agent import CoreBackgroundAgent

class MultiAgentGymAgent(CoreBackgroundAgent):
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
        - market_data_buffer_length: length of the buffer for the received market data

    Inidividual functions:
    - get_wake_freq:
    - act

    Functions corresponding to contained learning agents:
    - kernel_starting
    - receive_message
    - apply_actions
    - get_internal_data
    - get_parsed_mkt_data
    - get_parsed_volume_data
    - order_executed: only for that contained agent whose order is executed
    - order_accepted: only for that contained agent whose order is executed
    - order_cancelled: only for that contained agent whose order is executed
    - new_inter_wakeup_reset
    - new_step_reset

    Mixed
    - wakeup
    - act_on_wakeup
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        name: str,
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
        random_state: Optional[np.random.RandomState] = None,
        ## Info about contained learning agents
        names: Optional[List[str]] = None, #names = agent ids
        types: Optional[List[str]] = None,
        num_pts: int = 1,
        starting_cashs: List = [1_000_000,1_000_000],
        num_noise_agents: int = None, 
        num_value_agents: int = None, 
        num_momentum_agents: int = None,
        max_iter: int = None,
    ) -> None:
        ''' Only initialization is for this individual agent, all other 
        functionality is that of its contained learning agents'''
        
        super().__init__(
            id,
            symbol=symbol,
            starting_cash=starting_cash,
            log_orders=log_orders,
            name=name,
            random_state=random_state,
            max_iter=max_iter,
            wakeup_interval_generator=wakeup_interval_generator,
            state_buffer_length=state_buffer_length,
            market_data_buffer_length=market_data_buffer_length,
            first_interval=first_interval,
            subscribe=subscribe,
            subscribe_num_levels=subscribe_num_levels,
            subscribe_freq=subscribe_freq,
            num_noise_agents=num_noise_agents, 
            num_value_agents=num_value_agents, 
            num_momentum_agents=num_momentum_agents, 
            num_pts=num_pts,           
        )
        ## Info about contained learning agents
        self.names = names
        self.types = types
        self.starting_cashs = starting_cashs
        self.num_pts = num_pts
        self.agents = {}

        for i in range(len(self.names)): #for every agent represented by gym agent
            self.agents[i] = CoreBackgroundAgent(
                self.id + i + 1,
                symbol=symbol,
                starting_cash=self.starting_cashs[i],
                log_orders=log_orders,
                name=self.names[i] if self.names is not None else None,
                type=self.types[i] if self.types is not None else None,
                random_state=random_state,
                wakeup_interval_generator=wakeup_interval_generator,
                state_buffer_length=state_buffer_length,
                market_data_buffer_length=market_data_buffer_length,
                first_interval=first_interval,
                max_iter=max_iter,
                subscribe=subscribe,
                subscribe_num_levels=subscribe_num_levels,
                subscribe_freq=subscribe_freq,
                num_noise_agents=num_noise_agents, 
                num_value_agents=num_value_agents, 
                num_momentum_agents=num_momentum_agents,
                num_pts=num_pts,
            )
        
        self.raw_state: Dict[Deque[Dict[str, Any]]] = {}
        # self.raw_state = {}
        for i in range(len(self.names)):
            self.raw_state[self.names[i]] = self.agents[i].raw_state
        
    def kernel_starting(self, start_time: NanosecondTime) -> None:
        ### For contained agents
        super().kernel_starting(start_time + self.first_interval)
        for i in range(len(self.names)):
            # print(f'starting kernel for {self.agents[i].id}')
            self.agents[i].kernel_starting(start_time + self.first_interval)

    def wakeup(self, current_time: NanosecondTime):
        # print(f'in wakeup of {self.name} at {fmt_ts(current_time)}')
        super().wakeup(current_time)
        for i in range(len(self.names)):
            # print(f'GYM agent waking up {self.names[i]}')
            self.agents[i].wakeup(current_time)
        
        # compute the following wake up
        if (self.mkt_open != None) and (current_time >= self.mkt_open):  # compute the state (returned to the Gym Env)
            raw_state = self.act_on_wakeup()
            # print(f'{raw_state} returned from act on wakeup')
            return raw_state

    def act_on_wakeup(self):
        # print('in act on wakeup of gym agent')
        # for i in range(len(self.names)):
        #     # compute the state (returned to the Gym Env)
        #     # wakeup logic
        #     wake_time = (self.agents[i].current_time + self.agents[i].wakeup_interval_generator.next())  # generates next wakeup time
        #     self.agents[i].set_wakeup(wake_time)
        self.update_raw_state()
        raw_state = deepcopy(self.get_raw_state())
        self.new_step_reset()
        # return non None value so the kernel catches it and stops
        return raw_state

    def receive_message(self, current_time: NanosecondTime, sender_id: int, message: Message) -> None:
        # print('in receive msg of GYM')
        super().receive_message(current_time,sender_id,message)
        # if self.mkt_open is not None and self.mkt_close is not None:
        # # Set first wakeup of gym agent since receive meesgae does it only for contained agents and next wakeups
        #     self.set_wakeup(self.mkt_open + self.get_wake_frequency())
        for i in range(len(self.names)):
            self.agents[i].receive_message(current_time,sender_id,message)
        
    def apply_actions(self, actions_list: Dict[List[Dict[str, Any]],Any]) -> None: 
        # print(f'in apply actions of {self.name}')
        for i in range(len(self.names)): #for each learning agent
            # if i == 0: #Cancel all MM/PT orders before placing more
            self.agents[i].cancel_all_orders()
            agent_id = self.names[i]
            self.agents[i].apply_actions(actions_list[agent_id])
        self.set_wakeup(self.current_time + self.wakeup_interval_generator.next())

    def update_raw_state(self) -> None:
        # print(f'in update raw state of {self.name}')
        for i in range(len(self.names)):
            # mkt data
            parsed_mkt_data_buffer = deepcopy(self.agents[i].parsed_mkt_data_buffer)
            # internal data
            internal_data = self.agents[i].get_internal_data()
            # volume data
            parsed_volume_data_buffer = deepcopy(self.agents[i].parsed_volume_data_buffer)

            new = {
                "parsed_mkt_data": parsed_mkt_data_buffer,
                "internal_data": internal_data,
                "parsed_volume_data": parsed_volume_data_buffer,
            }
            self.raw_state[self.names[i]].append(new)

    def get_raw_state(self) -> Dict:
        return self.raw_state

    def order_executed(self, order: Order) -> None:
        for i in range(len(self.names)):
            if self.agents[i].id == order.agent_id:
                self.agents[i].order_executed(order)
                break

    def order_accepted(self, order: Order) -> None:
        for i in range(len(self.names)):
            if self.agents[i].id == order.agent_id:
                self.agents[i].order_accepted(order)
                break

    def order_cancelled(self, order: Order) -> None:
        for i in range(len(self.names)):
            if self.agents[i].id == order.agent_id:
                self.agents[i].order_cancelled(order)
                break

    def new_inter_wakeup_reset(self) -> None:
        for i in range(len(self.names)):
            self.agents[i].new_inter_wakeup_reset()

    def new_step_reset(self) -> None:
        for i in range(len(self.names)):
            self.agents[i].new_step_reset()
    
    def get_matching_agents(self) -> Dict:
        matching_agents = {}
        for i in range(len(self.names)):
            matching_agents[self.names[i]] = self.agents[i].matching_agents
            # print(matching_agents)
        return matching_agents

    def get_matched_value_agent_orders(self) -> np.array:
        return self.agents[0].matched_value_agent_orders
