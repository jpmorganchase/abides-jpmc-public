from collections import deque
from copy import deepcopy
from typing import Any, Deque, Dict, List, Optional, Tuple
from matplotlib.pyplot import isinteractive

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.generators import ConstantTimeGenerator, InterArrivalTimeGenerator
from abides_core.utils import fmt_ts, str_to_ns
from abides_markets.agents.trading_agent import TradingAgent
from abides_markets.messages.marketdata import (
    MarketDataMsg,
    L2SubReqMsg,
    TransactedVolDataMsg,
)
from abides_markets.messages.marketdata import (
    L2DataMsg,
    L2SubReqMsg,
    TransactedVolSubReqMsg,
)
from abides_markets.messages.orderbook import (
    OrderAcceptedMsg, 
    OrderExecutedMsg
)
from abides_markets.messages.marl import OrderMatchedWithWhomMsg, OrderMatchedValueAgentMsg
from abides_markets.orders import Order, Side

class CoreBackgroundAgent(TradingAgent):
    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        subscribe_freq: int = int(1e8),
        lookback_period: Optional[int] = None,  # for volume subscription
        subscribe: bool = True,
        subscribe_num_levels: Optional[int] = None,
        wakeup_interval_generator: InterArrivalTimeGenerator = ConstantTimeGenerator(
            step_duration=str_to_ns("1min")
        ),
        order_size_generator=None,  # TODO: not sure about this one
        state_buffer_length: int = 2,
        market_data_buffer_length: int = 5,
        first_interval: Optional[NanosecondTime] = None,
        log_orders: bool = False,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        num_noise_agents: int = None, 
        num_value_agents: int = None, 
        num_momentum_agents: int = None,
        max_iter: int = None,
        num_pts: int = 1,
    ) -> None:
        super().__init__(
            id,
            starting_cash=starting_cash,
            log_orders=log_orders,
            name=name,
            type=type,
            random_state=random_state,
        )
        self.symbol: str = symbol
        # Frequency of agent data subscription up in ns-1
        self.subscribe_freq: int = subscribe_freq
        self.subscribe: bool = subscribe
        self.subscribe_num_levels: int = subscribe_num_levels
        self.first_interval: Optional[NanosecondTime] = first_interval
        self.wakeup_interval_generator: InterArrivalTimeGenerator = (
            wakeup_interval_generator
        )
        self.order_size_generator = (
            order_size_generator  # TODO: no diea here for typing
        )

        if hasattr(self.wakeup_interval_generator, "random_generator"):
            self.wakeup_interval_generator.random_generator = self.random_state

        self.state_buffer_length: int = state_buffer_length
        self.market_data_buffer_length: int = market_data_buffer_length
        self.first_interval: Optional[NanosecondTime] = first_interval
        if self.order_size_generator != None:  # TODO: check this one
            self.order_size_generator.random_generator = self.random_state

        self.lookback_period: NanosecondTime = self.wakeup_interval_generator.mean()

        # internal variables
        self.has_subscribed: bool = False
        self.episode_executed_orders: List[
            Order
        ] = []  # list of executed orders during full episode

        # list of executed orders between steps - is reset at every step
        self.inter_wakeup_executed_orders: List[
            Order
        ] = []  # list of executed orders between steps - is reset at every step
        self.parsed_episode_executed_orders: List[Tuple[int, int]] = []  # (price, qty)
        self.parsed_inter_wakeup_executed_orders: List[
            Tuple[int, int]
        ] = []  # (price, qty)
        self.parsed_mkt_data: Dict[str, Any] = {}
        self.parsed_mkt_data_buffer: Deque[Dict[str, Any]] = deque(
            maxlen=self.market_data_buffer_length
        )
        self.parsed_volume_data = {}
        self.parsed_volume_data_buffer: Deque[Dict[str, Any]] = deque(
            maxlen=self.market_data_buffer_length
        )
        self.raw_state: Deque[Dict[str, Any]] = deque(maxlen=self.state_buffer_length)
        # dictionary to track order status:
        # - keys = order_id
        # - value = dictionary {'active'|'cancelled'|'executed', Order, 'active_qty','executed_qty', 'cancelled_qty }
        self.order_status: Dict[int, Dict[str, Any]] = {}
        self.num_noise_agents = num_noise_agents
        self.num_value_agents = num_value_agents
        self.num_momentum_agents = num_momentum_agents
        self.num_pts = num_pts
        self.max_iter = max_iter
        self.matching_agents = np.zeros(4+self.num_pts,dtype=int) 
        # number of orders matched by 0: Noise agent, 1: Value agent, 2: Momentum agent, 3: MM, 4,..: PT
        self.matched_value_agent_orders = np.zeros((2,self.max_iter)) 
        #0,t: price at which order was executed within tth interval; 1,t: side \in [0,2]

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)

    def wakeup(self, current_time: NanosecondTime, act_on_wakeup_flag = False) -> bool:
        # TODO: parent class (TradingAgent) returns bool of "ready to trade"
        """Agent interarrival wake up times are determined by wakeup_interval_generator"""
        super().wakeup(current_time)
        if not self.has_subscribed:
            super().request_data_subscription(
                L2SubReqMsg(
                    symbol=self.symbol,
                    freq=self.subscribe_freq,
                    depth=self.subscribe_num_levels,
                )
            )
            super().request_data_subscription(
                TransactedVolSubReqMsg(
                    symbol=self.symbol,
                    freq=self.subscribe_freq,
                    lookback=self.lookback_period,
                )
            )

            self.has_subscribed = True
        # compute the following wake up
        if act_on_wakeup_flag:
            if (self.mkt_open != None) and (
                current_time >= self.mkt_open
            ):  # compute the state (returned to the Gym Env)
                raw_state = self.act_on_wakeup()
                # TODO: wakeup function should return bool
                return raw_state

    ##return non None value so the kernel catches it and stops
    # return raw_state

    def act_on_wakeup(self):
        # Needs type signature
        raise NotImplementedError

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        """Processes message from exchange. Main function is to update orders in orderbook relative to mid-price.
        :param simulation current time
        :param message received by self from ExchangeAgent
        :type current_time: pd.Timestamp
        :type message: str
        :return:
        """
        # if self.id == 114 and isinstance(message, OrderExecutedMsg):
        #     print('MM received msg from kernel at: ',message,fmt_ts(current_time),sender_id)
        # TODO: will prob need to see for transacted volume if we enrich the state
        super().receive_message(current_time, sender_id, message)
        if self.subscribe:
            if isinstance(message, MarketDataMsg):
                if isinstance(message, L2DataMsg):
                    self.parsed_mkt_data = self.get_parsed_mkt_data(message)
                    self.parsed_mkt_data_buffer.append(self.parsed_mkt_data)
                elif isinstance(message, TransactedVolDataMsg):
                    self.parsed_volume_data = self.get_parsed_volume_data(message)
                    self.parsed_volume_data_buffer.append(self.parsed_volume_data)
        if isinstance(message, OrderMatchedWithWhomMsg):
            # print(f'{self.id} received matching msg {message}')
            if 1 <= message.matching_agent_id <= self.num_noise_agents: #Matched by Noise Agent
                self.matching_agents[0] += 1
            elif 1 + self.num_noise_agents <= message.matching_agent_id <= self.num_noise_agents + self.num_value_agents: #Matched by Value Agent
                self.matching_agents[1] += 1
            elif 1 + self.num_noise_agents + self.num_value_agents <= message.matching_agent_id <= self.num_noise_agents + self.num_value_agents + self.num_momentum_agents: #Matched by Momentum Agent
                self.matching_agents[2] += 1
            elif message.matching_agent_id == self.num_noise_agents + self.num_value_agents + self.num_momentum_agents + 2: #Matched by MM
                self.matching_agents[3] += 1
            elif message.matching_agent_id >= self.num_noise_agents + self.num_value_agents + self.num_momentum_agents + 3: #Matched by PT
                self.matching_agents[message.matching_agent_id - self.num_noise_agents - \
                    self.num_value_agents - self.num_momentum_agents + 1] += 1
            else:
                print(f'Order of {self.id} matched by unknown trader!!!!')
            # print(self.matching_agents)
        if isinstance(message, OrderMatchedValueAgentMsg):
            t = int((current_time - str_to_ns(self.mkt_open))/self.wakeup_interval_generator.step_duration)
            if t >= self.max_iter:
                t = self.max_iter - 1
            self.matched_value_agent_orders[0,t] = message.price
            if message.side == "SELL":
                self.matched_value_agent_orders[1,t] = 0
            elif message.side == "BUY":
                self.matched_value_agent_orders[1,t] = 2
            
        # elif isinstance(message, OrderAcceptedMsg):
        #     self.order_accepted(message.order)

    def get_wake_frequency(self) -> NanosecondTime:
        # first wakeup interval from open
        time_first_wakeup = (
            self.first_interval
            if self.first_interval != None
            else self.wakeup_interval_generator.next()
        )
        return time_first_wakeup

    def apply_actions(self, actions: List[Dict[str, Any]]) -> None:
        # take action from kernel in general representation
        # convert in ABIDES-SIMULATOR API
        # print(actions)
        # TODO Add cancel in actions
        # print(actions)
        # if self.id == 114:
        #     print('applying actions of MM at: ',fmt_ts(self.current_time))
        for action in actions:
            if action["type"] == "MKT":
                side = Side.BID if action["direction"] == "BUY" else Side.ASK
                # print(action['direction'])
                # print(side)
                self.place_market_order(self.symbol, action["size"], side)
            elif action["type"] == "LMT":
                side = Side.BID if action["direction"] == "BUY" else Side.ASK
                self.place_limit_order(
                    self.symbol, action["size"], side, action["limit_price"]
                )

            # TODO: test the cancel based on the id
            elif action["type"] == "CCL_ALL":
                # order = self.order_status[action['order_id']]
                self.cancel_all_orders()

            else:
                raise ValueError(f"Action Type {action['type']} is not supported")

    def update_raw_state(self) -> None:
        # mkt data
        parsed_mkt_data_buffer = deepcopy(self.parsed_mkt_data_buffer)
        # internal data
        internal_data = self.get_internal_data()
        # volume data
        parsed_volume_data_buffer = deepcopy(self.parsed_volume_data_buffer)

        new = {
            "parsed_mkt_data": parsed_mkt_data_buffer,
            "internal_data": internal_data,
            "parsed_volume_data": parsed_volume_data_buffer,
        }
        self.raw_state.append(new)

    def get_raw_state(self) -> Dict:
        # TODO: Incompatible return value type (got "deque[Any]", expected "Dict[Any, Any]")
        return self.raw_state

    def get_parsed_mkt_data(self, message: L2DataMsg) -> Dict[str, Any]:
        # TODO: probaly will need to include what type of subscription in parameters here
        bids = message.bids
        asks = message.asks
        last_transaction = message.last_transaction
        exchange_ts = message.exchange_ts
        mkt_data = {
            "bids": bids,
            "asks": asks,
            "last_transaction": last_transaction,
            "exchange_ts": exchange_ts,
        }
        return mkt_data

    def get_parsed_volume_data(self, message: TransactedVolDataMsg) -> Dict[str, Any]:
        last_transaction = message.last_transaction
        exchange_ts = message.exchange_ts
        bid_volume = message.bid_volume
        ask_volume = message.ask_volume
        total_volume = bid_volume + ask_volume

        volume_data = {
            "last_transaction": last_transaction,
            "exchange_ts": exchange_ts,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "total_volume": total_volume,
        }
        return volume_data

    def get_internal_data(self) -> Dict[str, Any]:
        holdings = self.get_holdings(self.symbol)
        cash = self.get_holdings("CASH")
        inter_wakeup_executed_orders = self.inter_wakeup_executed_orders
        episode_executed_orders = self.episode_executed_orders
        parsed_episode_executed_orders = self.parsed_episode_executed_orders
        parsed_inter_wakeup_executed_orders = self.parsed_inter_wakeup_executed_orders
        current_time = self.current_time
        order_status = self.order_status
        mkt_open = self.mkt_open
        mkt_close = self.mkt_close
        internal_data = {
            "holdings": holdings,
            "cash": cash,
            "inter_wakeup_executed_orders": inter_wakeup_executed_orders,
            "episode_executed_orders": episode_executed_orders,
            "parsed_episode_executed_orders": parsed_episode_executed_orders,
            "parsed_inter_wakeup_executed_orders": parsed_inter_wakeup_executed_orders,
            "starting_cash": self.starting_cash,
            "current_time": current_time,
            "order_status": order_status,
            "mkt_open": mkt_open,
            "mkt_close": mkt_close,
        }
        # if self.id == 114:
        #     print('Getting internal data of MM with inter orders at: ',\
        #         inter_wakeup_executed_orders,fmt_ts(self.current_time))
        return internal_data

    def order_executed(self, order: Order) -> None:
        super().order_executed(order)
        # parsing of the order message
        executed_qty = order.quantity
        executed_price = order.fill_price
        assert executed_price is not None
        order_id = order.order_id
        # step lists
        self.inter_wakeup_executed_orders.append(order)
        self.parsed_inter_wakeup_executed_orders.append((executed_qty, executed_price))
        # episode lists
        self.episode_executed_orders.append(order)
        self.parsed_episode_executed_orders.append((executed_qty, executed_price))
        # update order status dictionnary
        # test if it was mkt order and first execution received from it
        try:
            self.order_status[order_id]
            flag = True
        except KeyError:
            flag = False

        if flag:
            self.order_status[order_id]["executed_qty"] += executed_qty
            self.order_status[order_id]["active_qty"] -= executed_qty
            if self.order_status[order_id]["active_qty"] <= 0:
                self.order_status[order_id]["status"] = "executed"
        else:
            self.order_status[order_id] = {
                "status": "mkt_immediately_filled",
                "order": order,
                "active_qty": 0,
                "executed_qty": executed_qty,
                "cancelled_qty": 0,
            }
        # if self.id == 114:
        #     print('Inter wakeup executed orders for MM at: ',self.inter_wakeup_executed_orders,fmt_ts(self.current_time))

    def order_accepted(self, order: Order) -> None:
        super().order_accepted(order)
        # update order status dictionnary
        self.order_status[order.order_id] = {
            "status": "active",
            "order": order,
            "active_qty": order.quantity,
            "executed_qty": 0,
            "cancelled_qty": 0,
        }

    def order_cancelled(self, order: Order) -> None:
        super().order_cancelled(order)
        order_id = order.order_id
        quantity = order.quantity
        self.order_status[order_id] = {
            "status": "cancelled",
            "order": order,
            "cancelled_qty": quantity,
        }

    def new_inter_wakeup_reset(self) -> None:
        self.inter_wakeup_executed_orders = (
            []
        )  # list of executed orders between steps - is reset at every step
        self.parsed_inter_wakeup_executed_orders = []  # just tuple (price, qty)

    def act(self, raw_state):
        # used by the background agent
        raise NotImplementedError

    def new_step_reset(self) -> None:
        self.inter_wakeup_executed_orders = (
            []
        )  # list of executed orders between steps - is reset at every step
        self.parsed_inter_wakeup_executed_orders = []  # just tuple (price, qty)
