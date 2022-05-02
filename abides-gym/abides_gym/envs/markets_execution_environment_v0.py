import importlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List
from abc import ABC

import gym
import numpy as np

import abides_markets.agents.utils as markets_agent_utils
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from abides_core.generators import ConstantTimeGenerator

from .markets_environment import AbidesGymMarketsEnv


class SubGymMarketsExecutionEnv_v0(AbidesGymMarketsEnv):
    """
    Execution V0 environnement. It defines one of the ABIDES-Gym-markets environnement.
    This environment presents an example of the algorithmic orderexecution problem.
    The agent has either an initial inventory of the stocks it tries to trade out of or no initial inventory and
    tries to acquire a target number of shares. The goal is to realize thistask while minimizing transaction cost from spreads
     and marketimpact. It does so by splitting the parent order into several smallerchild orders.

    Arguments:
        - background_config: the handcrafted agents configuration used for the environnement
        - mkt_close: time the market day ends
        - timestep_duration: how long between 2 wakes up of the gym experimental agent
        - starting_cash: cash of the agents at the beginning of the simulation
        - order_fixed_size: size of the order placed by the experimental gym agent
        - state_history_length: length of the raw state buffer
        - market_data_buffer_length: length of the market data buffer
        - first_interval: how long the simulation is run before the first wake up of the gym experimental agent
        - parent_order_size: Total size the agent has to execute (eitherbuy or sell).
        - execution_window: Time length the agent is given to proceed with 𝑝𝑎𝑟𝑒𝑛𝑡𝑂𝑟𝑑𝑒𝑟𝑆𝑖𝑧𝑒execution.
        - direction: direction of the 𝑝𝑎𝑟𝑒𝑛𝑡𝑂𝑟𝑑𝑒𝑟 (buy or sell)
        - not_enough_reward_update: it is a constant penalty per non-executed share atthe end of the𝑡𝑖𝑚𝑒𝑊𝑖𝑛𝑑𝑜𝑤
        - just_quantity_reward_update: update reward if all order is completed
        - reward_mode: can use a dense of sparse reward formulation
        - done_ratio: ratio (mark2market_t/starting_cash) that defines when an episode is done (if agent has lost too much mark to market value)
        - debug_mode: arguments to change the info dictionnary (lighter version if performance is an issue)
        - background_config_extra_kvargs: dictionary of extra key value  arguments passed to the background config builder function

    Daily Investor V0:
        - Action Space:
            - MKT order_fixed_size
            - LMT order_fixed_size
            - Hold
        - State Space:
            - holdings_pct
            - time_pct
            - diff_pct
            - imbalance_all
            - imbalance_5
            - price_impact
            - spread
            - direction
            - returns
    """

    raw_state_pre_process = markets_agent_utils.ignore_buffers_decorator
    raw_state_to_state_pre_process = (
        markets_agent_utils.ignore_mkt_data_buffer_decorator
    )

    @dataclass
    class CustomMetricsTracker(ABC):
        """
        Data Class used to track custom metrics that are output to rllib
        """

        slippage_reward: float = 0
        late_penalty_reward: float = 0  # at the end of the episode

        executed_quantity: int = 0  # at the end of the episode
        remaining_quantity: int = 0  # at the end of the episode

        action_counter: Dict[str, int] = field(default_factory=dict)

        holdings_pct: float = 0
        time_pct: float = 0
        diff_pct: float = 0
        imbalance_all: float = 0
        imbalance_5: float = 0
        price_impact: int = 0
        spread: int = 0
        direction_feature: float = 0
        num_max_steps_per_episode: float = 0

    def __init__(
        self,
        background_config: Any = "rmsc04",
        mkt_close: str = "16:00:00",
        timestep_duration: str = "60s",
        starting_cash: int = 1_000_000,
        order_fixed_size: int = 10,
        state_history_length: int = 4,
        market_data_buffer_length: int = 5,
        first_interval: str = "00:00:30",
        parent_order_size: int = 1000,
        execution_window: str = "00:10:00",
        direction: str = "BUY",
        not_enough_reward_update: int = -1000,
        too_much_reward_update: int = -100,
        just_quantity_reward_update: int = 0,
        debug_mode: bool = False,
        background_config_extra_kvargs: Dict[str, Any] = {},
    ) -> None:
        self.background_config: Any = importlib.import_module(
            "abides_markets.configs.{}".format(background_config), package=None
        )
        self.mkt_close: NanosecondTime = str_to_ns(mkt_close)
        self.timestep_duration: NanosecondTime = str_to_ns(timestep_duration)
        self.starting_cash: int = starting_cash
        self.order_fixed_size: int = order_fixed_size
        self.state_history_length: int = state_history_length
        self.market_data_buffer_length: int = market_data_buffer_length
        self.first_interval: NanosecondTime = str_to_ns(first_interval)
        self.parent_order_size: int = parent_order_size
        self.execution_window: str = str_to_ns(execution_window)
        self.direction: str = direction
        self.debug_mode: bool = debug_mode

        self.too_much_reward_update: int = too_much_reward_update
        self.not_enough_reward_update: int = not_enough_reward_update
        self.just_quantity_reward_update: int = just_quantity_reward_update

        self.entry_price: int = 1
        self.far_touch: int = 1
        self.near_touch: int = 1
        self.step_index: int = 0

        self.custom_metrics_tracker = (
            self.CustomMetricsTracker()
        )  # init the custom metric tracker

        ##################
        # CHECK PROPERTIES
        assert background_config in [
            "rmsc03",
            "rmsc04",
            "smc_01",
        ], "Select rmsc03 or rmsc04 as config"

        assert (self.first_interval <= str_to_ns("16:00:00")) & (
            self.first_interval >= str_to_ns("00:00:00")
        ), "Select authorized FIRST_INTERVAL delay"

        assert (self.mkt_close <= str_to_ns("16:00:00")) & (
            self.mkt_close >= str_to_ns("09:30:00")
        ), "Select authorized market hours"

        assert (self.timestep_duration <= str_to_ns("06:30:00")) & (
            self.timestep_duration >= str_to_ns("00:00:00")
        ), "Select authorized timestep_duration"

        assert (type(self.starting_cash) == int) & (
            self.starting_cash >= 0
        ), "Select positive integer value for starting_cash"

        assert (type(self.order_fixed_size) == int) & (
            self.order_fixed_size >= 0
        ), "Select positive integer value for order_fixed_size"

        assert (type(self.state_history_length) == int) & (
            self.state_history_length >= 0
        ), "Select positive integer value for order_fixed_size"

        assert (type(self.market_data_buffer_length) == int) & (
            self.market_data_buffer_length >= 0
        ), "Select positive integer value for order_fixed_size"

        assert self.debug_mode in [
            True,
            False,
        ], "debug_mode needs to be True or False"

        assert self.direction in [
            "BUY",
            "SELL",
        ], "direction needs to be BUY or SELL"

        assert (type(self.parent_order_size) == int) & (
            self.order_fixed_size >= 0
        ), "Select positive integer value for parent_order_size"

        assert (self.execution_window <= str_to_ns("06:30:00")) & (
            self.execution_window >= str_to_ns("00:00:00")
        ), "Select authorized execution_window"

        assert (
            type(self.too_much_reward_update) == int
        ), "Select integer value for too_much_reward_update"

        assert (
            type(self.not_enough_reward_update) == int
        ), "Select integer value for not_enough_reward_update"
        assert (
            type(self.just_quantity_reward_update) == int
        ), "Select integer value for just_quantity_reward_update"

        background_config_args = {"end_time": self.mkt_close}
        background_config_args.update(background_config_extra_kvargs)
        super().__init__(
            background_config_pair=(
                self.background_config.build_config,
                background_config_args,
            ),
            wakeup_interval_generator=ConstantTimeGenerator(
                step_duration=self.timestep_duration
            ),
            starting_cash=self.starting_cash,
            state_buffer_length=self.state_history_length,
            market_data_buffer_length=self.market_data_buffer_length,
            first_interval=self.first_interval,
        )

        # Action Space

        # MKT order_fixed_size | LMT order_fixed_size | Hold
        self.num_actions: int = 3
        self.action_space: gym.Space = gym.spaces.Discrete(self.num_actions)

        # instantiate the action counter
        for i in range(self.num_actions):
            self.custom_metrics_tracker.action_counter[f"action_{i}"] = 0

        num_ns_episode = self.first_interval + self.execution_window
        step_length = self.timestep_duration
        num_max_steps_per_episode = num_ns_episode / step_length
        self.custom_metrics_tracker.num_max_steps_per_episode = (
            num_max_steps_per_episode
        )

        # State Space
        # [holdings, imbalance,spread, direction_feature] + padded_returns
        self.num_state_features: int = 8 + self.state_history_length - 1
        # construct state space "box"
        # holdings_pct, time_pct, diff_pct, imbalance_all, imbalance_5, price_impact, spread, direction, returns
        self.state_highs: np.ndarray = np.array(
            [
                2,  # holdings_pct
                2,  # time_pct
                4,  # diff_pct
                1,  # imbalance_all
                1,  # imbalance_5
                np.finfo(np.float32).max,  # price_impact
                np.finfo(np.float32).max,  # spread
                np.finfo(np.float32).max,
            ]
            + (self.state_history_length - 1)  # directiom
            * [np.finfo(np.float32).max],  # returns
            dtype=np.float32,
        ).reshape(self.num_state_features, 1)

        self.state_lows: np.ndarray = np.array(
            [
                -2,  # holdings_pct
                -2,  # time_pct
                -4,  # diff_pct
                0,  # imbalance_all
                0,  # imbalance_5
                np.finfo(np.float32).min,  # price_impact
                np.finfo(np.float32).min,  # spread
                np.finfo(np.float32).min,
            ]
            + (self.state_history_length - 1)  # direction
            * [np.finfo(np.float32).min],  # returns
            dtype=np.float32,
        ).reshape(self.num_state_features, 1)

        self.observation_space: gym.Space = gym.spaces.Box(
            self.state_lows,
            self.state_highs,
            shape=(self.num_state_features, 1),
            dtype=np.float32,
        )
        # initialize previous_marked_to_market to starting_cash (No holding at the beginning of the episode)
        self.previous_marked_to_market: int = self.starting_cash

    def _map_action_space_to_ABIDES_SIMULATOR_SPACE(
        self, action: int
    ) -> List[Dict[str, Any]]:
        """
        utility function that maps open ai action definition (integers) to environnement API action definition (list of dictionaries)
        The action space ranges [0, 1, 2] where:
        - `0` MKT direction order_fixed_size
        - '1' LMT direction order_fixed_size
        - '2' DO NOTHING

        Arguments:
            - action: integer representation of the different actions

        Returns:
            - action_list: list of the corresponding series of action mapped into abides env apis
        """

        self.custom_metrics_tracker.action_counter[
            f"action_{action}"
        ] += 1  # increase counter
        if action == 0:
            return [
                {"type": "CCL_ALL"},
                {
                    "type": "MKT",
                    "direction": self.direction,
                    "size": self.order_fixed_size,
                },
            ]

        elif action == 1:
            return [
                {"type": "CCL_ALL"},
                {
                    "type": "LMT",
                    "direction": self.direction,
                    "size": self.order_fixed_size,
                    "limit_price": self.near_touch,
                },
            ]
        elif action == 2:
            return []
        else:
            raise ValueError(
                f"Action {action} is not part of the actions supported by the function."
            )

    @raw_state_to_state_pre_process
    def raw_state_to_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """
        method that transforms a raw state into a state representation

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - state: state representation defining the MDP for the execution v0 environnement
        """
        # 0) Preliminary
        bids = raw_state["parsed_mkt_data"]["bids"]
        asks = raw_state["parsed_mkt_data"]["asks"]
        last_transactions = raw_state["parsed_mkt_data"]["last_transaction"]

        # 1) Holdings
        holdings = raw_state["internal_data"]["holdings"]
        holdings_pct = holdings[-1] / self.parent_order_size

        # 2) Timing
        # 2)a) mkt_open
        mkt_open = raw_state["internal_data"]["mkt_open"][-1]
        # 2)b) time from beginning of execution (parent arrival)
        current_time = raw_state["internal_data"]["current_time"][-1]
        time_from_parent_arrival = current_time - mkt_open - self.first_interval
        assert (
            current_time >= mkt_open + self.first_interval
        ), "Agent has woken up earlier than its first interval"
        # 2)c) time limit
        time_limit = self.execution_window
        # 2)d) compute percentage time advancement
        time_pct = time_from_parent_arrival / time_limit

        # 3) Advancement Comparison
        diff_pct = holdings_pct - time_pct

        # 3) Imbalance
        imbalances_all = [
            markets_agent_utils.get_imbalance(b, a, depth=None)
            for (b, a) in zip(bids, asks)
        ]
        imbalance_all = imbalances_all[-1]

        imbalances_5 = [
            markets_agent_utils.get_imbalance(b, a, depth=5)
            for (b, a) in zip(bids, asks)
        ]
        imbalance_5 = imbalances_5[-1]

        # 4) price_impact
        mid_prices = [
            markets_agent_utils.get_mid_price(b, a, lt)
            for (b, a, lt) in zip(bids, asks, last_transactions)
        ]
        mid_price = mid_prices[-1]

        if self.step_index == 0:  # 0 order has been executed yet
            self.entry_price = mid_price

        entry_price = self.entry_price

        book = (
            raw_state["parsed_mkt_data"]["bids"][-1]
            if self.direction == "BUY"
            else raw_state["parsed_mkt_data"]["asks"][-1]
        )

        self.near_touch = book[0][0] if len(book) > 0 else last_transactions[-1]

        # Compute the price impact
        price_impact = (
            np.log(mid_price / entry_price)
            if self.direction == "BUY"
            else np.log(entry_price / mid_price)
        )

        # 5) Spread
        best_bids = [
            bids[0][0] if len(bids) > 0 else mid
            for (bids, mid) in zip(bids, mid_prices)
        ]
        best_asks = [
            asks[0][0] if len(asks) > 0 else mid
            for (asks, mid) in zip(asks, mid_prices)
        ]

        spreads = np.array(best_asks) - np.array(best_bids)
        spread = spreads[-1]

        # 6) direction feature
        direction_features = np.array(mid_prices) - np.array(last_transactions)
        direction_feature = direction_features[-1]

        # 7) mid_price
        mid_prices = [
            markets_agent_utils.get_mid_price(b, a, lt)
            for (b, a, lt) in zip(bids, asks, last_transactions)
        ]
        returns = np.diff(mid_prices)
        padded_returns = np.zeros(self.state_history_length - 1)
        padded_returns[-len(returns) :] = (
            returns if len(returns) > 0 else padded_returns
        )

        # log custom metrics to tracker
        self.custom_metrics_tracker.holdings_pct = holdings_pct
        self.custom_metrics_tracker.time_pct = time_pct
        self.custom_metrics_tracker.diff_pct = diff_pct
        self.custom_metrics_tracker.imbalance_all = imbalance_all
        self.custom_metrics_tracker.imbalance_5 = imbalance_5
        self.custom_metrics_tracker.price_impact = price_impact
        self.custom_metrics_tracker.spread = spread
        self.custom_metrics_tracker.direction_feature = direction_feature

        # 8) Computed State
        computed_state = np.array(
            [
                holdings_pct,
                time_pct,
                diff_pct,
                imbalance_all,
                imbalance_5,
                price_impact,
                spread,
                direction_feature,
            ]
            + padded_returns.tolist(),
            dtype=np.float32,
        )
        #
        self.step_index += 1
        return computed_state.reshape(self.num_state_features, 1)

    @raw_state_pre_process
    def raw_state_to_reward(self, raw_state: Dict[str, Any]) -> float:
        """
        method that transforms a raw state into the reward obtained during the step

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: immediate reward computed at each step  for the execution v0 environnement
        """
        # here we define the reward as cash + position marked to market normalized by parent_order_size

        # 1) entry_price
        entry_price = self.entry_price

        # 2) inter_wakeup_executed_orders
        inter_wakeup_executed_orders = raw_state["internal_data"][
            "inter_wakeup_executed_orders"
        ]

        # 3) Compute PNL of the orders
        if len(inter_wakeup_executed_orders) == 0:
            pnl = 0
        else:
            pnl = (
                sum(
                    (entry_price - order.fill_price) * order.quantity
                    for order in inter_wakeup_executed_orders
                )
                if self.direction == "BUY"
                else sum(
                    (order.fill_price - entry_price) * order.quantity
                    for order in inter_wakeup_executed_orders
                )
            )
        self.pnl = pnl

        # 4) normalization
        reward = pnl / self.parent_order_size
        # log custom metrics to tracker
        self.custom_metrics_tracker.slippage_reward = reward
        return reward

    @raw_state_pre_process
    def raw_state_to_update_reward(self, raw_state: Dict[str, Any]) -> float:
        """
        method that transforms a raw state into the final step reward update (if needed)

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: update reward computed at the end of the episode for the execution v0 environnement
        """
        # can update with additional reward at end of episode depending on scenario normalized by parent_order_size

        # 1) Holdings
        holdings = raw_state["internal_data"]["holdings"]

        # 2) parent_order_size
        parent_order_size = self.parent_order_size

        # 3) Compute update_reward
        if (self.direction == "BUY") and (holdings >= parent_order_size):
            update_reward = (
                abs(holdings - parent_order_size) * self.too_much_reward_update
            )  # executed buy too much

        elif (self.direction == "BUY") and (holdings < parent_order_size):
            update_reward = (
                abs(holdings - parent_order_size) * self.not_enough_reward_update
            )  # executed buy not enough

        elif (self.direction == "SELL") and (holdings <= -parent_order_size):
            update_reward = (
                abs(holdings - parent_order_size) * self.too_much_reward_update
            )  # executed sell too much
        elif (self.direction == "SELL") and (holdings > -parent_order_size):
            update_reward = (
                abs(holdings - parent_order_size) * self.not_enough_reward_update
            )  # executed sell not enough
        else:
            update_reward = self.just_quantity_reward_update

        # 4) Normalization
        update_reward = update_reward / self.parent_order_size

        self.custom_metrics_tracker.late_penalty_reward = update_reward
        return update_reward

    @raw_state_pre_process
    def raw_state_to_done(self, raw_state: Dict[str, Any]) -> bool:
        """
        method that transforms a raw state into the flag if an episode is done

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - done: flag that describes if the episode is terminated or not  for the execution v0 environnement
        """
        # episode can stop because market closes or because some condition is met
        # here the condition is parent order fully executed

        # 1) Holdings
        holdings = raw_state["internal_data"]["holdings"]

        # 2) parent_order_size
        parent_order_size = self.parent_order_size

        # 3) current time
        current_time = raw_state["internal_data"]["current_time"]

        # 4) time_limit
        # 4)a) mkt_open
        mkt_open = raw_state["internal_data"]["mkt_open"]
        # 4)b time_limit
        time_limit = mkt_open + self.first_interval + self.execution_window

        # 5) conditions
        if (self.direction == "BUY") and (holdings >= parent_order_size):
            done = True  # Buy parent order executed
        elif (self.direction == "SELL") and (holdings <= -parent_order_size):
            done = True  # Sell parent order executed
        elif current_time >= time_limit:
            done = True  # Mkt Close
        else:
            done = False

        self.custom_metrics_tracker.executed_quantity = (
            holdings if self.direction == "BUY" else -holdings
        )
        self.custom_metrics_tracker.remaining_quantity = (
            parent_order_size - self.custom_metrics_tracker.executed_quantity
        )

        return done

    @raw_state_pre_process
    def raw_state_to_info(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        method that transforms a raw state into an info dictionnary

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: info dictionnary computed at each step for the execution v0 environnement
        """
        # Agent cannot use this info for taking decision
        # only for debugging

        # 1) Last Known Market Transaction Price
        last_transaction = raw_state["parsed_mkt_data"]["last_transaction"]

        # 2) Last Known best bid
        bids = raw_state["parsed_mkt_data"]["bids"]
        best_bid = bids[0][0] if len(bids) > 0 else last_transaction

        # 3) Last Known best ask
        asks = raw_state["parsed_mkt_data"]["asks"]
        best_ask = asks[0][0] if len(asks) > 0 else last_transaction

        # 4) Current Time
        current_time = raw_state["internal_data"]["current_time"]

        # 5) Holdings
        holdings = raw_state["internal_data"]["holdings"]

        if self.debug_mode == True:
            return {
                "last_transaction": last_transaction,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "current_time": current_time,
                "holdings": holdings,
                "parent_size": self.parent_order_size,
                "pnl": self.pnl,
                "reward": self.pnl / self.parent_order_size,
            }
        else:
            return asdict(self.custom_metrics_tracker)
