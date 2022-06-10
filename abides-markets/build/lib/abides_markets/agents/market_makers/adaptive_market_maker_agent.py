import logging
from math import floor, ceil
from typing import Dict, List, Optional, Tuple

import numpy as np

from abides_core import Message, NanosecondTime

from ...utils import sigmoid
from ...messages.marketdata import (
    MarketDataMsg,
    L2SubReqMsg,
    BookImbalanceDataMsg,
    BookImbalanceSubReqMsg,
    MarketDataEventMsg,
)
from ...messages.query import QuerySpreadResponseMsg, QueryTransactedVolResponseMsg
from ...orders import Side
from ..trading_agent import TradingAgent


ANCHOR_TOP_STR = "top"
ANCHOR_BOTTOM_STR = "bottom"
ANCHOR_MIDDLE_STR = "middle"

ADAPTIVE_SPREAD_STR = "adaptive"
INITIAL_SPREAD_VALUE = 50


logger = logging.getLogger(__name__)


class AdaptiveMarketMakerAgent(TradingAgent):
    """This class implements a modification of the Chakraborty-Kearns `ladder` market-making strategy, wherein the
    the size of order placed at each level is set as a fraction of measured transacted volume in the previous time
    period.

    Can skew orders to size of current inventory using beta parameter, whence beta == 0 represents inventory being
    ignored and beta == infinity represents all liquidity placed on one side of book.

    ADAPTIVE SPREAD: the market maker's spread can be set either as a fixed or value or can be adaptive,
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        pov: float = 0.05,
        min_order_size: int = 20,
        window_size: float = 5,
        anchor: str = ANCHOR_MIDDLE_STR,
        num_ticks: int = 20,
        level_spacing: float = 0.5,
        wake_up_freq: NanosecondTime = 1_000_000_000,  # 1 second
        poisson_arrival: bool = True,
        subscribe: bool = False,
        subscribe_freq: float = 10e9,
        subscribe_num_levels: int = 1,
        cancel_limit_delay: int = 50,
        skew_beta=0,
        price_skew_param=None,
        spread_alpha: float = 0.85,
        backstop_quantity: int = 0,
        log_orders: bool = False,
        min_imbalance=0.9,
    ) -> None:

        super().__init__(id, name, type, random_state, starting_cash, log_orders)
        self.is_adaptive: bool = False
        self.symbol: str = symbol  # Symbol traded
        self.pov: float = (
            pov  # fraction of transacted volume placed at each price level
        )
        self.min_order_size: int = (
            min_order_size  # minimum size order to place at each level, if pov <= min
        )
        self.anchor: str = self.validate_anchor(
            anchor
        )  # anchor either top of window or bottom of window to mid-price
        self.window_size: float = self.validate_window_size(
            window_size
        )  # Size in ticks (cents) of how wide the window around mid price is. If equal to
        # string 'adaptive' then ladder starts at best bid and ask
        self.num_ticks: int = num_ticks  # number of ticks on each side of window in which to place liquidity
        self.level_spacing: float = (
            level_spacing  #  level spacing as a fraction of the spread
        )
        self.wake_up_freq: str = wake_up_freq  # Frequency of agent wake up
        self.poisson_arrival: bool = (
            poisson_arrival  # Whether to arrive as a Poisson process
        )
        if self.poisson_arrival:
            self.arrival_rate = self.wake_up_freq

        self.subscribe: bool = subscribe  # Flag to determine whether to subscribe to data or use polling mechanism
        self.subscribe_freq: float = subscribe_freq  # Frequency in nanoseconds^-1 at which to receive market updates
        # in subscribe mode
        self.min_imbalance = min_imbalance
        self.subscribe_num_levels: int = (
            subscribe_num_levels  # Number of orderbook levels in subscription mode
        )
        self.cancel_limit_delay: int = cancel_limit_delay  # delay in nanoseconds between order cancellations and new limit order placements

        self.skew_beta = (
            skew_beta  # parameter for determining order placement imbalance
        )
        self.price_skew_param = (
            price_skew_param  # parameter determining how much to skew price level.
        )
        self.spread_alpha: float = spread_alpha  # parameter for exponentially weighted moving average of spread. 1 corresponds to ignoring old values, 0 corresponds to no updates
        self.backstop_quantity: int = backstop_quantity  # how many orders to place at outside order level, to prevent liquidity dropouts. If None then place same as at other levels.
        self.log_orders: float = log_orders

        self.has_subscribed = False

        ## Internal variables

        self.subscription_requested: bool = False
        self.state: Dict[str, bool] = self.initialise_state()
        self.buy_order_size: int = self.min_order_size
        self.sell_order_size: int = self.min_order_size

        self.last_mid: Optional[int] = None  # last observed mid price
        self.last_spread: float = (
            INITIAL_SPREAD_VALUE  # last observed spread moving average
        )
        self.tick_size: Optional[int] = (
            None if self.is_adaptive else ceil(self.last_spread * self.level_spacing)
        )
        self.LIQUIDITY_DROPOUT_WARNING: str = (
            f"Liquidity dropout for agent {self.name}."
        )

        self.two_side: bool = (
            False if self.price_skew_param is None else True
        )  # switch to control self.get_transacted_volume
        # method

    def initialise_state(self) -> Dict[str, bool]:
        """Returns variables that keep track of whether spread and transacted volume have been observed."""

        if self.subscribe:
            return {"AWAITING_MARKET_DATA": True, "AWAITING_TRANSACTED_VOLUME": True}
        else:
            return {"AWAITING_SPREAD": True, "AWAITING_TRANSACTED_VOLUME": True}

    def validate_anchor(self, anchor: str) -> str:
        """Checks that input parameter anchor takes allowed value, raises ``ValueError`` if not.

        Arguments:
            anchor:

        Returns:
            The anchor if validated.
        """

        if anchor not in [ANCHOR_TOP_STR, ANCHOR_BOTTOM_STR, ANCHOR_MIDDLE_STR]:
            raise ValueError(
                f"Variable anchor must take the value `{ANCHOR_BOTTOM_STR}`, `{ANCHOR_MIDDLE_STR}` or "
                f"`{ANCHOR_TOP_STR}`"
            )
        else:
            return anchor

    def validate_window_size(self, window_size: float) -> Optional[int]:
        """Checks that input parameter window_size takes allowed value, raises ``ValueError`` if not.

        Arguments:
            window_size:

        Returns:
            The window_size if validated
        """

        try:  # fixed window size specified
            return int(window_size)
        except:
            if window_size.lower() == "adaptive":
                self.is_adaptive = True
                self.anchor = ANCHOR_MIDDLE_STR
                return None
            else:
                raise ValueError(
                    f"Variable window_size must be of type int or string {ADAPTIVE_SPREAD_STR}."
                )

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)

    def wakeup(self, current_time: NanosecondTime):
        """Agent wakeup is determined by self.wake_up_freq."""

        can_trade = super().wakeup(current_time)

        if not self.has_subscribed:
            super().request_data_subscription(
                BookImbalanceSubReqMsg(
                    symbol=self.symbol,
                    min_imbalance=self.min_imbalance,
                )
            )
            self.last_time_book_order = current_time
            self.has_subscribed = True

        if self.subscribe and not self.subscription_requested:
            super().request_data_subscription(
                L2SubReqMsg(
                    symbol=self.symbol,
                    freq=self.subscribe_freq,
                    depth=self.subscribe_num_levels,
                )
            )
            self.subscription_requested = True
            self.get_transacted_volume(self.symbol, lookback_period=self.subscribe_freq)
            self.state = self.initialise_state()

        elif can_trade and not self.subscribe:
            self.cancel_all_orders()
            self.delay(self.cancel_limit_delay)
            self.get_current_spread(self.symbol, depth=self.subscribe_num_levels)
            self.get_transacted_volume(self.symbol, lookback_period=self.wake_up_freq)
            self.initialise_state()

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        """Processes message from exchange.

        Main function is to update orders in orderbook relative to mid-price.

        Arguments:
            current_time: Simulation current time.
            message: Message received by self from ExchangeAgent.
        """

        super().receive_message(current_time, sender_id, message)

        mid = None
        if self.last_mid is not None:
            mid = self.last_mid

        if self.last_spread is not None and self.is_adaptive:
            self._adaptive_update_window_and_tick_size()

        if (
            isinstance(message, QueryTransactedVolResponseMsg)
            and self.state["AWAITING_TRANSACTED_VOLUME"] is True
        ):
            self.update_order_size()
            self.state["AWAITING_TRANSACTED_VOLUME"] = False

        if isinstance(message, BookImbalanceDataMsg):
            if message.stage == MarketDataEventMsg.Stage.START:
                try:
                    self.place_orders(mid)
                    self.last_time_book_order = current_time
                except:
                    pass

        if not self.subscribe:
            if (
                isinstance(message, QuerySpreadResponseMsg)
                and self.state["AWAITING_SPREAD"] is True
            ):
                bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
                if bid and ask:
                    mid = int((ask + bid) / 2)
                    self.last_mid = mid
                    if self.is_adaptive:
                        spread = int(ask - bid)
                        self._adaptive_update_spread(spread)

                    self.state["AWAITING_SPREAD"] = False
                else:
                    logger.debug("SPREAD MISSING at time {}", current_time)
                    self.state[
                        "AWAITING_SPREAD"
                    ] = False  # use last mid price and spread

            if (
                self.state["AWAITING_SPREAD"] is False
                and self.state["AWAITING_TRANSACTED_VOLUME"] is False
                and mid is not None
            ):
                self.place_orders(mid)
                self.state = self.initialise_state()
                self.set_wakeup(current_time + self.get_wake_frequency())

        else:  # subscription mode
            if (
                isinstance(message, MarketDataMsg)
                and self.state["AWAITING_MARKET_DATA"] is True
            ):
                bid = (
                    self.known_bids[self.symbol][0][0]
                    if self.known_bids[self.symbol]
                    else None
                )
                ask = (
                    self.known_asks[self.symbol][0][0]
                    if self.known_asks[self.symbol]
                    else None
                )
                if bid and ask:
                    mid = int((ask + bid) / 2)
                    self.last_mid = mid
                    if self.is_adaptive:
                        spread = int(ask - bid)
                        self._adaptive_update_spread(spread)

                    self.state["AWAITING_MARKET_DATA"] = False
                else:
                    logger.debug("SPREAD MISSING at time {}", current_time)
                    self.state["AWAITING_MARKET_DATA"] = False

            if (
                self.state["MARKET_DATA"] is False
                and self.state["AWAITING_TRANSACTED_VOLUME"] is False
            ):
                self.place_orders(mid)
                self.state = self.initialise_state()

    def _adaptive_update_spread(self, spread) -> None:
        """Update internal spread estimate with exponentially weighted moving average.

        Arguments:
            spread
        """

        spread_ewma = (
            self.spread_alpha * spread + (1 - self.spread_alpha) * self.last_spread
        )
        self.window_size = spread_ewma
        self.last_spread = spread_ewma

    def _adaptive_update_window_and_tick_size(self) -> None:
        """Update window size and tick size relative to internal spread estimate."""

        self.window_size = self.last_spread
        self.tick_size = round(self.level_spacing * self.window_size)
        if self.tick_size == 0:
            self.tick_size = 1

    def update_order_size(self) -> None:
        """Updates size of order to be placed."""

        buy_transacted_volume = self.transacted_volume[self.symbol][0]
        sell_transacted_volume = self.transacted_volume[self.symbol][1]
        total_transacted_volume = buy_transacted_volume + sell_transacted_volume

        qty = round(self.pov * total_transacted_volume)

        if self.skew_beta == 0:  # ignore inventory
            self.buy_order_size = (
                qty if qty >= self.min_order_size else self.min_order_size
            )
            self.sell_order_size = (
                qty if qty >= self.min_order_size else self.min_order_size
            )
        else:
            holdings = self.get_holdings(self.symbol)
            proportion_sell = sigmoid(holdings, self.skew_beta)
            sell_size = ceil(proportion_sell * qty)
            buy_size = floor((1 - proportion_sell) * qty)

            self.buy_order_size = (
                buy_size if buy_size >= self.min_order_size else self.min_order_size
            )
            self.sell_order_size = (
                sell_size if sell_size >= self.min_order_size else self.min_order_size
            )

    def compute_orders_to_place(self, mid: int) -> Tuple[List[int], List[int]]:
        """Given a mid price, computes the orders that need to be removed from
        orderbook, and adds these orders to bid and ask deques.

        Arguments:
            mid: Mid price.
        """

        if self.price_skew_param is None:
            mid_point = mid
        else:
            buy_transacted_volume = self.transacted_volume[self.symbol][0]
            sell_transacted_volume = self.transacted_volume[self.symbol][1]

            if (buy_transacted_volume == 0) and (sell_transacted_volume == 0):
                mid_point = mid
            else:
                # trade imbalance, +1 means all transactions are buy, -1 means all transactions are sell
                trade_imbalance = (
                    2
                    * buy_transacted_volume
                    / (buy_transacted_volume + sell_transacted_volume)
                ) - 1
                mid_point = int(mid + (trade_imbalance * self.price_skew_param))

        if self.anchor == ANCHOR_MIDDLE_STR:
            highest_bid = int(mid_point) - floor(0.5 * self.window_size)
            lowest_ask = int(mid_point) + ceil(0.5 * self.window_size)
        elif self.anchor == ANCHOR_BOTTOM_STR:
            highest_bid = int(mid_point - 1)
            lowest_ask = int(mid_point + self.window_size)
        elif self.anchor == ANCHOR_TOP_STR:
            highest_bid = int(mid_point - self.window_size)
            lowest_ask = int(mid_point + 1)

        lowest_bid = highest_bid - ((self.num_ticks - 1) * self.tick_size)
        highest_ask = lowest_ask + ((self.num_ticks - 1) * self.tick_size)

        bids_to_place = [
            price
            for price in range(lowest_bid, highest_bid + self.tick_size, self.tick_size)
        ]
        asks_to_place = [
            price
            for price in range(lowest_ask, highest_ask + self.tick_size, self.tick_size)
        ]

        return bids_to_place, asks_to_place

    def place_orders(self, mid: int) -> None:
        """Given a mid-price, compute new orders that need to be placed, then
        send the orders to the Exchange.

        Arguments:
            mid: Mid price.
        """

        bid_orders, ask_orders = self.compute_orders_to_place(mid)

        orders = []

        if self.backstop_quantity != 0:
            bid_price = bid_orders[0]
            logger.debug(
                "{}: Placing BUY limit order of size {} @ price {}",
                self.name,
                self.backstop_quantity,
                bid_price,
            )
            orders.append(
                self.create_limit_order(
                    self.symbol, self.backstop_quantity, Side.BID, bid_price
                )
            )
            bid_orders = bid_orders[1:]

            ask_price = ask_orders[-1]
            logger.debug(
                "{}: Placing SELL limit order of size {} @ price {}",
                self.name,
                self.backstop_quantity,
                ask_price,
            )
            orders.append(
                self.create_limit_order(
                    self.symbol, self.backstop_quantity, Side.ASK, ask_price
                )
            )
            ask_orders = ask_orders[:-1]

        for bid_price in bid_orders:
            logger.debug(
                "{}: Placing BUY limit order of size {} @ price {}",
                self.name,
                self.buy_order_size,
                bid_price,
            )
            orders.append(
                self.create_limit_order(
                    self.symbol, self.buy_order_size, Side.BID, bid_price
                )
            )

        for ask_price in ask_orders:
            logger.debug(
                "{}: Placing SELL limit order of size {} @ price {}",
                self.name,
                self.sell_order_size,
                ask_price,
            )
            orders.append(
                self.create_limit_order(
                    self.symbol, self.sell_order_size, Side.ASK, ask_price
                )
            )

        self.place_multiple_orders(orders)

    def get_wake_frequency(self) -> NanosecondTime:
        if not self.poisson_arrival:
            return self.wake_up_freq
        else:
            delta_time = self.random_state.exponential(scale=self.arrival_rate)
            return int(round(delta_time))
