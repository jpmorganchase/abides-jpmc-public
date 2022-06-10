import logging
import sys
import warnings
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import fmt_ts

from ..messages.market import (
    MarketClosePriceRequestMsg,
    MarketClosePriceMsg,
    MarketClosedMsg,
    MarketHoursRequestMsg,
    MarketHoursMsg,
)
from ..messages.marketdata import MarketDataSubReqMsg, MarketDataMsg, L2DataMsg
from ..messages.order import (
    LimitOrderMsg,
    MarketOrderMsg,
    PartialCancelOrderMsg,
    CancelOrderMsg,
    ModifyOrderMsg,
    ReplaceOrderMsg,
)
from ..messages.orderbook import (
    OrderAcceptedMsg,
    OrderExecutedMsg,
    OrderCancelledMsg,
    OrderPartialCancelledMsg,
    OrderModifiedMsg,
    OrderReplacedMsg,
)
from ..messages.query import (
    QueryLastTradeMsg,
    QueryLastTradeResponseMsg,
    QuerySpreadMsg,
    QuerySpreadResponseMsg,
    QueryOrderStreamMsg,
    QueryOrderStreamResponseMsg,
    QueryTransactedVolMsg,
    QueryTransactedVolResponseMsg,
)
from ..orders import Order, LimitOrder, MarketOrder, Side
from .financial_agent import FinancialAgent
from .exchange_agent import ExchangeAgent


logger = logging.getLogger(__name__)


class TradingAgent(FinancialAgent):
    """
    The TradingAgent class (via FinancialAgent, via Agent) is intended as the
    base class for all trading agents (i.e. not things like exchanges) in a
    market simulation.

    It handles a lot of messaging (inbound and outbound) and state maintenance
    automatically, so subclasses can focus just on implementing a strategy without
    too much bookkeeping.
    """

    def __init__(
        self,
        id: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        starting_cash: int = 100000,
        log_orders: bool = False,
    ) -> None:
        # Base class init.
        super().__init__(id, name, type, random_state)

        # We don't yet know when the exchange opens or closes.
        self.mkt_open: Optional[NanosecondTime] = None
        self.mkt_close: Optional[NanosecondTime] = None

        # Log order activity?
        self.log_orders: bool = log_orders

        # Log all activity to file?
        if log_orders is None:
            self.log_orders = False
            self.log_to_file = False

        # Store starting_cash in case we want to refer to it for performance stats.
        # It should NOT be modified.  Use the 'CASH' key in self.holdings.
        # 'CASH' is always in cents!  Note that agents are limited by their starting
        # cash, currently without leverage.  Taking short positions is permitted,
        # but does NOT increase the amount of at-risk capital allowed.
        self.starting_cash: int = starting_cash

        # TradingAgent has constants to support simulated market orders.
        self.MKT_BUY = sys.maxsize
        self.MKT_SELL = 0

        # The base TradingAgent will track its holdings and outstanding orders.
        # Holdings is a dictionary of symbol -> shares.  CASH is a special symbol
        # worth one cent per share.  Orders is a dictionary of active, open orders
        # (not cancelled, not fully executed) keyed by order_id.
        self.holdings: Dict[str, int] = {"CASH": starting_cash}
        self.orders: Dict[int, Order] = {}

        # The base TradingAgent also tracks last known prices for every symbol
        # for which it has received as QUERY_LAST_TRADE message.  Subclass
        # agents may use or ignore this as they wish.  Note that the subclass
        # agent must request pricing when it wants it.  This agent does NOT
        # automatically generate such requests, though it has a helper function
        # that can be used to make it happen.
        self.last_trade: Dict[str, int] = {}

        # used in subscription mode to record the timestamp for which the data was current in the ExchangeAgent
        self.exchange_ts: Dict[str, NanosecondTime] = {}

        # When a last trade price comes in after market close, the trading agent
        # automatically records it as the daily close price for a symbol.
        self.daily_close_price: Dict[str, int] = {}

        self.nav_diff: int = 0
        self.basket_size: int = 0

        # The agent remembers the last known bids and asks (with variable depth,
        # showing only aggregate volume at each price level) when it receives
        # a response to QUERY_SPREAD.
        self.known_bids: Dict = {}
        self.known_asks: Dict = {}

        # The agent remembers the order history communicated by the exchange
        # when such is requested by an agent (for example, a heuristic belief
        # learning agent).
        self.stream_history: Dict[str, Any] = {}

        # The agent records the total transacted volume in the exchange for a given symbol and lookback period
        self.transacted_volume: Dict = {}

        # Each agent can choose to log the orders executed
        self.executed_orders: List = []

        # For special logging at the first moment the simulator kernel begins
        # running (which is well after agent init), it is useful to keep a simple
        # boolean flag.
        self.first_wake: bool = True

        # Remember whether we have already passed the exchange close time, as far
        # as we know.
        self.mkt_closed: bool = False

    # Simulation lifecycle messages.

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        """
        Arguments:
            start_time: The time that the simulation started.
        """

        assert self.kernel is not None

        # self.kernel is set in Agent.kernel_initializing()
        self.logEvent("STARTING_CASH", self.starting_cash, True)

        # Find an exchange with which we can place orders.  It is guaranteed
        # to exist by now (if there is one).
        self.exchange_id: int = self.kernel.find_agents_by_type(ExchangeAgent)[0]

        logger.debug(
            f"Agent {self.id} requested agent of type Agent.ExchangeAgent.  Given Agent ID: {self.exchange_id}"
        )

        # Request a wake-up call as in the base Agent.
        super().kernel_starting(start_time)

    def kernel_stopping(self) -> None:
        # Always call parent method to be safe.
        super().kernel_stopping()

        assert self.kernel is not None

        # Print end of day holdings.
        self.logEvent(
            "FINAL_HOLDINGS", self.fmt_holdings(self.holdings), deepcopy_event=False
        )
        self.logEvent("FINAL_CASH_POSITION", self.holdings["CASH"], True)

        # Mark to market.
        cash = self.mark_to_market(self.holdings)

        self.logEvent("ENDING_CASH", cash, True)
        logger.debug(
            "Final holdings for {}: {}. Marked to market: {}".format(
                self.name, self.fmt_holdings(self.holdings), cash
            )
        )

        # Record final results for presentation/debugging.  This is an ugly way
        # to do this, but it is useful for now.
        mytype = self.type
        gain = cash - self.starting_cash

        if mytype in self.kernel.mean_result_by_agent_type:
            self.kernel.mean_result_by_agent_type[mytype] += gain
            self.kernel.agent_count_by_type[mytype] += 1
        else:
            self.kernel.mean_result_by_agent_type[mytype] = gain
            self.kernel.agent_count_by_type[mytype] = 1

    # Simulation participation messages.

    def wakeup(self, current_time: NanosecondTime) -> bool:
        """
        Arguments:
            current_time: The time that this agent was woken up by the kernel.

        Returns:
            For the sake of subclasses, TradingAgent now returns a boolean
            indicating whether the agent is "ready to trade" -- has it received
            the market open and closed times, and is the market not already closed.
        """

        super().wakeup(current_time)

        if self.first_wake:
            # Log initial holdings.
            self.logEvent("HOLDINGS_UPDATED", self.holdings)
            self.first_wake = False

            # Tell the exchange we want to be sent the final prices when the market closes.
            self.send_message(self.exchange_id, MarketClosePriceRequestMsg())

        if self.mkt_open is None:
            # Ask our exchange when it opens and closes.
            self.send_message(self.exchange_id, MarketHoursRequestMsg())

        return (self.mkt_open and self.mkt_close) and not self.mkt_closed

    def request_data_subscription(
        self, subscription_message: MarketDataSubReqMsg
    ) -> None:
        """
        Used by any Trading Agent subclass to create a subscription to market data from
        the Exchange Agent.

        Arguments:
            subscription_message: An instance of a MarketDataSubReqMessage.
        """

        subscription_message.cancel = False

        self.send_message(recipient_id=self.exchange_id, message=subscription_message)

    def cancel_data_subscription(
        self, subscription_message: MarketDataSubReqMsg
    ) -> None:
        """
        Used by any Trading Agent subclass to cancel subscription to market data from
        the Exchange Agent.

        Arguments:
            subscription_message: An instance of a MarketDataSubReqMessage.
        """

        subscription_message.cancel = True

        self.send_message(recipient_id=self.exchange_id, message=subscription_message)

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        """
        Arguments:
            current_time: The time that this agent received the message.
            sender_id: The ID of the agent who sent the message.
            message: The message contents.
        """

        assert self.kernel is not None

        super().receive_message(current_time, sender_id, message)

        # Do we know the market hours?
        had_mkt_hours = self.mkt_open is not None and self.mkt_close is not None

        # Record market open or close times.
        if isinstance(message, MarketHoursMsg):
            self.mkt_open = message.mkt_open
            self.mkt_close = message.mkt_close

            logger.debug("Recorded market open: {}".format(fmt_ts(self.mkt_open)))
            logger.debug("Recorded market close: {}".format(fmt_ts(self.mkt_close)))

        elif isinstance(message, MarketClosePriceMsg):
            # Update our local last trade prices with the accurate last trade prices from
            # the exchange so we can accurately calculate our mark-to-market values.
            for symbol, close_price in message.close_prices.items():
                self.last_trade[symbol] = close_price

        elif isinstance(message, MarketClosedMsg):
            # We've tried to ask the exchange for something after it closed.  Remember this
            # so we stop asking for things that can't happen.
            self.market_closed()

        elif isinstance(message, OrderExecutedMsg):
            # Call the order_executed method, which subclasses should extend.  This parent
            # class could implement default "portfolio tracking" or "returns tracking"
            # behavior.
            self.order_executed(message.order)

        elif isinstance(message, OrderAcceptedMsg):
            # Call the order_accepted method, which subclasses should extend.
            self.order_accepted(message.order)

        elif isinstance(message, OrderCancelledMsg):
            # Call the order_cancelled method, which subclasses should extend.
            self.order_cancelled(message.order)

        elif isinstance(message, OrderPartialCancelledMsg):
            # Call the order_cancelled method, which subclasses should extend.
            self.order_partial_cancelled(message.new_order)

        elif isinstance(message, OrderModifiedMsg):
            # Call the order_cancelled method, which subclasses should extend.
            self.order_modified(message.new_order)

        elif isinstance(message, OrderReplacedMsg):
            # Call the order_cancelled method, which subclasses should extend.
            self.order_replaced(message.old_order, message.new_order)

        elif isinstance(message, QueryLastTradeResponseMsg):
            # Call the query_last_trade method, which subclasses may extend.
            # Also note if the market is closed.
            if message.mkt_closed:
                self.mkt_closed = True

            self.query_last_trade(message.symbol, message.last_trade)

        elif isinstance(message, QuerySpreadResponseMsg):
            # Call the query_spread method, which subclasses may extend.
            # Also note if the market is closed.
            if message.mkt_closed:
                self.mkt_closed = True

            self.query_spread(
                message.symbol, message.last_trade, message.bids, message.asks, ""
            )

        elif isinstance(message, QueryOrderStreamResponseMsg):
            # Call the query_order_stream method, which subclasses may extend.
            # Also note if the market is closed.
            if message.mkt_closed:
                self.mkt_closed = True

            self.query_order_stream(message.symbol, message.orders)

        elif isinstance(message, QueryTransactedVolResponseMsg):
            if message.mkt_closed:
                self.mkt_closed = True

            self.query_transacted_volume(
                message.symbol, message.bid_volume, message.ask_volume
            )

        elif isinstance(message, MarketDataMsg):
            self.handle_market_data(message)

        # Now do we know the market hours?
        have_mkt_hours = self.mkt_open is not None and self.mkt_close is not None

        # Once we know the market open and close times, schedule a wakeup call for market open.
        # Only do this once, when we first have both items.
        if have_mkt_hours and not had_mkt_hours:
            # Agents are asked to generate a wake offset from the market open time.  We structure
            # this as a subclass request so each agent can supply an appropriate offset relative
            # to its trading frequency.
            ns_offset = self.get_wake_frequency()

            self.set_wakeup(self.mkt_open + ns_offset)

    def get_last_trade(self, symbol: str) -> None:
        """
        Used by any Trading Agent subclass to query the last trade price for a symbol.

        This activity is not logged.

        Arguments:
            symbol: The symbol to query.
        """

        self.send_message(self.exchange_id, QueryLastTradeMsg(symbol))

    def get_current_spread(self, symbol: str, depth: int = 1) -> None:
        """
        Used by any Trading Agent subclass to query the current spread for a symbol.

        This activity is not logged.

        Arguments:
            symbol: The symbol to query.
            depth:
        """

        self.send_message(self.exchange_id, QuerySpreadMsg(symbol, depth))

    def get_order_stream(self, symbol: str, length: int = 1) -> None:
        """
        Used by any Trading Agent subclass to query the recent order s  tream for a symbol.

        Arguments:
            symbol: The symbol to query.
            length:
        """

        self.send_message(self.exchange_id, QueryOrderStreamMsg(symbol, length))

    def get_transacted_volume(
        self, symbol: str, lookback_period: str = "10min"
    ) -> None:
        """
        Used by any trading agent subclass to query the total transacted volume in a
        given lookback period.

        Arguments:
            symbol: The symbol to query.
            lookback_period: The length of time to consider when calculating the volume.
        """

        self.send_message(
            self.exchange_id, QueryTransactedVolMsg(symbol, lookback_period)
        )

    def create_limit_order(
        self,
        symbol: str,
        quantity: int,
        side: Side,
        limit_price: int,
        order_id: Optional[int] = None,
        is_hidden: bool = False,
        is_price_to_comply: bool = False,
        insert_by_id: bool = False,
        is_post_only: bool = False,
        ignore_risk: bool = True,
        tag: Any = None,
    ) -> LimitOrder:
        """
        Used by any Trading Agent subclass to create a limit order.

        Arguments:
            symbol: A valid symbol.
            quantity: Positive share quantity.
            side: Side.BID or Side.ASK.
            limit_price: Price in cents.
            order_id: An optional order id (otherwise global autoincrement is used).
            is_hidden:
            is_price_to_comply:
            insert_by_id:
            is_post_only:
            ignore_risk: Whether cash or risk limits should be enforced or ignored for
                the order.
            tag:
        """

        order = LimitOrder(
            agent_id=self.id,
            time_placed=self.current_time,
            symbol=symbol,
            quantity=quantity,
            side=side,
            limit_price=limit_price,
            is_hidden=is_hidden,
            is_price_to_comply=is_price_to_comply,
            insert_by_id=insert_by_id,
            is_post_only=is_post_only,
            order_id=order_id,
            tag=tag,
        )

        if quantity > 0:
            # Test if this order can be permitted given our at-risk limits.
            new_holdings = self.holdings.copy()

            q = order.quantity if order.side.is_bid() else -order.quantity

            if order.symbol in new_holdings:
                new_holdings[order.symbol] += q
            else:
                new_holdings[order.symbol] = q

            # If at_risk is lower, always allow.  Otherwise, new_at_risk must be below starting cash.
            if not ignore_risk:
                # Compute before and after at-risk capital.
                at_risk = self.mark_to_market(self.holdings) - self.holdings["CASH"]
                new_at_risk = self.mark_to_market(new_holdings) - new_holdings["CASH"]

                if (new_at_risk > at_risk) and (new_at_risk > self.starting_cash):
                    logger.debug(
                        "TradingAgent ignored limit order due to at-risk constraints: {}\n{}".format(
                            order, self.fmt_holdings(self.holdings)
                        )
                    )
                    return

            return order

        else:
            warnings.warn(f"TradingAgent ignored limit order of quantity zero: {order}")

    def place_limit_order(
        self,
        symbol: str,
        quantity: int,
        side: Side,
        limit_price: int,
        order_id: Optional[int] = None,
        is_hidden: bool = False,
        is_price_to_comply: bool = False,
        insert_by_id: bool = False,
        is_post_only: bool = False,
        ignore_risk: bool = True,
        tag: Any = None,
    ) -> None:
        """
        Used by any Trading Agent subclass to place a limit order.

        Arguments:
            symbol: A valid symbol.
            quantity: Positive share quantity.
            side: Side.BID or Side.ASK.
            limit_price: Price in cents.
            order_id: An optional order id (otherwise global autoincrement is used).
            is_hidden:
            is_price_to_comply:
            insert_by_id:
            is_post_only:
            ignore_risk: Whether cash or risk limits should be enforced or ignored for
                the order.
            tag:
        """

        order = self.create_limit_order(
            symbol,
            quantity,
            side,
            limit_price,
            order_id,
            is_hidden,
            is_price_to_comply,
            insert_by_id,
            is_post_only,
            ignore_risk,
            tag,
        )

        if order is not None:
            self.orders[order.order_id] = deepcopy(order)
            self.send_message(self.exchange_id, LimitOrderMsg(order))

            if self.log_orders:
                self.logEvent("ORDER_SUBMITTED", order.to_dict(), deepcopy_event=False)

    def place_market_order(
        self,
        symbol: str,
        quantity: int,
        side: Side,
        order_id: Optional[int] = None,
        ignore_risk: bool = True,
        tag: Any = None,
    ) -> None:
        """
        Used by any Trading Agent subclass to place a market order.

        The market order is created as multiple limit orders crossing the spread
        walking the book until all the quantities are matched.

        Arguments:
            symbol: Name of the stock traded.
            quantity: Order quantity.
            side: Side.BID or Side.ASK.
            order_id: Order ID for market replay.
            ignore_risk: Whether cash or risk limits should be enforced or ignored for
                the order.
            tag:
        """

        order = MarketOrder(
            self.id, self.current_time, symbol, quantity, side, order_id, tag
        )
        if quantity > 0:
            # compute new holdings
            new_holdings = self.holdings.copy()
            q = order.quantity if order.side.is_bid() else -order.quantity
            if order.symbol in new_holdings:
                new_holdings[order.symbol] += q
            else:
                new_holdings[order.symbol] = q

            if not ignore_risk:
                # Compute before and after at-risk capital.
                at_risk = self.mark_to_market(self.holdings) - self.holdings["CASH"]
                new_at_risk = self.mark_to_market(new_holdings) - new_holdings["CASH"]

                if (new_at_risk > at_risk) and (new_at_risk > self.starting_cash):
                    logger.debug(
                        "TradingAgent ignored market order due to at-risk constraints: {}\n{}".format(
                            order, self.fmt_holdings(self.holdings)
                        )
                    )
                    return
            self.orders[order.order_id] = deepcopy(order)
            self.send_message(self.exchange_id, MarketOrderMsg(order))
            if self.log_orders:
                self.logEvent("ORDER_SUBMITTED", order.to_dict(), deepcopy_event=False)

        else:
            warnings.warn(
                "TradingAgent ignored market order of quantity zero: {}", order
            )

    def place_multiple_orders(
        self, orders: List[Union[LimitOrder, MarketOrder]]
    ) -> None:
        """
        Used by any Trading Agent subclass to place multiple orders at the same time.

        Arguments:
            orders: A list of Orders to place with the exchange as a single batch.
        """

        messages = []

        for order in orders:
            if isinstance(order, LimitOrder):
                messages.append(LimitOrderMsg(order))
            elif isinstance(order, MarketOrder):
                messages.append(MarketOrderMsg(order))
            else:
                raise Exception("Expected LimitOrder or MarketOrder")

            # Copy the intended order for logging, so any changes made to it elsewhere
            # don't retroactively alter our "as placed" log of the order.  Eventually
            # it might be nice to make the whole history of the order into transaction
            # objects inside the order (we're halfway there) so there CAN be just a single
            # object per order, that never alters its original state, and eliminate all
            # these copies.
            self.orders[order.order_id] = deepcopy(order)

            if self.log_orders:
                self.logEvent("ORDER_SUBMITTED", order.to_dict(), deepcopy_event=False)

        if len(messages) > 0:
            self.send_message_batch(self.exchange_id, messages)

    def cancel_order(
        self, order: LimitOrder, tag: Optional[str] = None, metadata: dict = {}
    ) -> None:
        """
        Used by derived classes of TradingAgent to cancel a limit order.

        The order must currently appear in the agent's open orders list.

        Arguments:
            order: The limit order to cancel.
            tag:
            metadata:
        """

        if isinstance(order, LimitOrder):
            self.send_message(self.exchange_id, CancelOrderMsg(order, tag, metadata))
            if self.log_orders:
                self.logEvent("CANCEL_SUBMITTED", order.to_dict(), deepcopy_event=False)
        else:
            warnings.warn(f"Order {order} of type, {type(order)} cannot be cancelled")

    def cancel_all_orders(self):
        """
        Cancels all current limit orders held by this agent.
        """

        for order in self.orders.values():
            if isinstance(order, LimitOrder):
                self.cancel_order(order)

    def partial_cancel_order(
        self,
        order: LimitOrder,
        quantity: int,
        tag: Optional[str] = None,
        metadata: dict = {},
    ) -> None:
        """
        Used by any Trading Agent subclass to modify any existing limit order.

        The order must currently appear in the agent's open orders list.
        Arguments:
            order: The limit order to partially cancel.
            quantity:
            tag:
            metadata:
        """

        self.send_message(
            self.exchange_id, PartialCancelOrderMsg(order, quantity, tag, metadata)
        )

        if self.log_orders:
            self.logEvent("CANCEL_PARTIAL_ORDER", order.to_dict(), deepcopy_event=False)

    def modify_order(self, order: LimitOrder, new_order: LimitOrder) -> None:
        """
        Used by any Trading Agent subclass to modify any existing limit order.

        The order must currently appear in the agent's open orders list.  Some
        additional tests might be useful here to ensure the old and new orders are
        the same in some way.

        Arguments:
            order: The existing limit order.
            new_order: The limit order to update the existing order with.
        """

        self.send_message(self.exchange_id, ModifyOrderMsg(order, new_order))

        if self.log_orders:
            self.logEvent("MODIFY_ORDER", order.to_dict(), deepcopy_event=False)

    def replace_order(self, order: LimitOrder, new_order: LimitOrder) -> None:
        """
        Used by any Trading Agent subclass to replace any existing limit order.

        The order must currently appear in the agent's open orders list.  Some
        additional tests might be useful here to ensure the old and new orders are
        the same in some way.

        Arguments:
            order: The existing limit order.
            new_order: The new limit order to replace the existing order with.
        """

        self.send_message(self.exchange_id, ReplaceOrderMsg(self.id, order, new_order))

        if self.log_orders:
            self.logEvent("REPLACE_ORDER", order.to_dict(), deepcopy_event=False)

    def order_executed(self, order: Order) -> None:
        """
        Handles OrderExecuted messages from an exchange agent.

        Subclasses may wish to extend, but should still call parent method for basic
        portfolio/returns tracking.

        Arguments:
            order: The order that has been executed by the exchange.
        """

        logger.debug(f"Received notification of execution for: {order}")

        if self.log_orders:
            self.logEvent("ORDER_EXECUTED", order.to_dict(), deepcopy_event=False)

        # At the very least, we must update CASH and holdings at execution time.
        qty = order.quantity if order.side.is_bid() else -1 * order.quantity
        sym = order.symbol

        if sym in self.holdings:
            self.holdings[sym] += qty
        else:
            self.holdings[sym] = qty

        if self.holdings[sym] == 0:
            del self.holdings[sym]

        # As with everything else, CASH holdings are in CENTS.
        self.holdings["CASH"] -= qty * order.fill_price

        # If this original order is now fully executed, remove it from the open orders list.
        # Otherwise, decrement by the quantity filled just now.  It is _possible_ that due
        # to timing issues, it might not be in the order list (i.e. we issued a cancellation
        # but it was executed first, or something).
        if order.order_id in self.orders:
            o = self.orders[order.order_id]

            if order.quantity >= o.quantity:
                del self.orders[order.order_id]
            else:
                o.quantity -= order.quantity

        else:
            warnings.warn(f"Execution received for order not in orders list: {order}")

        logger.debug(f"After order execution, agent open orders: {self.orders}")

        self.logEvent("HOLDINGS_UPDATED", self.holdings)

    def order_accepted(self, order: LimitOrder) -> None:
        """
        Handles OrderAccepted messages from an exchange agent.

        Subclasses may wish to extend.

        Arguments:
            order: The order that has been accepted from the exchange.
        """

        logger.debug(f"Received notification of acceptance for: {order}")

        if self.log_orders:
            self.logEvent("ORDER_ACCEPTED", order.to_dict(), deepcopy_event=False)

        # We may later wish to add a status to the open orders so an agent can tell whether
        # a given order has been accepted or not (instead of needing to override this method).

    def order_cancelled(self, order: LimitOrder) -> None:
        """
        Handles OrderCancelled messages from an exchange agent.

        Subclasses may wish to extend.

        Arguments:
            order: The order that has been cancelled by the exchange.
        """

        logger.debug(f"Received notification of cancellation for: {order}")

        if self.log_orders:
            self.logEvent("ORDER_CANCELLED", order.to_dict(), deepcopy_event=False)

        # Remove the cancelled order from the open orders list.  We may of course wish to have
        # additional logic here later, so agents can easily "look for" cancelled orders.  Of
        # course they can just override this method.
        if order.order_id in self.orders:
            del self.orders[order.order_id]
        else:
            warnings.warn(
                f"Cancellation received for order not in orders list: {order}"
            )

    def order_partial_cancelled(self, order: LimitOrder) -> None:
        """
        Handles OrderCancelled messages from an exchange agent.

        Subclasses may wish to extend.

        Arguments:
            order: The order that has been partially cancelled by the exchange.
        """

        logger.debug(f"Received notification of partial cancellation for: {order}")

        if self.log_orders:
            self.logEvent("PARTIAL_CANCELLED", order.to_dict())

        # if orders still in the list of agent's order update agent's knowledge of
        # current state of the order
        if order.order_id in self.orders:
            self.orders[order.order_id] = order

        else:
            warnings.warn(
                f"partial cancellation received for order not in orders list: {order}"
            )

        logger.debug(
            f"After order partial cancellation, agent open orders: {self.orders}"
        )

        self.logEvent("HOLDINGS_UPDATED", self.holdings)

    def order_modified(self, order: LimitOrder) -> None:
        """
        Handles OrderModified messages from an exchange agent.

        Subclasses may wish to extend.

        Arguments:
            order: The order that has been modified at the exchange.
        """

        logger.debug(f"Received notification of modification for: {order}")

        if self.log_orders:
            self.logEvent("ORDER_MODIFIED", order.to_dict())

        # if orders still in the list of agent's order update agent's knowledge of
        # current state of the order
        if order.order_id in self.orders:
            self.orders[order.order_id] = order

        else:
            warnings.warn("Execution received for order not in orders list: {order}")

        logger.debug(f"After order modification, agent open orders: {self.orders}")

        self.logEvent("HOLDINGS_UPDATED", self.holdings)

    def order_replaced(self, old_order: LimitOrder, new_order: LimitOrder) -> None:
        """
        Handles OrderReplaced messages from an exchange agent.

        Subclasses may wish to extend.

        Arguments:
            order: The order that has been modified at the exchange.
        """

        logger.debug(f"Received notification of replacement for: {old_order}")

        if self.log_orders:
            self.logEvent("ORDER_REPLACED", old_order.to_dict())

        # if orders still in the list of agent's order update agent's knowledge of
        # current state of the order
        if old_order.order_id in self.orders:
            del self.orders[old_order.order_id]
        else:
            warnings.warn(
                f"Execution received for order not in orders list: {old_order}"
            )

        self.orders[new_order.order_id] = new_order

        logger.debug(f"After order replacement, agent open orders: {self.orders}")

        # After execution, log holdings.
        self.logEvent("HOLDINGS_UPDATED", self.holdings)

    def market_closed(self) -> None:
        """
        Handles MarketClosedMsg messages from an exchange agent.

        Subclasses may wish to extend.
        """

        logger.debug("Received notification of market closure.")

        # Log this activity.
        self.logEvent("MKT_CLOSED")

        # Remember that this has happened.
        self.mkt_closed = True

    def query_last_trade(self, symbol: str, price: int) -> None:
        """
        Handles QueryLastTradeResponseMsg messages from an exchange agent.

        Arguments:
            symbol: The symbol that was queried.
            price: The price at which the last trade executed at.
        """

        self.last_trade[symbol] = price

        logger.debug(
            "Received last trade price of {} for {}.".format(
                self.last_trade[symbol], symbol
            )
        )

        if self.mkt_closed:
            # Note this as the final price of the day.
            self.daily_close_price[symbol] = self.last_trade[symbol]

            logger.debug(
                "Received daily close price of {} for {}.".format(
                    self.last_trade[symbol], symbol
                )
            )

    def query_spread(
        self,
        symbol: str,
        price: int,
        bids: List[List[Tuple[int, int]]],
        asks: List[List[Tuple[int, int]]],
        book: str,
    ) -> None:
        """
        Handles QuerySpreadResponseMsg messages from an exchange agent.

        Arguments:
            symbol: The symbol that was queried.
            price:
            bids:
            asks:
            book:
        """

        # The spread message now also includes last price for free.
        self.query_last_trade(symbol, price)

        self.known_bids[symbol] = bids
        self.known_asks[symbol] = asks

        if bids:
            best_bid, best_bid_qty = (bids[0][0], bids[0][1])
        else:
            best_bid, best_bid_qty = ("No bids", 0)

        if asks:
            best_ask, best_ask_qty = (asks[0][0], asks[0][1])
        else:
            best_ask, best_ask_qty = ("No asks", 0)

        logger.debug(
            "Received spread of {} @ {} / {} @ {} for {}".format(
                best_bid_qty, best_bid, best_ask_qty, best_ask, symbol
            )
        )

        self.logEvent("BID_DEPTH", bids)
        self.logEvent("ASK_DEPTH", asks)
        self.logEvent(
            "IMBALANCE", [sum([x[1] for x in bids]), sum([x[1] for x in asks])]
        )

        self.book = book

    def handle_market_data(self, message: MarketDataMsg) -> None:
        """
        Handles Market Data messages for agents using subscription mechanism.

        Arguments:
            message: The market data message,
        """

        if isinstance(message, L2DataMsg):
            symbol = message.symbol
            self.known_asks[symbol] = message.asks
            self.known_bids[symbol] = message.bids
            self.last_trade[symbol] = message.last_transaction
            self.exchange_ts[symbol] = message.exchange_ts

    def query_order_stream(self, symbol: str, orders) -> None:
        """
        Handles QueryOrderStreamResponseMsg messages from an exchange agent.

        It is up to the requesting agent to do something with the data, which is a list
        of dictionaries keyed by order id. The list index is 0 for orders since the most
        recent trade, 1 for orders that led up to the most recent trade, and so on.
        Agents are not given index 0 (orders more recent than the last trade).

        Arguments:
            symbol: The symbol that was queried.
            orders:
        """

        self.stream_history[symbol] = orders

    def query_transacted_volume(
        self, symbol: str, bid_volume: int, ask_volume: int
    ) -> None:
        """
        Handles the QueryTransactedVolResponseMsg messages from the exchange agent.

        Arguments:
            symbol: The symbol that was queried.
            bid_vol: The volume that has transacted on the bid side for the queried period.
            ask_vol: The volume that has transacted on the ask side for the queried period.
        """

        self.transacted_volume[symbol] = (bid_volume, ask_volume)

    # Utility functions that perform calculations from available knowledge, but implement no
    # particular strategy.

    def get_known_bid_ask(self, symbol: str, best: bool = True):
        """
        Extract the current known bid and asks.

        This does NOT request new information.

        Arguments:
            symbol: The symbol to query.
            best:
        """

        if best:
            bid = self.known_bids[symbol][0][0] if self.known_bids[symbol] else None
            ask = self.known_asks[symbol][0][0] if self.known_asks[symbol] else None
            bid_vol = self.known_bids[symbol][0][1] if self.known_bids[symbol] else 0
            ask_vol = self.known_asks[symbol][0][1] if self.known_asks[symbol] else 0
            return bid, bid_vol, ask, ask_vol
        else:
            bids = self.known_bids[symbol] if self.known_bids[symbol] else None
            asks = self.known_asks[symbol] if self.known_asks[symbol] else None
            return bids, asks

    def get_known_liquidity(self, symbol: str, within: float = 0.00) -> Tuple[int, int]:
        """
        Extract the current bid and ask liquidity within a certain proportion of the
        inside bid and ask.  (i.e. within=0.01 means to report total BID shares
        within 1% of the best bid price, and total ASK shares within 1% of the best
        ask price)

        Arguments:
            symbol: The symbol to query.
            within:

        Returns:
            (bid_liquidity, ask_liquidity).  Note that this is from the order book
            perspective, not the agent perspective.  (The agent would be selling into
            the bid liquidity, etc.)
        """

        bid_liq = self.get_book_liquidity(self.known_bids[symbol], within)
        ask_liq = self.get_book_liquidity(self.known_asks[symbol], within)

        logger.debug("Bid/ask liq: {}, {}".format(bid_liq, ask_liq))
        logger.debug("Known bids: {}".format(self.known_bids[self.symbol]))
        logger.debug("Known asks: {}".format(self.known_asks[self.symbol]))

        return bid_liq, ask_liq

    def get_book_liquidity(self, book: Iterable[Tuple[int, int]], within: float) -> int:
        """
        Helper function for the above.  Checks one side of the known order book.

        Arguments:
            book:
            within:
        """
        liq = 0
        for i, (price, shares) in enumerate(book):
            if i == 0:
                best = price

            # Is this price within "within" proportion of the best price?
            if abs(best - price) <= int(round(best * within)):
                logger.debug(
                    "Within {} of {}: {} with {} shares".format(
                        within, best, price, shares
                    )
                )
                liq += shares

        return liq

    def mark_to_market(
        self, holdings: Mapping[str, int], use_midpoint: bool = False
    ) -> int:
        """
        Marks holdings to market (including cash).

        Arguments:
            holdings:
            use_midpoint:
        """

        cash = holdings["CASH"]

        cash += self.basket_size * self.nav_diff

        for symbol, shares in holdings.items():
            if symbol == "CASH":
                continue

            if use_midpoint:
                bid, ask, midpoint = self.get_known_bid_ask_midpoint(symbol)
                if bid is None or ask is None or midpoint is None:
                    value = self.last_trade[symbol] * shares
                else:
                    value = midpoint * shares
            else:
                value = self.last_trade[symbol] * shares

            cash += value

            self.logEvent(
                "MARK_TO_MARKET",
                "{} {} @ {} == {}".format(
                    shares, symbol, self.last_trade[symbol], value
                ),
            )

        self.logEvent("MARKED_TO_MARKET", cash)

        return cash

    def get_holdings(self, symbol: str) -> int:
        """
        Gets holdings.  Returns zero for any symbol not held.

        Arguments:
            symbol: The symbol to query.
        """

        return self.holdings[symbol] if symbol in self.holdings else 0

    def get_known_bid_ask_midpoint(
        self, symbol: str
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Get the known best bid, ask, and bid/ask midpoint from cached data. No volume.

        Arguments:
            symbol: The symbol to query.
        """

        bid = self.known_bids[symbol][0][0] if self.known_bids[symbol] else None
        ask = self.known_asks[symbol][0][0] if self.known_asks[symbol] else None

        midpoint = (
            int(round((bid + ask) / 2)) if bid is not None and ask is not None else None
        )

        return bid, ask, midpoint

    def get_average_transaction_price(self) -> float:
        """Calculates the average price paid (weighted by the order size)."""

        return round(
            sum(
                executed_order.quantity * executed_order.fill_price
                for executed_order in self.executed_orders
            )
            / sum(executed_order.quantity for executed_order in self.executed_orders),
            2,
        )

    def fmt_holdings(self, holdings: Mapping[str, int]) -> str:
        """
        Prints holdings.

        Standard dictionary->string representation is almost fine, but it is less
        confusing to see the CASH holdings in dollars and cents, instead of just integer
        cents.  We could change to a Holdings object that knows to print CASH "special".

        Arguments:
            holdings:
        """

        h = ""
        for k, v in sorted(holdings.items()):
            if k == "CASH":
                continue
            h += "{}: {}, ".format(k, v)

        # There must always be a CASH entry.
        h += "{}: {}".format("CASH", holdings["CASH"])
        h = "{ " + h + " }"
        return h
