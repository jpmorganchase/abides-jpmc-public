from dataclasses import dataclass
from typing import Dict, Optional

from abides_core import Message, NanosecondTime


@dataclass
class MarketClosedMsg(Message):
    """
    This message is sent from an ``ExchangeAgent`` to a ``TradingAgent`` when a ``TradingAgent`` has
    made a request that cannot be completed because the market the ``ExchangeAgent`` trades
    is closed.
    """

    pass


@dataclass
class MarketHoursRequestMsg(Message):
    """
    This message can be sent to an ``ExchangeAgent`` to query the opening hours of the market
    it trades. A ``MarketHoursMsg`` is sent in response.
    """

    pass


@dataclass
class MarketHoursMsg(Message):
    """
    This message is sent by an ``ExchangeAgent`` in response to a ``MarketHoursRequestMsg``
    message sent from a ``TradingAgent``.

    Attributes:
        mkt_open: The time that the market traded by the ``ExchangeAgent`` opens.
        mkt_close: The time that the market traded by the ``ExchangeAgent`` closes.
    """

    mkt_open: NanosecondTime
    mkt_close: NanosecondTime


@dataclass
class MarketClosePriceRequestMsg(Message):
    """
    This message can be sent to an ``ExchangeAgent`` to request that the close price of
    the market is sent when the exchange closes. This is used to accurately calculate
    the agent's final mark-to-market value.
    """


@dataclass
class MarketClosePriceMsg(Message):
    """
    This message is sent by an ``ExchangeAgent`` when the exchange closes to all agents
    that habve requested this message. The value is used to accurately calculate the
    agent's final mark-to-market value.

    Attributes:
        close_prices: A mapping of symbols to closing prices.
    """

    close_prices: Dict[str, Optional[int]]
