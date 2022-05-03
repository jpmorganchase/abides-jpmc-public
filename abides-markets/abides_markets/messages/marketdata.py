import sys
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

from abides_core import Message, NanosecondTime

from ..orders import Side


@dataclass
class MarketDataSubReqMsg(Message, ABC):
    """
    Base class for creating or cancelling market data subscriptions with an
    ``ExchangeAgent``.

    Attributes:
        symbol: The symbol of the security to request a data subscription for.
        cancel: If True attempts to create a new subscription, if False attempts to
            cancel an existing subscription.
    """

    symbol: str
    cancel: bool = False


@dataclass
class MarketDataFreqBasedSubReqMsg(MarketDataSubReqMsg, ABC):
    """
    Base class for creating or cancelling market data subscriptions with an
    ``ExchangeAgent``.

    Attributes:
        symbol: The symbol of the security to request a data subscription for.
        cancel: If True attempts to create a new subscription, if False attempts to
            cancel an existing subscription.
        freq: The frequency in nanoseconds^-1 at which to receive market updates.
    """

    # Inherited Fields:
    # symbol: str
    # cancel: bool = False
    freq: int = 1


@dataclass
class MarketDataEventBasedSubReqMsg(MarketDataSubReqMsg, ABC):
    """
    Base class for creating or cancelling market data subscriptions with an
    ``ExchangeAgent``.

    Attributes:
        symbol: The symbol of the security to request a data subscription for.
        cancel: If True attempts to create a new subscription, if False attempts to
            cancel an existing subscription.
    """

    # Inherited Fields:
    # symbol: str
    # cancel: bool = False


@dataclass
class L1SubReqMsg(MarketDataFreqBasedSubReqMsg):
    """
    This message requests the creation or cancellation of a subscription to L1 order
    book data from an ``ExchangeAgent``.

    Attributes:
        symbol: The symbol of the security to request a data subscription for.
        cancel: If True attempts to create a new subscription, if False attempts to
            cancel an existing subscription.
        freq: The frequency in nanoseconds^-1 at which to receive market updates.
    """

    # Inherited Fields:
    # symbol: str
    # cancel: bool = False
    # freq: int = 1
    pass


@dataclass
class L2SubReqMsg(MarketDataFreqBasedSubReqMsg):
    """
    This message requests the creation or cancellation of a subscription to L2 order
    book data from an ``ExchangeAgent``.

    Attributes:
        symbol: The symbol of the security to request a data subscription for.
        cancel: If True attempts to create a new subscription, if False attempts to
            cancel an existing subscription.
        freq: The frequency in nanoseconds^-1 at which to receive market updates.
        depth: The maximum number of price levels on both sides of the order book to
            return data for. Defaults to the entire book.
    """

    # Inherited Fields:
    # symbol: str
    # cancel: bool = False
    # freq: int = 1
    depth: int = sys.maxsize


@dataclass
class L3SubReqMsg(MarketDataFreqBasedSubReqMsg):
    """
    This message requests the creation or cancellation of a subscription to L3 order
    book data from an ``ExchangeAgent``.

    Attributes:
        symbol: The symbol of the security to request a data subscription for.
        cancel: If True attempts to create a new subscription, if False attempts to
            cancel an existing subscription.
        freq: The frequency in nanoseconds^-1 at which to receive market updates.
        depth: The maximum number of price levels on both sides of the order book to
            return data for. Defaults to the entire book.
    """

    # Inherited Fields:
    # symbol: str
    # cancel: bool = False
    # freq: int = 1
    depth: int = sys.maxsize


@dataclass
class TransactedVolSubReqMsg(MarketDataFreqBasedSubReqMsg):
    """
    This message requests the creation or cancellation of a subscription to transacted
    volume order book data from an ``ExchangeAgent``.

    Attributes:
        symbol: The symbol of the security to request a data subscription for.
        cancel: If True attempts to create a new subscription, if False attempts to
            cancel an existing subscription.
        freq: The frequency in nanoseconds^-1 at which to receive market updates.
        lookback: The period in time backwards from the present to sum the transacted
            volume for.
    """

    # Inherited Fields:
    # symbol: str
    # cancel: bool = False
    # freq: int = 1
    lookback: str = "1min"


@dataclass
class BookImbalanceSubReqMsg(MarketDataEventBasedSubReqMsg):
    """
    This message requests the creation or cancellation of a subscription to book
    imbalance events.

    Attributes:
        symbol: The symbol of the security to request a data subscription for.
        cancel: If True attempts to create a new subscription, if False attempts to
            cancel an existing subscription.
        min_imbalance: The minimum book imbalance needed to trigger this subscription.

    0.0 is no imbalance.
    1.0 is full imbalance (ie. liquidity drop).
    """

    # Inherited Fields:
    # symbol: str
    # cancel: bool = False
    min_imbalance: float = 1.0


@dataclass
class MarketDataMsg(Message, ABC):
    """
    Base class for returning market data subscription results from an ``ExchangeAgent``.

    The ``last_transaction`` and ``exchange_ts`` fields are not directly related to the
    subscription data but are included for bookkeeping purposes.

    Attributes:
        symbol: The symbol of the security this data is for.
        last_transaction: The time of the last transaction that happened on the exchange.
        exchange_ts: The time that the message was sent from the exchange.
    """

    symbol: str
    last_transaction: int
    exchange_ts: NanosecondTime


@dataclass
class MarketDataEventMsg(MarketDataMsg, ABC):
    """
    Base class for returning market data subscription results from an ``ExchangeAgent``.

    The ``last_transaction`` and ``exchange_ts`` fields are not directly related to the
    subscription data but are included for bookkeeping purposes.

    Attributes:
        symbol: The symbol of the security this data is for.
        last_transaction: The time of the last transaction that happened on the exchange.
        exchange_ts: The time that the message was sent from the exchange.
        stage: The stage of this event (start or finish).
    """

    class Stage(Enum):
        START = "START"
        FINISH = "FINISH"

    stage: Stage


@dataclass
class L1DataMsg(MarketDataMsg):
    """
    This message returns L1 order book data as part of an L1 data subscription.

    Attributes:
        symbol: The symbol of the security this data is for.
        last_transaction: The time of the last transaction that happened on the exchange.
        exchange_ts: The time that the message was sent from the exchange.
        bid: The best bid price and the available volume at that price.
        ask: The best ask price and the available volume at that price.
    """

    # Inherited Fields:
    # symbol: str
    # last_transaction: int
    # exchange_ts: NanosecondTime
    bid: Tuple[int, int]
    ask: Tuple[int, int]


@dataclass
class L2DataMsg(MarketDataMsg):
    """
    This message returns L2 order book data as part of an L2 data subscription.

    Attributes:
        symbol: The symbol of the security this data is for.
        last_transaction: The time of the last transaction that happened on the exchange.
        exchange_ts: The time that the message was sent from the exchange.
        bids: A list of tuples containing the price and available volume at each bid
            price level.
        asks: A list of tuples containing the price and available volume at each ask
            price level.
    """

    # Inherited Fields:
    # symbol: str
    # last_transaction: int
    # exchange_ts: NanosecondTime
    bids: List[Tuple[int, int]]
    asks: List[Tuple[int, int]]

    # TODO: include requested depth


@dataclass
class L3DataMsg(MarketDataMsg):
    """
    This message returns L3 order book data as part of an L3 data subscription.

    Attributes:
        symbol: The symbol of the security this data is for.
        last_transaction: The time of the last transaction that happened on the exchange.
        exchange_ts: The time that the message was sent from the exchange.
        bids: A list of tuples containing the price and a list of order sizes at each
            bid price level.
        asks: A list of tuples containing the price and a list of order sizes at each
            ask price level.
    """

    # Inherited Fields:
    # symbol: str
    # last_transaction: int
    # exchange_ts: NanosecondTime
    bids: List[Tuple[int, List[int]]]
    asks: List[Tuple[int, List[int]]]

    # TODO: include requested depth


@dataclass
class TransactedVolDataMsg(MarketDataMsg):
    """
    This message returns order book transacted volume data as part of an transacted
    volume data subscription.

    Attributes:
        symbol: The symbol of the security this data is for.
        last_transaction: The time of the last transaction that happened on the exchange.
        exchange_ts: The time that the message was sent from the exchange.
        bid_volume: The total transacted volume of bid orders for the given lookback period.
        ask_volume: The total transacted volume of ask orders for the given lookback period.
    """

    # Inherited Fields:
    # symbol: str
    # last_transaction: int
    # exchange_ts: NanosecondTime
    bid_volume: int
    ask_volume: int

    # TODO: include lookback period


@dataclass
class BookImbalanceDataMsg(MarketDataEventMsg):
    """
    Sent when the book imbalance reaches a certain threshold dictated in the
    subscription request message.

    Attributes:
        symbol: The symbol of the security this data is for.
        last_transaction: The time of the last transaction that happened on the exchange.
        exchange_ts: The time that the message was sent from the exchange.
        stage: The stage of this event (start or finish).
        imbalance: Proportional size of the imbalance.
        side: Side of the book that the imbalance is towards.
    """

    # Inherited Fields:
    # symbol: str
    # last_transaction: int
    # exchange_ts: pd.Timestamp
    # stage: MarketDataEventMsg.Stage
    imbalance: float
    side: Side
