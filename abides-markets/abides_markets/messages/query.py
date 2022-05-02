from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from abides_core import Message


@dataclass
class QueryMsg(Message, ABC):
    symbol: str


@dataclass
class QueryResponseMsg(Message, ABC):
    symbol: str
    mkt_closed: bool


@dataclass
class QueryLastTradeMsg(QueryMsg):
    # Inherited Fields:
    # symbol: str
    pass


@dataclass
class QueryLastTradeResponseMsg(QueryResponseMsg):
    # Inherited Fields:
    # symbol: str
    # mkt_closed: bool
    last_trade: Optional[int]


@dataclass
class QuerySpreadMsg(QueryMsg):
    # Inherited Fields:
    # symbol: str
    depth: int


@dataclass
class QuerySpreadResponseMsg(QueryResponseMsg):
    # Inherited Fields:
    # symbol: str
    # mkt_closed: bool
    depth: int
    bids: List[Tuple[int, int]]
    asks: List[Tuple[int, int]]
    last_trade: Optional[int]


@dataclass
class QueryOrderStreamMsg(QueryMsg):
    # Inherited Fields:
    # symbol: str
    length: int


@dataclass
class QueryOrderStreamResponseMsg(QueryResponseMsg):
    # Inherited Fields:
    # symbol: str
    # mkt_closed: bool
    length: int
    orders: List[Dict[str, Any]]


@dataclass
class QueryTransactedVolMsg(QueryMsg):
    # Inherited Fields:
    # symbol: str
    lookback_period: str


@dataclass
class QueryTransactedVolResponseMsg(QueryResponseMsg):
    # Inherited Fields:
    # symbol: str
    # mkt_closed: bool
    bid_volume: int
    ask_volume: int
