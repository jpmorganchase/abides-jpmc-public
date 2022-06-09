from abc import ABC
from dataclasses import dataclass
from abides_core import Message

@dataclass
class OrderMatchedWithWhomMsg(Message, ABC):
    matching_agent_id: int

@dataclass
class OrderMatchedValueAgentMsg(Message, ABC):
    price: int
    side: str #"SELL" or "BID"