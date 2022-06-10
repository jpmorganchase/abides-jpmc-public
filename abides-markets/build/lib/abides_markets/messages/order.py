from abc import ABC
from dataclasses import dataclass

from abides_core import Message

from ..orders import LimitOrder, MarketOrder


@dataclass
class OrderMsg(Message, ABC):
    pass


@dataclass
class LimitOrderMsg(OrderMsg):
    order: LimitOrder


@dataclass
class MarketOrderMsg(OrderMsg):
    order: MarketOrder


@dataclass
class CancelOrderMsg(OrderMsg):
    order: LimitOrder
    tag: str
    metadata: dict


@dataclass
class PartialCancelOrderMsg(OrderMsg):
    order: LimitOrder
    quantity: int
    tag: str
    metadata: dict


@dataclass
class ModifyOrderMsg(OrderMsg):
    old_order: LimitOrder
    new_order: LimitOrder


@dataclass
class ReplaceOrderMsg(OrderMsg):
    agent_id: int
    old_order: LimitOrder
    new_order: LimitOrder
