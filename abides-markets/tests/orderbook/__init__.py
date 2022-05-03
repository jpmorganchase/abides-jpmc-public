from typing import List, Tuple

from abides_core import Message
from abides_markets.order_book import OrderBook
from abides_markets.orders import LimitOrder, Side


SYMBOL = "X"
TIME = 0


class FakeExchangeAgent:
    def __init__(self):
        self.messages = []
        self.current_time = TIME
        self.mkt_open = TIME
        self.book_logging = None
        self.stream_history = 10

    def reset(self):
        self.messages = []

    def send_message(self, recipient_id: int, message: Message, _: int = 0):
        self.messages.append((recipient_id, message))

    def logEvent(self, *args, **kwargs):
        pass


def setup_book_with_orders(
    bids: List[Tuple[int, List[int]]] = [], asks: List[Tuple[int, List[int]]] = []
) -> Tuple[OrderBook, FakeExchangeAgent, List[LimitOrder]]:
    agent = FakeExchangeAgent()
    book = OrderBook(agent, SYMBOL)
    orders = []

    for price, quantities in bids:
        for quantity in quantities:
            order = LimitOrder(1, TIME, SYMBOL, quantity, Side.BID, price)
            book.handle_limit_order(order)
            orders.append(order)

    for price, quantities in asks:
        for quantity in quantities:
            order = LimitOrder(1, TIME, SYMBOL, quantity, Side.ASK, price)
            book.handle_limit_order(order)
            orders.append(order)

    agent.reset()

    return book, agent, orders
