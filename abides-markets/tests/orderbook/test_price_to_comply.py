from copy import deepcopy

from abides_markets.messages.orderbook import OrderAcceptedMsg, OrderExecutedMsg
from abides_markets.order_book import OrderBook
from abides_markets.orders import LimitOrder, MarketOrder, Side


from . import FakeExchangeAgent, SYMBOL, TIME


def test_create_price_to_comply_order():
    order = LimitOrder(
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=10,
        side=Side.BID,
        is_price_to_comply=True,
        limit_price=100,
    )

    agent = FakeExchangeAgent()
    book = OrderBook(agent, SYMBOL)
    book.handle_limit_order(deepcopy(order))

    hidden_half = deepcopy(order)
    hidden_half.is_hidden = True
    hidden_half.limit_price += 1

    visible_half = order

    assert len(book.asks) == 0
    assert len(book.bids) == 2

    assert book.bids[0].hidden_orders == [
        (hidden_half, dict(ptc_hidden=True, ptc_other_half=visible_half))
    ]
    assert book.bids[0].visible_orders == []
    assert book.bids[1].hidden_orders == []
    assert book.bids[1].visible_orders == [
        (visible_half, dict(ptc_hidden=False, ptc_other_half=hidden_half))
    ]


def test_fill_price_to_comply_order():
    order = LimitOrder(
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=10,
        side=Side.BID,
        is_price_to_comply=True,
        limit_price=100,
    )

    agent = FakeExchangeAgent()
    book = OrderBook(agent, SYMBOL)
    book.handle_limit_order(order)

    hidden_half = deepcopy(order)
    hidden_half.is_hidden = True
    hidden_half.limit_price += 1

    visible_half = order

    market_order = MarketOrder(
        agent_id=2,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=10,
        side=Side.ASK,
    )

    book.handle_market_order(market_order)

    assert len(book.asks) == 0
    assert len(book.bids) == 0

    assert len(agent.messages) == 3

    assert agent.messages[0][0] == 1
    assert isinstance(agent.messages[0][1], OrderAcceptedMsg)
    assert agent.messages[0][1].order.agent_id == 1
    assert agent.messages[0][1].order.side == Side.BID
    assert agent.messages[0][1].order.quantity == 10

    assert agent.messages[1][0] == 1
    assert isinstance(agent.messages[1][1], OrderExecutedMsg)
    assert agent.messages[1][1].order.agent_id == 1
    assert agent.messages[1][1].order.side == Side.BID
    assert agent.messages[1][1].order.fill_price == 101
    assert agent.messages[1][1].order.quantity == 10

    assert agent.messages[2][0] == 2
    assert isinstance(agent.messages[2][1], OrderExecutedMsg)
    assert agent.messages[2][1].order.agent_id == 2
    assert agent.messages[2][1].order.side == Side.ASK
    assert agent.messages[2][1].order.fill_price == 101
    assert agent.messages[2][1].order.quantity == 10


def test_cancel_price_to_comply_order():
    order = LimitOrder(
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=10,
        side=Side.BID,
        is_price_to_comply=True,
        limit_price=100,
    )

    agent = FakeExchangeAgent()
    book = OrderBook(agent, SYMBOL)
    book.handle_limit_order(order)

    assert book.cancel_order(order) == True

    assert len(book.asks) == 0
    assert len(book.bids) == 0


def test_modify_price_to_comply_order():
    pass

    # TODO


def test_replace_price_to_comply_order():
    old_order = LimitOrder(
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=10,
        side=Side.BID,
        is_price_to_comply=True,
        limit_price=100,
    )

    agent = FakeExchangeAgent()
    book = OrderBook(agent, SYMBOL)
    book.handle_limit_order(old_order)

    assert len(book.asks) == 0
    assert len(book.bids) == 2

    new_order = LimitOrder(
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=10,
        side=Side.ASK,
        is_price_to_comply=False,
        limit_price=100,
    )

    book.replace_order(1, old_order, new_order)

    assert len(book.asks) == 1
    assert len(book.bids) == 0
