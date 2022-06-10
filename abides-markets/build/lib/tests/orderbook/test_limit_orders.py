import pytest

from abides_markets.messages.orderbook import OrderExecutedMsg
from abides_markets.order_book import OrderBook
from abides_markets.orders import LimitOrder, Side
from abides_markets.price_level import PriceLevel

from . import setup_book_with_orders, FakeExchangeAgent, SYMBOL, TIME


def test_handle_limit_orders():
    # Test insert on bid side
    bid_order = LimitOrder(
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=10,
        side=Side.BID,
        limit_price=100,
    )

    agent = FakeExchangeAgent()
    book = OrderBook(agent, SYMBOL)
    book.handle_limit_order(bid_order)

    assert book.bids == [PriceLevel([(bid_order, {})])]
    assert book.asks == []

    assert len(agent.messages) == 1
    assert agent.messages[0][0] == 1
    assert agent.messages[0][1].order.agent_id == 1
    assert agent.messages[0][1].order.side == Side.BID
    assert agent.messages[0][1].order.limit_price == 100
    assert agent.messages[0][1].order.quantity == 10

    # Test insert on ask side
    ask_order = LimitOrder(
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=10,
        side=Side.ASK,
        limit_price=100,
    )

    agent = FakeExchangeAgent()
    book = OrderBook(agent, SYMBOL)
    book.handle_limit_order(ask_order)

    assert book.bids == []
    assert book.asks == [PriceLevel([(ask_order, {})])]

    assert len(agent.messages) == 1
    assert agent.messages[0][0] == 1
    assert agent.messages[0][1].order.agent_id == 1
    assert agent.messages[0][1].order.side == Side.ASK
    assert agent.messages[0][1].order.limit_price == 100
    assert agent.messages[0][1].order.quantity == 10


def test_handle_hidden_limit_orders():
    # Test insert on bid side
    bid_order = LimitOrder(
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=10,
        side=Side.BID,
        is_hidden=True,
        limit_price=100,
    )

    agent = FakeExchangeAgent()
    book = OrderBook(agent, SYMBOL)
    book.handle_limit_order(bid_order)

    assert book.bids == [PriceLevel([(bid_order, {})])]
    assert book.asks == []

    assert len(agent.messages) == 1
    assert agent.messages[0][0] == 1
    assert agent.messages[0][1].order.agent_id == 1
    assert agent.messages[0][1].order.side == Side.BID
    assert agent.messages[0][1].order.is_hidden == True
    assert agent.messages[0][1].order.limit_price == 100
    assert agent.messages[0][1].order.quantity == 10

    # Test insert on ask side
    ask_order = LimitOrder(
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=10,
        side=Side.ASK,
        is_hidden=True,
        limit_price=100,
    )

    agent = FakeExchangeAgent()
    book = OrderBook(agent, SYMBOL)
    book.handle_limit_order(ask_order)

    assert book.bids == []
    assert book.asks == [PriceLevel([(ask_order, {})])]

    assert len(agent.messages) == 1
    assert agent.messages[0][0] == 1
    assert agent.messages[0][1].order.agent_id == 1
    assert agent.messages[0][1].order.side == Side.ASK
    assert agent.messages[0][1].order.is_hidden == True
    assert agent.messages[0][1].order.limit_price == 100
    assert agent.messages[0][1].order.quantity == 10


def test_handle_matching_limit_orders():
    # Test insert on bid side
    book, agent, _ = setup_book_with_orders(
        asks=[
            (100, [30]),
        ],
    )

    bid_order = LimitOrder(
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=30,
        side=Side.BID,
        is_hidden=False,
        limit_price=110,
    )

    book.handle_limit_order(bid_order)

    assert book.bids == []
    assert book.asks == []

    assert len(agent.messages) == 2

    assert agent.messages[0][0] == 1
    assert isinstance(agent.messages[0][1], OrderExecutedMsg)
    assert agent.messages[0][1].order.agent_id == 1
    assert agent.messages[0][1].order.side == Side.ASK
    assert agent.messages[0][1].order.limit_price == 100
    assert agent.messages[1][1].order.fill_price == 100
    assert agent.messages[0][1].order.quantity == 30

    assert agent.messages[1][0] == 1
    assert isinstance(agent.messages[1][1], OrderExecutedMsg)
    assert agent.messages[1][1].order.agent_id == 1
    assert agent.messages[1][1].order.side == Side.BID
    assert agent.messages[1][1].order.limit_price == 110
    assert agent.messages[1][1].order.fill_price == 100
    assert agent.messages[1][1].order.quantity == 30

    # Test insert on ask side
    book, agent, _ = setup_book_with_orders(
        bids=[
            (100, [30]),
        ],
    )

    ask_order = LimitOrder(
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=30,
        side=Side.ASK,
        is_hidden=False,
        limit_price=90,
    )

    book.handle_limit_order(ask_order)

    assert book.bids == []
    assert book.asks == []

    assert len(agent.messages) == 2

    assert agent.messages[0][0] == 1
    assert isinstance(agent.messages[0][1], OrderExecutedMsg)
    assert agent.messages[0][1].order.agent_id == 1
    assert agent.messages[0][1].order.side == Side.BID
    assert agent.messages[0][1].order.limit_price == 100
    assert agent.messages[1][1].order.fill_price == 100
    assert agent.messages[0][1].order.quantity == 30

    assert agent.messages[1][0] == 1
    assert isinstance(agent.messages[1][1], OrderExecutedMsg)
    assert agent.messages[1][1].order.agent_id == 1
    assert agent.messages[1][1].order.side == Side.ASK
    assert agent.messages[1][1].order.limit_price == 90
    assert agent.messages[1][1].order.fill_price == 100
    assert agent.messages[1][1].order.quantity == 30


def test_handle_bad_limit_orders():
    agent = FakeExchangeAgent()
    book = OrderBook(agent, SYMBOL)

    # Symbol does not match book
    order = LimitOrder(
        agent_id=1,
        time_placed=TIME,
        symbol="BAD",
        quantity=10,
        side=Side.BID,
        is_hidden=True,
        limit_price=100,
    )

    with pytest.warns(UserWarning):
        book.handle_limit_order(order)

    # Order quantity not integer
    order = LimitOrder(
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=1.5,
        side=Side.BID,
        is_hidden=True,
        limit_price=100,
    )

    with pytest.warns(UserWarning):
        book.handle_limit_order(order)

    # Order quantity is negative
    order = LimitOrder(
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=-10,
        side=Side.BID,
        is_hidden=True,
        limit_price=100,
    )

    with pytest.warns(UserWarning):
        book.handle_limit_order(order)

    with pytest.warns(UserWarning):
        book.handle_limit_order(order)

    # Order limit price is negative
    order = LimitOrder(
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=10,
        side=Side.BID,
        is_hidden=True,
        limit_price=-100,
    )

    with pytest.warns(UserWarning):
        book.handle_limit_order(order)


def test_handle_insert_by_id_limit_order():
    agent = FakeExchangeAgent()
    book = OrderBook(agent, SYMBOL)

    order1 = LimitOrder(
        order_id=1,
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=10,
        side=Side.BID,
        limit_price=100,
    )

    order2 = LimitOrder(
        order_id=2,
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=20,
        side=Side.BID,
        limit_price=100,
    )

    order3 = LimitOrder(
        order_id=3,
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=30,
        side=Side.BID,
        limit_price=100,
        insert_by_id=True,
    )

    order4 = LimitOrder(
        order_id=4,
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=40,
        side=Side.BID,
        limit_price=100,
    )

    book.handle_limit_order(order1)
    book.handle_limit_order(order2)
    book.handle_limit_order(order4)

    # Insert out-of-order
    book.handle_limit_order(order3)

    assert book.bids[0].visible_orders == [
        (order1, {}),
        (order2, {}),
        (order3, {}),
        (order4, {}),
    ]
