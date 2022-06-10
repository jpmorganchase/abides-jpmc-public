from copy import deepcopy

from abides_markets.messages.orderbook import OrderModifiedMsg
from abides_markets.orders import LimitOrder, Order, Side

from . import setup_book_with_orders, SYMBOL, TIME

# fmt: off


def test_modify_order_quantity_down():
    book, agent, orders = setup_book_with_orders(
        bids=[
            (100, [40, 10]),
            (200, [10, 30, 20, 10]),
        ],
        asks=[
            (300, [10, 50, 20]),
            (400, [40, 10]),
            (500, [20]),
        ],
    )

    # Modify order quantity down from 40 to 30
    modified_order = deepcopy(orders[0])
    modified_order.quantity = 30

    book.modify_order(orders[0], modified_order)

    assert book.get_l3_bid_data() == [
        (200, [10, 30, 20, 10]),
        (100, [30, 10]),
    ]

    assert len(agent.messages) == 1
    assert agent.messages[0][0] == 1
    assert isinstance(agent.messages[0][1], OrderModifiedMsg)

    assert agent.messages[0][1].new_order.agent_id == 1
    assert agent.messages[0][1].new_order.side == Side.BID
    assert agent.messages[0][1].new_order.limit_price == 100
    assert agent.messages[0][1].new_order.quantity == 30


def test_modify_order_quantity_up():
    book, agent, orders = setup_book_with_orders(
        bids=[
            (100, [40, 10]),
            (200, [10, 30, 20, 10]),
        ],
        asks=[
            (300, [10, 50, 20]),
            (400, [40, 10]),
            (500, [20]),
        ],
    )

    # Modify order quantity up from 40 to 70
    modified_order = deepcopy(orders[0])
    modified_order.quantity = 70

    book.modify_order(orders[0], modified_order)

    assert book.get_l3_bid_data() == [
        (200, [10, 30, 20, 10]),
        (100, [10, 70]),
    ]

    assert len(agent.messages) == 1
    assert agent.messages[0][0] == 1
    assert isinstance(agent.messages[0][1], OrderModifiedMsg)

    assert agent.messages[0][1].new_order.agent_id == 1
    assert agent.messages[0][1].new_order.side == Side.BID
    assert agent.messages[0][1].new_order.limit_price == 100
    assert agent.messages[0][1].new_order.quantity == 70
