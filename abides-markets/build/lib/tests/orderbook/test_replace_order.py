from abides_markets.messages.orderbook import OrderReplacedMsg
from abides_markets.orders import LimitOrder, Side

from . import setup_book_with_orders, SYMBOL, TIME

# fmt: off


def test_replace_order():
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

    # Replace 30 @ $200 with 50 @ $100, bid side of book
    new_order = LimitOrder(
        agent_id=1,
        time_placed=TIME,
        symbol=SYMBOL,
        quantity=50,
        side=Side.BID,
        limit_price=100,
    )

    book.replace_order(1, orders[3], new_order)

    assert book.get_l3_bid_data() == [
        (200, [10, 20, 10]),
        (100, [40, 10, 50]),
    ]

    assert len(agent.messages) == 1
    assert agent.messages[0][0] == 1
    assert isinstance(agent.messages[0][1], OrderReplacedMsg)

    assert agent.messages[0][1].old_order.agent_id == 1
    assert agent.messages[0][1].old_order.side == Side.BID
    assert agent.messages[0][1].old_order.limit_price == 200
    assert agent.messages[0][1].old_order.quantity == 30

    assert agent.messages[0][1].new_order.agent_id == 1
    assert agent.messages[0][1].new_order.side == Side.BID
    assert agent.messages[0][1].new_order.limit_price == 100
    assert agent.messages[0][1].new_order.quantity == 50
