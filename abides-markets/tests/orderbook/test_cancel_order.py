from abides_markets.orders import Side

from . import setup_book_with_orders

# fmt: off


def test_cancel_order():
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

    # Cancel bid order, mid book
    book.cancel_order(orders[1])

    assert book.get_l3_bid_data() == [
        (200, [10, 30, 20, 10]),
        (100, [40]),
    ]

    assert len(agent.messages) == 1
    assert agent.messages[0][0] == 1
    assert agent.messages[0][1].order.agent_id == 1
    assert agent.messages[0][1].order.side == Side.BID
    assert agent.messages[0][1].order.limit_price == 100
    assert agent.messages[0][1].order.quantity == 10

    agent.reset()

    # Cancel ask order, end of book
    book.cancel_order(orders[-1])

    assert book.get_l3_ask_data() == [
        (300, [10, 50, 20]),
        (400, [40, 10]),
    ]

    assert len(agent.messages) == 1
    assert agent.messages[0][0] == 1
    assert agent.messages[0][1].order.agent_id == 1
    assert agent.messages[0][1].order.side == Side.ASK
    assert agent.messages[0][1].order.limit_price == 500
    assert agent.messages[0][1].order.quantity == 20
