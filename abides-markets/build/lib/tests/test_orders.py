from copy import deepcopy

import pytest

from abides_markets.orders import Order, LimitOrder, MarketOrder


TIME = 0


def test_order_id_generation():
    # Test incremental ID counter for MarketOrder class
    Order._order_id_counter = 0

    order1 = MarketOrder(1, TIME, "X", 1, True)
    order2 = MarketOrder(1, TIME, "X", 1, True)
    order3 = MarketOrder(1, TIME, "X", 1, True)

    assert order1.order_id == 0
    assert order2.order_id == 1
    assert order3.order_id == 2

    # Test incremental ID counter for LimitOrder class
    Order._order_id_counter = 0

    order1 = LimitOrder(1, TIME, "X", 1, True, 1)
    order2 = LimitOrder(1, TIME, "X", 1, True, 1)
    order3 = LimitOrder(1, TIME, "X", 1, True, 1)

    assert order1.order_id == 0
    assert order2.order_id == 1
    assert order3.order_id == 2

    # Test incremental ID counter for mix of order classes
    Order._order_id_counter = 0

    order1 = MarketOrder(1, TIME, "X", 1, True)
    order2 = LimitOrder(1, TIME, "X", 1, True, 1)

    assert order1.order_id == 0
    assert order2.order_id == 1

    # Test setting duplicate ID does not affect order ID generation
    Order._order_id_counter = 0

    order1 = MarketOrder(1, TIME, "X", 1, True)
    order2 = MarketOrder(1, TIME, "X", 1, True, order_id=0)
    order3 = MarketOrder(1, TIME, "X", 1, True)

    assert order1.order_id == 0
    assert order2.order_id == 0
    assert order3.order_id == 1


def test_order_equality():
    order1 = LimitOrder(1, TIME, "X", 1, True, 1)
    order2 = LimitOrder(1, TIME, "X", 1, True, 1)

    assert order1 == order1
    assert order1 == deepcopy(order1)

    assert order1 != order2


def test_base_order_init():
    with pytest.raises(TypeError):
        Order(1, TIME, "X", 1, True)
