import pytest

from abides_markets.orders import LimitOrder, Side
from abides_markets.price_level import PriceLevel

from .. import reset_env


@pytest.fixture
def price_level():
    reset_env()
    return PriceLevel(
        [
            (LimitOrder(0, 0, "", 10, Side.BID, 100, is_hidden=False), {}),
            (LimitOrder(0, 0, "", 10, Side.BID, 100, is_hidden=True), {}),
            (LimitOrder(0, 0, "", 10, Side.BID, 100, is_hidden=False), {}),
            (LimitOrder(0, 0, "", 10, Side.BID, 100, is_hidden=True), {}),
            (LimitOrder(0, 0, "", 10, Side.BID, 100, is_hidden=False), {}),
        ]
    )


def test_init(price_level):
    assert len(price_level.visible_orders) == 3
    assert len(price_level.hidden_orders) == 2

    assert price_level.price == 100
    assert price_level.side == Side.BID


def test_bad_init():
    with pytest.raises(ValueError):
        _ = PriceLevel([])


def test_add_order(price_level):
    order = LimitOrder(0, 0, "", 10, Side.BID, 100, is_hidden=False)
    price_level.add_order(order)
    assert price_level.visible_orders[-1] == (order, {})

    order = LimitOrder(0, 0, "", 10, Side.BID, 100, is_hidden=True)
    price_level.add_order(order)
    assert price_level.hidden_orders[-1] == (order, {})


def test_update_order_quantity(price_level):
    # VISIBLE:

    # Update with lower price, same position kept in queue:
    assert price_level.update_order_quantity(0, 5) == True
    assert price_level.visible_orders[0][0].order_id == 0

    # Update with higher price, moved to end of queue:
    assert price_level.update_order_quantity(0, 15) == True
    assert price_level.visible_orders[-1][0].order_id == 0

    # HIDDEN:

    # Update with lower price, same position kept in queue:
    assert price_level.update_order_quantity(1, 5) == True
    assert price_level.hidden_orders[0][0].order_id == 1

    # Update with higher price, moved to end of queue:
    assert price_level.update_order_quantity(1, 15) == True
    assert price_level.hidden_orders[-1][0].order_id == 1

    # NOT IN BOOK:

    assert price_level.update_order_quantity(10, 5) == False


def test_remove_order(price_level):
    # VISIBLE:

    order, _ = price_level.remove_order(0)
    assert isinstance(order, LimitOrder)
    assert order.order_id == 0
    assert len(price_level.visible_orders) == 2

    # HIDDEN:

    order, _ = price_level.remove_order(1)
    assert isinstance(order, LimitOrder)
    assert order.order_id == 1
    assert len(price_level.hidden_orders) == 1

    # NOT IN BOOK:

    assert price_level.remove_order(10) == None


def test_peek(price_level):
    # VISIBLE:

    assert price_level.peek() == price_level.visible_orders[0]

    # HIDDEN:

    price_level.visible_orders = []
    assert price_level.peek() == price_level.hidden_orders[0]

    # EMPTY BOOK:

    price_level.hidden_orders = []
    with pytest.raises(ValueError):
        price_level.peek()


def test_pop(price_level):
    # VISIBLE:

    order = price_level.visible_orders[0]
    assert price_level.pop() == order

    # HIDDEN:

    price_level.visible_orders = []
    order = price_level.hidden_orders[0]
    assert price_level.pop() == order

    # EMPTY BOOK:

    price_level.hidden_orders = []
    with pytest.raises(ValueError):
        price_level.pop()


def test_order_is_match(price_level):
    # Test orders on opposite side of book:
    order = LimitOrder(0, 0, "", 10, Side.ASK, 90, is_hidden=False)
    assert price_level.order_is_match(order) == True

    order = LimitOrder(0, 0, "", 10, Side.ASK, 100, is_hidden=False)
    assert price_level.order_is_match(order) == True

    order = LimitOrder(0, 0, "", 10, Side.ASK, 110, is_hidden=False)
    assert price_level.order_is_match(order) == False

    # Test order on same side of book:
    order = LimitOrder(0, 0, "", 10, Side.BID, 100, is_hidden=False)

    with pytest.raises(ValueError):
        price_level.order_is_match(order)

    # Test order with empty price level:
    price_level.visible_orders = []
    price_level.hidden_orders = []

    with pytest.raises(ValueError):
        price_level.order_is_match(order)


def test_order_has_better_price(price_level):
    # Test orders on same side of book:
    order = LimitOrder(0, 0, "", 10, Side.BID, 90, is_hidden=False)
    assert price_level.order_has_better_price(order) == False

    order = LimitOrder(0, 0, "", 10, Side.BID, 100, is_hidden=False)
    assert price_level.order_has_better_price(order) == False

    order = LimitOrder(0, 0, "", 10, Side.BID, 110, is_hidden=False)
    assert price_level.order_has_better_price(order) == True

    # Test order on opposite side of book:
    order = LimitOrder(0, 0, "", 10, Side.ASK, 100, is_hidden=False)

    with pytest.raises(ValueError):
        price_level.order_has_better_price(order)

    # Test order with empty price level:
    price_level.visible_orders = []
    price_level.hidden_orders = []

    with pytest.raises(ValueError):
        price_level.order_has_better_price(order)


def test_order_has_worse_price(price_level):
    # Test orders on same side of book:
    order = LimitOrder(0, 0, "", 10, Side.BID, 90, is_hidden=False)
    assert price_level.order_has_worse_price(order) == True

    order = LimitOrder(0, 0, "", 10, Side.BID, 100, is_hidden=False)
    assert price_level.order_has_worse_price(order) == False

    order = LimitOrder(0, 0, "", 10, Side.BID, 110, is_hidden=False)
    assert price_level.order_has_worse_price(order) == False

    # Test order on opposite side of book:
    order = LimitOrder(0, 0, "", 10, Side.ASK, 100, is_hidden=False)

    with pytest.raises(ValueError):
        price_level.order_has_worse_price(order)

    # Test order with empty price level:
    price_level.visible_orders = []
    price_level.hidden_orders = []

    with pytest.raises(ValueError):
        price_level.order_has_worse_price(order)


def test_order_has_equal_price(price_level):
    # Test orders on same side of book:
    order = LimitOrder(0, 0, "", 10, Side.BID, 90, is_hidden=False)
    assert price_level.order_has_equal_price(order) == False

    order = LimitOrder(0, 0, "", 10, Side.BID, 100, is_hidden=False)
    assert price_level.order_has_equal_price(order) == True

    order = LimitOrder(0, 0, "", 10, Side.BID, 110, is_hidden=False)
    assert price_level.order_has_worse_price(order) == False

    # Test order on opposite side of book:
    order = LimitOrder(0, 0, "", 10, Side.ASK, 100, is_hidden=False)

    with pytest.raises(ValueError):
        price_level.order_has_equal_price(order)

    # Test order with empty price level:
    price_level.visible_orders = []
    price_level.hidden_orders = []

    with pytest.raises(ValueError):
        price_level.order_has_equal_price(order)


def test_total_quantity(price_level):
    assert price_level.total_quantity == 30

    # Test with empty price level:
    price_level.visible_orders = []
    price_level.hidden_orders = []

    assert price_level.total_quantity == 0


def test_is_empty(price_level):
    assert price_level.is_empty == False

    price_level.visible_orders = []
    assert price_level.is_empty == False

    price_level.hidden_orders = []
    assert price_level.is_empty == True


def test_eq(price_level):
    assert price_level == price_level

    lo = LimitOrder(0, 0, "", 10, Side.BID, 90, is_hidden=False)

    assert PriceLevel([(lo, {})]) != price_level
