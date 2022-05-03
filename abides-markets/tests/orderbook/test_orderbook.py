from abides_markets.order_book import OrderBook

from . import FakeExchangeAgent, SYMBOL

# fmt: off


def test_empty_book():
    book = OrderBook(FakeExchangeAgent(), SYMBOL)

    assert book.get_l1_bid_data() == None
    assert book.get_l1_ask_data() == None
    assert book.get_l2_bid_data() == []
    assert book.get_l2_ask_data() == []
    assert book.get_l3_bid_data() == []
    assert book.get_l3_ask_data() == []
    assert book.get_transacted_volume() == (0, 0)
