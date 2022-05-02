from abides_markets.orders import MarketOrder, Side

from . import setup_book_with_orders, SYMBOL, TIME

# fmt: off


def test_get_l1_bid_ask_data():
    book, _, _ = setup_book_with_orders(bids=[], asks=[])

    assert book.get_l1_bid_data() == None
    assert book.get_l1_ask_data() == None

    book, _, _ = setup_book_with_orders(
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

    assert book.get_l1_bid_data() == (200, 70)
    assert book.get_l1_ask_data() == (300, 80)


def test_get_l2_bid_ask_data():
    book, _, _ = setup_book_with_orders(bids=[], asks=[])

    assert book.get_l2_bid_data() == []
    assert book.get_l2_ask_data() == []

    book, _, _ = setup_book_with_orders(
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

    assert book.get_l2_bid_data() == [
        (200, 70),
        (100, 50),
    ]

    assert book.get_l2_ask_data() == [
        (300, 80),
        (400, 50),
        (500, 20),
    ]


def test_get_l3_bid_ask_data():
    book, _, _ = setup_book_with_orders(bids=[], asks=[])

    assert book.get_l3_bid_data() == []
    assert book.get_l3_ask_data() == []

    book, _, _ = setup_book_with_orders(
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

    assert book.get_l3_bid_data() == [
        (200, [10, 30, 20, 10]),
        (100, [40, 10]),
    ]

    assert book.get_l3_ask_data() == [
        (300, [10, 50, 20]),
        (400, [40, 10]),
        (500, [20]),
    ]


def test_get_transacted_volume():
    book, _, _ = setup_book_with_orders(
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

    for q in [10, 30, 20, 10]:
        order = MarketOrder(
            agent_id=1,
            time_placed=TIME,
            symbol=SYMBOL,
            quantity=q,
            side=Side.BID,
        )

        book.handle_market_order(order)

    assert book.get_transacted_volume() == (sum([10, 30, 20, 10]), 0)

    for q in [50, 10, 40]:
        order = MarketOrder(
            agent_id=1,
            time_placed=TIME,
            symbol=SYMBOL,
            quantity=q,
            side=Side.ASK,
        )

        book.handle_market_order(order)

    assert book.get_transacted_volume() == (sum([10, 30, 20, 10]), sum([50, 10, 40]))


def test_get_imbalance():
    book, _, _ = setup_book_with_orders(
        bids=[(100, [10])],
        asks=[(200, [10])],
    )

    assert book.get_imbalance() == (0, None)

    book, _, _ = setup_book_with_orders(
        bids=[(100, [20])],
        asks=[(200, [10])],
    )

    assert book.get_imbalance() == (0.5, Side.BID)

    book, _, _ = setup_book_with_orders(
        bids=[(100, [10])],
        asks=[(200, [20])],
    )

    assert book.get_imbalance() == (0.5, Side.ASK)

    book, _, _ = setup_book_with_orders(
        bids=[(100, [100])],
        asks=[(200, [10])],
    )

    assert book.get_imbalance() == (0.9, Side.BID)

    book, _, _ = setup_book_with_orders(
        bids=[(100, [10])],
        asks=[(200, [100])],
    )

    assert book.get_imbalance() == (0.9, Side.ASK)

    book, _, _ = setup_book_with_orders(
        bids=[(100, [20])],
        asks=[],
    )

    assert book.get_imbalance() == (1.0, Side.BID)

    book, _, _ = setup_book_with_orders(
        bids=[],
        asks=[(200, [20])],
    )

    assert book.get_imbalance() == (1.0, Side.ASK)
