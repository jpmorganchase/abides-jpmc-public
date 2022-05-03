from abides_markets.orders import Order


def reset_env():
    Order._order_id_counter = 0
