from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from ..price_level import PriceLevel


################## STATE MANIPULATION ###############################
def list_dict_flip(ld: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Utility that returns a dictionnary of list of dictionnary into a dictionary of list

    Arguments:
        - ld: list of dictionaary
    Returns:
        - flipped: dictionnary of lists
    Example:
        - ld = [{"a":1, "b":2}, {"a":3, "b":4}]
        - flipped = {'a': [1, 3], 'b': [2, 4]}
    """
    flipped = dict((k, []) for (k, v) in ld[0].items())
    for rs in ld:
        for k in flipped.keys():
            flipped[k].append(rs[k])
    return flipped


def identity_decorator(func):
    """
    identy for decorators: take a function and return that same function

    Arguments:
        - func: function
    Returns:
        - wrapper_identity_decorator: function
    """

    def wrapper_identity_decorator(*args, **kvargs):
        return func(*args, **kvargs)

    return wrapper_identity_decorator


def ignore_mkt_data_buffer_decorator(func):
    """
    Decorator for function that takes as input self and raw_state.
    Applies the given function while ignoring the buffering in the market data.
    Only last element of the market data buffer is kept
    Arguments:
        - func: function
    Returns:
        - wrapper_mkt_data_buffer_decorator: function
    """

    def wrapper_mkt_data_buffer_decorator(self, raw_state):
        raw_state_copy = deepcopy(raw_state)
        for i in range(len(raw_state)):
            raw_state[i]["parsed_mkt_data"] = raw_state_copy[i]["parsed_mkt_data"][-1]
            raw_state[i]["parsed_volume_data"] = raw_state_copy[i][
                "parsed_volume_data"
            ][-1]
        raw_state2 = list_dict_flip(raw_state)
        flipped = dict((k, list_dict_flip(v)) for (k, v) in raw_state2.items())
        return func(self, flipped)

    return wrapper_mkt_data_buffer_decorator


def ignore_buffers_decorator(func):
    """
    Decorator for function that takes as input self and raw_state.
    Applies the given function while ignoring the buffering in both the market data and the general raw state.
    Only last elements are kept.
    Arguments:
        - func: function
    Returns:
        - wrapper_mkt_data_buffer_decorator: function
    """

    def wrapper_ignore_buffers_decorator(self, raw_state):
        raw_state = raw_state[-1]
        if len(raw_state["parsed_mkt_data"]) == 0:
            pass
        else:
            raw_state["parsed_mkt_data"] = raw_state["parsed_mkt_data"][-1]
            if raw_state["parsed_volume_data"]:
                raw_state["parsed_volume_data"] = raw_state["parsed_volume_data"][-1]
        return func(self, raw_state)

    return wrapper_ignore_buffers_decorator


################# ORDERBOOK PRIMITIVES ######################
def get_mid_price(
    bids: List[PriceLevel], asks: List[PriceLevel], last_transaction: int
) -> int:

    """
    Utility that computes the mid price from the snapshot of bid and ask side

    Arguments:
        - bids: list of list snapshot of bid side
        - asks: list of list snapshot of ask side
        - last_trasaction: last transaction in the market, used for corner cases when one side of the OB is empty
    Returns:
        - mid_price value
    """
    if len(bids) == 0 and len(asks) == 0:
        return last_transaction
    elif len(bids) == 0:
        return asks[0][0]
    elif len(asks) == 0:
        return bids[0][0]
    else:
        return (bids[0][0] + asks[0][0]) / 2


def get_val(book: List[PriceLevel], level: int) -> Tuple[int, int]:
    """
    utility to compute the price and level at the level-th level of the order book

    Arguments:
        - book: side of the order book (bid or ask)
        - level: level of interest in the OB side (index starts at 0 for best bid/ask)

    Returns:
        - tuple price, volume for the i-th value
    """
    if book == []:
        return 0, 0
    else:
        try:
            price = book[level][0]
            volume = book[level][1]
            return price, volume
        except:
            return 0, 0


def get_last_val(book: List[PriceLevel], mid_price: int) -> int:
    """
    utility to compute the price of the deepest placed order in the side of the order book

    Arguments:
        - book: side of the order book (bid or ask)
        - mid_price: current mid price used for corner cases

    Returns:
        - mid price value
    """
    if book == []:
        return mid_price
    else:
        return book[-1][0]


def get_volume(book: List[PriceLevel], depth: Optional[int] = None) -> int:
    """
    utility to compute the volume placed between the top of the book (depth 0) and the depth

    Arguments:
        - book: side of the order book (bid or ask)
        - depth: depth used to compute sum of the volume

    Returns:
        - volume placed
    """
    if depth is None:
        return sum([v[1] for v in book])
    else:
        return sum([v[1] for v in book[:depth]])


def get_imbalance(
    bids: List[PriceLevel],
    asks: List[PriceLevel],
    direction: str = "BUY",
    depth: Optional[int] = None,
) -> float:
    """
    utility to compute the imbalance computed between the top of the book and the depth-th value of depth

    Arguments:
        - bids: list of list snapshot of bid side
        - asks: list of list snapshot of ask side
        - direction: side used to compute the numerator in the division
        - depth: depth used to compute sum of the volume

    Returns:
        - imbalance
    """
    # None corresponds to the whole book depth
    if (bids == []) and (asks == []):
        return 0.5
    elif bids == []:
        if direction == "BUY":
            return 0
        else:
            return 1
    elif asks == []:
        if direction == "BUY":
            return 1
        else:
            return 0
    else:
        if depth == None:
            bid_vol = sum([v[1] for v in bids])
            ask_vol = sum([v[1] for v in asks])
        else:
            bid_vol = sum([v[1] for v in bids[:depth]])
            ask_vol = sum([v[1] for v in asks[:depth]])
    if direction == "BUY":
        return bid_vol / (bid_vol + ask_vol)
    else:
        return ask_vol / (bid_vol + ask_vol)
