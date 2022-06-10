import datetime as dt
import logging
from math import exp, sqrt
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CoreOracle:
    def __init__(self, mkt_open, symbols, random_state=None):
        self.mkt_open = mkt_open
        self.symbols = symbols
        self.random_state = random_state if random_state != None else (np.random)

    def get_observation(self, symbol: str, current_time: pd.Timestamp) -> int:
        raise NotImplementedError

    def get_daily_open_price(self, symbol: str, mkt_open: pd.Timestamp) -> float:
        """Return the daily open price for the given symbol."""
        logger.debug(
            "Oracle: client requested {} at market open: {}", symbol, self.mkt_open
        )
        opening_price = self.symbols[symbol]["opening_price"]
        logger.debug("Oracle: market open price was was {}", opening_price)
        return opening_price
