from typing import List, Optional, Union

import numpy as np

from abides_core import Agent

from ..utils import dollarize


class FinancialAgent(Agent):
    """
    The FinancialAgent class contains attributes and methods that should be available to
    all agent types (traders, exchanges, etc) in a financial market simulation.

    To be honest, it mainly exists because the base Agent class should not have any
    finance-specific aspects and it doesn't make sense for ExchangeAgent to inherit from
    TradingAgent. Hopefully we'll find more common ground for traders and exchanges to
    make this more useful later on.
    """

    def __init__(
        self,
        id: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        # Base class init.
        super().__init__(id, name, type, random_state)

    def dollarize(self, cents: Union[List[int], int]) -> Union[List[str], str]:
        """
        Used by any subclass to dollarize an int-cents price for printing.
        """
        return dollarize(cents)
