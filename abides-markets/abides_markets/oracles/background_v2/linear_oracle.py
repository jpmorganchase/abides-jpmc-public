import datetime as dt
import logging
from math import exp, sqrt
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .core_oracle import CoreOracle
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OracleParameters:
    # polynomial parameters
    l_0: int = 0  # constant
    l_1: float = 0
    l_2: float = 0
    l_3: float = 0
    # sinusoidal parameters
    sin_amp: float = 0
    sin_freq: float = 0
    # Noise parameters
    sigma: float = 0
    #
    cumulative_noise: bool = False


class LinearOracle(CoreOracle):
    def __init__(
        self, mkt_open, symbols, symbol, oracle_parameters=None, random_state=None
    ):
        super().__init__(mkt_open, symbols, random_state)
        self.opening_price = self.get_daily_open_price(symbol, mkt_open)
        self.oracle_parameters = oracle_parameters
        self.random_state = random_state
        self.reference_time = self.mkt_open
        self.reference_price = self.opening_price
        print(self.oracle_parameters)

    def observe_price(self, symbol: str, current_time: pd.Timestamp, random_state: None,
        sigma_n: None) -> int:
        time_distance_ns = current_time - self.reference_time
        time_distance_h = time_distance_ns / (60 * 60 * 1e9)  # convertion ns -> h

        # compute observation
        observation = self.reference_price
        observation += self.oracle_parameters.l_0
        observation += self.oracle_parameters.l_1 * time_distance_h
        observation += self.oracle_parameters.l_2 * (time_distance_h ** 2)
        observation += self.oracle_parameters.l_3 * (time_distance_h ** 3)
        observation += self.oracle_parameters.sin_amp * np.sin(
            time_distance_h ** 2 * np.pi * self.oracle_parameters.sin_freq
        )
        # random term
        if self.oracle_parameters.cumulative_noise:
            scale = self.oracle_parameters.sigma * time_distance_h
        else:
            scale = self.oracle_parameters.sigma
        observation += self.random_state.normal(scale=scale)

        if self.oracle_parameters.cumulative_noise:
            self.reference_time = current_time
            self.reference_price = observation
        return observation
