from abc import abstractmethod, ABC
from typing import Generic, Optional, TypeVar

import numpy as np


T = TypeVar("T")


class BaseGenerator(ABC, Generic[T]):
    """
    This is an abstract base class defining the interface for Generator objects in
    ABIDES. This class is not used directly and is instead inherited from child classes.

    Generators should produce an infinite amount of values.
    """

    @abstractmethod
    def next(self) -> T:
        """
        Produces the next value from the generator.
        """
        raise NotImplementedError

    @abstractmethod
    def mean(self) -> T:
        """
        Returns the average of the distribution of values generated.
        """
        raise NotImplementedError


class InterArrivalTimeGenerator(BaseGenerator[float], ABC):
    """
    General class for time generation. These generators are used to generates a delta time between currrent time and the next wakeup of the agent.
    """

    pass


class ConstantTimeGenerator(InterArrivalTimeGenerator):
    """
    Generates constant delta time of length step_duration

    Arguments:
        step_duration: length of the delta time in ns
    """

    def __init__(self, step_duration: float) -> None:
        self.step_duration: float = step_duration

    def next(self) -> float:
        """
        returns constant time delta for next wakeup
        """
        return self.step_duration

    def mean(self) -> float:
        """
        time delta is constant
        """
        return self.step_duration


class PoissonTimeGenerator(InterArrivalTimeGenerator):
    """
    Lambda must be specified either in second through lambda_time or seconds^-1
    through lambda_freq.

    Arguments:
        random_generator: configuration random generator
        lambda_freq: frequency (in s^-1)
        lambda_time: period (in seconds)
    """

    def __init__(
        self,
        random_generator: np.random.RandomState,
        lambda_freq: Optional[float] = None,
        lambda_time: Optional[float] = None,
    ) -> None:
        self.random_generator: np.random.RandomState = random_generator

        assert (lambda_freq is None and lambda_time is not None) or (
            lambda_time is None and lambda_freq is not None
        ), "specify lambda in frequency OR in time"

        self.lambda_s: float = lambda_freq or 1 / lambda_time

    def next(self) -> Optional[float]:
        """
        returns time delta for next wakeup with time delta following Poisson distribution
        """
        seconds = self.random_generator.exponential(1 / self.lambda_s)
        return seconds * 1_000_000_000 if seconds is not None else None

    def mean(self) -> float:
        """
        returns the mean of a Poisson(lambda) distribution (i.e., 1/lambda)
        """
        return 1 / self.lambda_s
