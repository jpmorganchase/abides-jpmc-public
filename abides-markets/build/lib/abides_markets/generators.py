from abc import abstractmethod, ABC

import numpy as np
from abides_core.generators import BaseGenerator


################## ORDER SIZE MODEL ###############################
class OrderSizeGenerator(BaseGenerator[int], ABC):
    pass


class ConstantOrderSizeGenerator(OrderSizeGenerator):
    def __init__(self, order_size: int) -> None:
        self.order_size: int = order_size

    def next(self) -> int:
        return self.order_size

    def mean(self) -> int:
        return self.order_size


class UniformOrderSizeGenerator(OrderSizeGenerator):
    def __init__(
        self,
        order_size_min: int,
        order_size_max: int,
        random_generator: np.random.RandomState,
    ) -> None:
        self.order_size_min: int = order_size_min
        self.order_size_max: int = order_size_max + 1
        self.random_generator: np.random.RandomState = random_generator

    def next(self) -> int:
        return self.random_generator.randint(self.order_size_min, self.order_size_max)

    def mean(self) -> float:
        return (self.order_size_max - self.order_size_min - 1) / 2


################## ORDER DEPTH MODEL ###############################
class OrderDepthGenerator(BaseGenerator[int], ABC):
    pass


class ConstantDepthGenerator(OrderDepthGenerator):
    def __init__(self, order_depth: int) -> None:
        self.order_depth: int = order_depth

    def next(self) -> int:
        return self.order_depth

    def mean(self) -> int:
        return self.order_depth


class UniformDepthGenerator(OrderDepthGenerator):
    def __init__(
        self,
        order_depth_min: int,
        order_depth_max: int,
        random_generator: np.random.RandomState,
    ) -> None:
        self.random_generator: np.random.RandomState = random_generator
        self.order_depth_min: int = order_depth_min
        self.order_depth_max: int = order_depth_max + 1

    def next(self) -> int:
        return self.random_generator.randint(self.order_depth_min, self.order_depth_max)

    def mean(self) -> float:
        return (self.order_depth_max - self.order_depth_min - 1) / 2
