import numpy as np

from abides_markets.generators import (
    ConstantDepthGenerator,
    ConstantOrderSizeGenerator,
    UniformDepthGenerator,
    UniformOrderSizeGenerator,
)


def test_constant_depth_generator():
    g = ConstantDepthGenerator(10)

    assert g.next() == 10
    assert g.mean() == 10


def test_constant_order_size_generator():
    g = ConstantOrderSizeGenerator(10)

    assert g.next() == 10
    assert g.mean() == 10


def test_uniform_depth_generator():
    g = UniformDepthGenerator(0, 10, np.random.RandomState())

    assert g.mean() == 5


def test_uniform_order_size_generator():
    g = UniformOrderSizeGenerator(0, 10, np.random.RandomState())

    assert g.mean() == 5
