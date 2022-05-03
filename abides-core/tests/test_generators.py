import numpy as np

from abides_core.generators import ConstantTimeGenerator, PoissonTimeGenerator


def test_constant_time_generator():
    g = ConstantTimeGenerator(10)

    assert g.next() == 10
    assert g.mean() == 10


def test_poisson_time_generator():
    g = PoissonTimeGenerator(np.random.RandomState(), lambda_freq=10)

    assert g.mean() == 0.1

    assert abs(np.mean([g.next() for _ in range(100000)]) / 1e9 - 0.1) < 0.001

    g = PoissonTimeGenerator(np.random.RandomState(), lambda_time=10)

    assert g.mean() == 10

    assert abs(np.mean([g.next() for _ in range(100000)]) / 1e9 - 10) < 0.1
