import numpy as np

from abides_core.generators import PoissonTimeGenerator


def test_poisson_time_generator():
    gen = PoissonTimeGenerator(
        lambda_time=2, random_generator=np.random.RandomState(seed=1)
    )

    for _ in range(10):
        print(gen.next())
