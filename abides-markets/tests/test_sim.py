import shutil

import numpy as np

from abides_core import Kernel
from abides_core.utils import subdict
from abides_markets.configs.rmsc04 import build_config as build_config_rmsc04


def test_rmsc04():
    config = build_config_rmsc04(
        seed=1,
        book_logging=False,
        end_time="10:00:00",
        log_orders=False,
        exchange_log_orders=False,
    )

    kernel_seed = np.random.randint(low=0, high=2 ** 32, dtype="uint64")

    kernel = Kernel(
        log_dir="__test_logs",
        random_state=np.random.RandomState(seed=kernel_seed),
        **subdict(
            config,
            [
                "start_time",
                "stop_time",
                "agents",
                "agent_latency_model",
                "default_computation_delay",
                "custom_properties",
            ],
        ),
        skip_log=True,
    )

    kernel.run()

    shutil.rmtree("log/__test_logs")
    ## just checking simulation runs without crashing and reaches the assert
    assert True
