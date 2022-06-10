import datetime as dt
import logging
from typing import Any, Dict, Optional

import coloredlogs
import numpy as np

from .kernel import Kernel
from .utils import subdict


logger = logging.getLogger("abides")


def run(
    config: Dict[str, Any],
    log_dir: str = "",
    kernel_seed: int = 0,
    kernel_random_state: Optional[np.random.RandomState] = None,
) -> Dict[str, Any]:
    """
    Wrapper function that enables to run one simulation.
    It does the following steps:
    - instantiation of the kernel
    - running of the simulation
    - return the end_state object

    Arguments:
        config: configuration file for the specific simulation
        log_dir: directory where log files are stored
        kernel_seed: simulation seed
        kernel_random_state: simulation random state
    """
    coloredlogs.install(
        level=config["stdout_log_level"],
        fmt="[%(process)d] %(levelname)s %(name)s %(message)s",
    )

    kernel = Kernel(
        random_state=kernel_random_state or np.random.RandomState(seed=kernel_seed),
        log_dir=log_dir,
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
    )

    sim_start_time = dt.datetime.now()

    logger.info(f"Simulation Start Time: {sim_start_time}")

    end_state = kernel.run()

    sim_end_time = dt.datetime.now()
    logger.info(f"Simulation End Time: {sim_end_time}")
    logger.info(f"Time taken to run simulation: {sim_end_time - sim_start_time}")

    return end_state
