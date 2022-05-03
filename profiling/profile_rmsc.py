# RMSC-3 (Reference Market Simulation Configuration):
# - 1     Exchange Agent
# - 1     POV Market Maker Agent
# - 100   Value Agents
# - 25    Momentum Agents
# - 5000  Noise Agents
# - 1     (Optional) POV Execution agent

import logging

import coloredlogs
import numpy as np
import datetime as dt

from abides_core import Kernel
from abides_core.utils import subdict
from abides_markets.configs.rmsc04 import build_config


logger = logging.getLogger("profile_rmsc")
coloredlogs.install(
    level="INFO", fmt="[%(process)d] %(levelname)s %(name)s %(message)s"
)

# from memory_profiler import profile

# @profile
def run(
    config,
    log_dir="",
    kernel_seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64"),
):

    print()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║ ABIDES: Agent-Based Interactive Discrete Event Simulation ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    kernel = Kernel(
        random_state=np.random.RandomState(seed=kernel_seed),
        log_dir="",
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


if __name__ == "__main__":
    run(build_config(seed=1, book_logging=False, end_time="16:00:00"))

    # import os
    # import subprocess

    # from profilehooks import profile

    # @profile(stdout=False, immediate=True, filename="rmsc03.prof")
    # def _run():
    #     run(build_config(seed=1, book_freq=None, end_time="16:00:00"))

    # _run()

    # subprocess.call(
    #     f"gprof2dot rmsc03.prof -f pstats > rmsc03.dot",
    #     shell=True,
    # )
    # subprocess.call(
    #     f"dot -Tsvg -o rmsc03.svg rmsc03.dot",
    #     shell=True,
    # )

    # os.remove("rmsc03.dot")
    # os.remove("rmsc03.prof")
