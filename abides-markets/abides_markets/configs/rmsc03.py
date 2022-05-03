# RMSC-3 (Reference Market Simulation Configuration):
# - 1     Exchange Agent
# - 2     Adaptive Market Maker Agents
# - 100   Value Agents
# - 25    Momentum Agents
# - 5000  Noise Agents

import numpy as np

from abides_core.utils import str_to_ns, datetime_str_to_ns, get_wake_time
from abides_markets.agents import (
    ExchangeAgent,
    NoiseAgent,
    ValueAgent,
    AdaptiveMarketMakerAgent,
    MomentumAgent,
    POVExecutionAgent,
)
from abides_markets.oracles import SparseMeanRevertingOracle
from abides_markets.orders import Side
from abides_markets.utils import generate_latency_model


########################################################################################################################
############################################### GENERAL CONFIG #########################################################


def build_config(
    ticker="ABM",
    historical_date="20200603",
    start_time="09:30:00",
    end_time="16:00:00",
    exchange_log_orders=True,
    log_orders=True,
    book_logging=True,
    book_log_depth=10,
    #   seed=int(NanosecondTime.now().timestamp() * 1000000) % (2 ** 32 - 1),
    seed=1,
    stdout_log_level="INFO",
    ##
    num_momentum_agents=25,
    num_noise_agents=5000,
    num_value_agents=100,
    ## exec agent
    execution_agents=True,
    execution_pov=0.1,
    ## market maker
    mm_pov=0.025,
    mm_window_size="adaptive",
    mm_min_order_size=1,
    mm_num_ticks=10,
    mm_wake_up_freq=str_to_ns("10S"),
    mm_skew_beta=0,
    mm_level_spacing=5,
    mm_spread_alpha=0.75,
    mm_backstop_quantity=50_000,
    ##fundamental/oracle
    fund_r_bar=100_000,
    fund_kappa=1.67e-16,
    fund_sigma_s=0,
    fund_vol=1e-8,
    fund_megashock_lambda_a=2.77778e-18,
    fund_megashock_mean=1000,
    fund_megashock_var=50_000,
    ##value agent
    val_r_bar=100_000,
    val_kappa=1.67e-15,
    val_vol=1e-8,
    val_lambda_a=7e-11,
):

    fund_sigma_n = fund_r_bar / 10
    val_sigma_n = val_r_bar / 10
    symbol = ticker

    ##setting numpy seed
    np.random.seed(seed)

    ########################################################################################################################
    ############################################### AGENTS CONFIG ##########################################################

    # Historical date to simulate.
    historical_date = datetime_str_to_ns(historical_date)
    mkt_open = historical_date + str_to_ns(start_time)
    mkt_close = historical_date + str_to_ns(end_time)
    agent_count, agents, agent_types = 0, [], []

    # Hyperparameters
    starting_cash = 10000000  # Cash in this simulator is always in CENTS.

    # Oracle
    symbols = {
        symbol: {
            "r_bar": fund_r_bar,
            "kappa": fund_kappa,
            "sigma_s": fund_sigma_s,
            "fund_vol": fund_vol,
            "megashock_lambda_a": fund_megashock_lambda_a,
            "megashock_mean": fund_megashock_mean,
            "megashock_var": fund_megashock_var,
            "random_state": np.random.RandomState(
                seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")
            ),
        }
    }

    oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

    # 1) Exchange Agent

    #  How many orders in the past to store for transacted volume computation
    agents.extend(
        [
            ExchangeAgent(
                id=0,
                name="EXCHANGE_AGENT",
                mkt_open=mkt_open,
                mkt_close=mkt_close,
                symbols=[symbol],
                book_logging=book_logging,
                book_log_depth=book_log_depth,
                log_orders=exchange_log_orders,
                pipeline_delay=0,
                computation_delay=0,
                stream_history=25_000,
            )
        ]
    )
    agent_types.extend("ExchangeAgent")
    agent_count += 1

    # 2) Noise Agents
    num_noise = num_noise_agents
    noise_mkt_open = historical_date + str_to_ns("09:00:00")
    noise_mkt_close = historical_date + str_to_ns("16:00:00")
    agents.extend(
        [
            NoiseAgent(
                id=j,
                symbol=symbol,
                starting_cash=starting_cash,
                wakeup_time=get_wake_time(noise_mkt_open, noise_mkt_close),
                log_orders=log_orders,
            )
            for j in range(agent_count, agent_count + num_noise)
        ]
    )
    agent_count += num_noise
    agent_types.extend(["NoiseAgent"])

    # 3) Value Agents
    num_value = num_value_agents
    agents.extend(
        [
            ValueAgent(
                id=j,
                name="Value Agent {}".format(j),
                symbol=symbol,
                starting_cash=starting_cash,
                sigma_n=val_sigma_n,
                r_bar=val_r_bar,
                kappa=val_kappa,
                lambda_a=val_lambda_a,
                log_orders=log_orders,
            )
            for j in range(agent_count, agent_count + num_value)
        ]
    )
    agent_count += num_value
    agent_types.extend(["ValueAgent"])

    # 4) Market Maker Agents

    """
    window_size ==  Spread of market maker (in ticks) around the mid price
    pov == Percentage of transacted volume seen in previous `mm_wake_up_freq` that
           the market maker places at each level
    num_ticks == Number of levels to place orders in around the spread
    wake_up_freq == How often the market maker wakes up
    
    """

    # each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)
    mm_params = 2 * [
        (mm_window_size, mm_pov, mm_num_ticks, mm_wake_up_freq, mm_min_order_size)
    ]

    num_mm_agents = len(mm_params)
    mm_cancel_limit_delay = 50  # 50 nanoseconds

    agents.extend(
        [
            AdaptiveMarketMakerAgent(
                id=j,
                name="ADAPTIVE_POV_MARKET_MAKER_AGENT_{}".format(j),
                type="AdaptivePOVMarketMakerAgent",
                symbol=symbol,
                starting_cash=starting_cash,
                pov=mm_params[idx][1],
                min_order_size=mm_params[idx][4],
                window_size=mm_params[idx][0],
                num_ticks=mm_params[idx][2],
                wake_up_freq=mm_params[idx][3],
                cancel_limit_delay=mm_cancel_limit_delay,
                skew_beta=mm_skew_beta,
                level_spacing=mm_level_spacing,
                spread_alpha=mm_spread_alpha,
                backstop_quantity=mm_backstop_quantity,
                log_orders=log_orders,
            )
            for idx, j in enumerate(range(agent_count, agent_count + num_mm_agents))
        ]
    )
    agent_count += num_mm_agents
    agent_types.extend("POVMarketMakerAgent")

    # 5) Momentum Agents
    num_momentum_agents = num_momentum_agents

    agents.extend(
        [
            MomentumAgent(
                id=j,
                name="MOMENTUM_AGENT_{}".format(j),
                symbol=symbol,
                starting_cash=starting_cash,
                min_size=1,
                max_size=10,
                wake_up_freq=str_to_ns("20s"),
                log_orders=log_orders,
            )
            for j in range(agent_count, agent_count + num_momentum_agents)
        ]
    )
    agent_count += num_momentum_agents
    agent_types.extend("MomentumAgent")

    # 6) Execution Agent

    trade = True if execution_agents else False

    #### Participation of Volume Agent parameters

    pov_agent_start_time = mkt_open + str_to_ns("00:30:00")
    pov_agent_end_time = mkt_close - str_to_ns("00:30:00")
    pov_proportion_of_volume = execution_pov
    pov_quantity = 12e5
    pov_frequency = str_to_ns("1min")
    pov_direction = Side.BID

    pov_agent = POVExecutionAgent(
        id=agent_count,
        name="POV_EXECUTION_AGENT",
        type="ExecutionAgent",
        symbol=symbol,
        starting_cash=starting_cash,
        start_time=pov_agent_start_time,
        end_time=pov_agent_end_time,
        freq=pov_frequency,
        lookback_period=pov_frequency,
        pov=pov_proportion_of_volume,
        direction=pov_direction,
        quantity=pov_quantity,
        trade=trade,
        log_orders=True,  # needed for plots so conflicts with others
    )

    execution_agents = [pov_agent]
    agents.extend(execution_agents)
    agent_types.extend("ExecutionAgent")
    agent_count += 1

    # extract kernel seed here to reproduce the state of random generator in old version
    random_state_kernel = np.random.RandomState(
        seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")
    )
    # LATENCY

    latency_model = generate_latency_model(agent_count)
    default_computation_delay = 50  # 50 nanoseconds

    ##kernel args
    kernelStartTime = historical_date
    kernelStopTime = mkt_close + str_to_ns("00:01:00")

    return {
        "start_time": kernelStartTime,
        "stop_time": kernelStopTime,
        "agents": agents,
        "agent_latency_model": latency_model,
        "default_computation_delay": default_computation_delay,
        "custom_properties": {"oracle": oracle},
        "random_state_kernel": random_state_kernel,
        "stdout_log_level": stdout_log_level,
    }
