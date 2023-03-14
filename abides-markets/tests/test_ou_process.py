import os 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from abides_markets.oracles import SparseMeanRevertingOracle
from abides_core.utils import get_wake_time, str_to_ns


def test_ou_process():
    DATE = int(pd.to_datetime("20210205").to_datetime64())
    MKT_OPEN = DATE + str_to_ns("09:30:00")
    MKT_CLOSE = DATE + str_to_ns("16:00:00")

    # oracle values
    r_bar = 100_000  # true mean fundamental value
    kappa_oracle=1.67e-16  # Mean-reversion of fundamental time series.
    fund_vol=5e-5  # Volatility of fundamental time series (std).
    # megashock properties
    megashock_lambda_a=2.77778e-18 
    megashock_mean=1000
    megashock_var=50_000

    stock_name = "AAPL"
    symbols = {
        stock_name : {
            "r_bar": r_bar,
            "kappa": kappa_oracle,
            "fund_vol": fund_vol,
            "megashock_lambda_a": megashock_lambda_a,
            "megashock_mean": megashock_mean,
            "megashock_var": megashock_var,
        }
    }

    ## compute observation 
    plot_logs = False
    if plot_logs:
        fig, ax = plt.subplots(figsize=(12, 8))

    ntraces = 10
    for seed in range(ntraces):

        ## create random state and ou oracle
        random_state = np.random.RandomState(seed)
        symbols[stock_name]["random_state"] = random_state
        oracle = SparseMeanRevertingOracle(MKT_OPEN, MKT_CLOSE, symbols)
        
        ## Compute the fundamental value
        out_df = {"minute" : [], "price" : []}
        for minute in range(MKT_OPEN, MKT_CLOSE, int(60*1e9)):
            rT = oracle.observe_price(
                stock_name, minute, sigma_n=0, random_state=random_state)

            out_df["price"] += [rT]
            out_df["minute"] += [minute]

        out_df = pd.DataFrame(out_df)
        assert 0.9 < out_df["price"].mean() / r_bar < 1.1, "The generated fundamental value is not mean-reverting"    

        if plot_logs:    
            out_df["price"] /= out_df.iloc[0]["price"]
            out_df["price"].plot(ax=ax)

    if plot_logs:    
        os.makedirs("logs", exist_ok=True)
        plt.savefig(os.path.join("logs", "test_ou_process.png"))

    assert True
