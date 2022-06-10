import datetime
import sys
import traceback
import warnings
from contextlib import contextmanager
from typing import List, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from abides_core import LatencyModel

# Utility method to flatten nested lists.
def delist(list_of_lists):
    return [x for b in list_of_lists for x in b]


def numeric(s):
    """Returns numeric type from string, stripping commas from the right.

    Adapted from https://stackoverflow.com/a/379966.
    """

    s = s.rstrip(",")
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def get_value_from_timestamp(s: pd.Series, ts: datetime.datetime):
    """Get the value of s corresponding to closest datetime to ts.

    Arguments:
        s: Pandas Series with pd.DatetimeIndex.
        ts: Timestamp at which to retrieve data.
    """

    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
    s = s.loc[~s.index.duplicated(keep="last")]
    locs = s.index.get_loc(ts_str, method="nearest")
    out = (
        s[locs][0]
        if (isinstance(s[locs], np.ndarray) or isinstance(s[locs], pd.Series))
        else s[locs]
    )

    return out


@contextmanager
def ignored(warning_str, *exceptions):
    """Context manager that wraps the code block in a try except statement, catching
    specified exceptions and printing warning supplied by user.

    Arguments:
        warning_str: Warning statement printed when exception encountered.
        exceptions: An exception type, e.g. ``ValueError``.

        https://stackoverflow.com/a/15573313
    """

    try:
        yield
    except exceptions:
        warnings.warn(warning_str, UserWarning, stacklevel=1)
        print(warning_str)


def generate_uniform_random_pairwise_dist_on_line(
    left: float, right: float, num_points: int, random_state: np.random.RandomState
) -> np.ndarray:
    """Uniformly generate points on an interval, and return numpy array of pairwise
    distances between points.

    Arguments:
        left: Left endpoint of interval.
        right: Right endpoint of interval.
        num_points: Number of points to use.
        random_state: ``np.random.RandomState`` object.
    """

    x_coords = random_state.uniform(low=left, high=right, size=num_points)
    x_coords = x_coords.reshape((x_coords.size, 1))
    out = pdist(x_coords, "euclidean")
    return squareform(out)


def meters_to_light_ns(x):
    """Converts x in units of meters to light nanoseconds."""

    x_lns = x / 299792458e-9
    x_lns = x_lns.astype(int)
    return x_lns


def validate_window_size(s):
    """Check if s is integer or string 'adaptive'."""

    try:
        return int(s)
    except ValueError:
        if s.lower() == "adaptive":
            return s.lower()
        else:
            raise ValueError(f'String {s} must be integer or string "adaptive".')


def sigmoid(x, beta):
    """Numerically stable sigmoid function.

    Adapted from https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/"
    """

    if x >= 0:
        z = np.exp(-beta * x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(beta * x)
        return z / (1 + z)


def subdict(d, keys):
    return dict((k, v) for k, v in d.items() if k in keys)


def restrictdict(d, keys):
    inter = [k for k in d.keys() if k in keys]
    return subdict(d, inter)


def dollarize(cents: Union[List[int], int]) -> Union[List[str], str]:
    """Dollarizes int-cents prices for printing.

    Defined outside the class for utility access by non-agent classes.

    Arguments:
      cents:
    """

    if isinstance(cents, list):
        return [dollarize(x) for x in cents]
    elif isinstance(cents, (int, np.int64)):
        return "${:0.2f}".format(cents / 100)
    else:
        # If cents is already a float, there is an error somewhere.
        raise ValueError(
            f"dollarize(cents) called without int or list of ints: {cents} (got type '{type(cents)}')"
        )


# LATENCY
def generate_latency_model(agent_count, latency_type="deterministic"):
    assert latency_type in [
        "deterministic",
        "no_latency",
    ], "Please select a correct latency_type"

    latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))
    pairwise = (agent_count, agent_count)

    if latency_type == "deterministic":
        # All agents sit on line from Seattle to NYC
        nyc_to_seattle_meters = 3866660
        pairwise_distances = generate_uniform_random_pairwise_dist_on_line(
            0.0, nyc_to_seattle_meters, agent_count, random_state=latency_rstate
        )
        pairwise_latencies = meters_to_light_ns(pairwise_distances)

    else:  # latency_type == "no_latency"
        pairwise_latencies = np.zeros(pairwise, dtype=int)

    latency_model = LatencyModel(
        latency_model="deterministic",
        random_state=latency_rstate,
        connected=True,
        min_latency=pairwise_latencies,
    )

    return latency_model


def config_add_agents(orig_config_state, agents):
    agent_count = len(orig_config_state["agents"])
    orig_config_state["agents"] = orig_config_state["agents"] + agents
    # adding an agent to the config implies modifying the latency model #TODO: tell aymeric
    lat_mod = generate_latency_model(agent_count + len(agents))
    orig_config_state["agent_latency_model"] = lat_mod
    return orig_config_state
