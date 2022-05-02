import os
import pandas as pd
import datetime as dt
import numpy as np
from multiprocessing import Pool
import psutil
import pathlib
from tqdm import tqdm

from p_tqdm import p_map
import itertools


def get_path(level):
    path = pathlib.Path(__file__).parent.absolute()
    path = str(path)
    if level == 0:
        return path
    else:
        path = path.split("/")[:-level]
        return ("/").join(path)


root_path_abides = get_path(1)
root_path_ec2 = get_path(3)

os.chdir(root_path_abides)
import sys

sys.path.insert(0, root_path_abides)

import version_testing.runasof as runasof


# TODO: use different end time in the new config


def get_paths(parameters):
    specific_path = f'{parameters["new"]["config"]}/{parameters["shared"]["end-time"].replace(":", "-")}/{parameters["shared"]["seed"]}'  # can add as many as there are parameters
    specific_path_underscore = f'{parameters["new"]["config"]}_{parameters["shared"]["end-time"].replace(":", "-")}_{parameters["shared"]["seed"]}'  # TODO: maybe something better
    return specific_path, specific_path_underscore


def run_test(test_):
    parameters, old_new_flag = test_
    # run test for one parameter dictionnary
    specific_path, specific_path_underscore = get_paths(parameters)

    # compute a unique stamp for log folder
    now = dt.datetime.now()
    stamp = now.strftime("%Y%m%d%H%M%S")

    # run old sha
    time = runasof.run_command(
        parameters["command"][old_new_flag],
        commit_sha=parameters[old_new_flag]["sha"],
        specific_path_underscore=specific_path_underscore,
        git_path=root_path_abides,
        old_new_flag=old_new_flag,
        pass_logdir_sha=(
            "--log_dir",
            lambda x: root_path_ec2
            + f"/tmp/{old_new_flag}_{stamp}/"
            + x
            + "/"
            + specific_path,
        ),
    )

    # output = parameters

    output = {}

    output["sha"] = parameters[old_new_flag]["sha"]
    output["config"] = parameters[old_new_flag]["config"]
    output["end-time"] = parameters["shared"]["end-time"]
    output["seed"] = parameters["shared"]["seed"]
    output["time"] = time
    ## compare order book logs from the simulations
    if parameters["with_log"]:
        path_to_ob = (
            root_path_ec2
            + f"/tmp/{old_new_flag}_{stamp}/{parameters[old_new_flag]['sha']}/{specific_path}/ORDERBOOK_ABM_FULL.bz2"
        )
    else:
        path_to_ob = "no_log"
    output["path_to_ob"] = path_to_ob
    output["flag"] = old_new_flag

    return output


def compute_ob(path_old, path_new):
    ob_old = pd.read_pickle(path_old)
    ob_new = pd.read_pickle(path_new)
    if ob_old.equals(ob_new):
        return 0
    else:
        return 1


def run_tests(LIST_PARAMETERS, varying_parameters):

    old_new_flags = ["old", "new"]
    tests = list(itertools.product(LIST_PARAMETERS, old_new_flags))

    # test_ = tests[0]
    # run_test(test_)
    outputs = p_map(run_test, tests)

    df = pd.DataFrame(outputs)

    df_old = df[df["flag"] == "old"]
    df_new = df[df["flag"] == "new"]

    print(f"THERE ARE {len(df_new)} TESTS RESULTS.")

    if LIST_PARAMETERS[0]["with_log"]:
        path_olds = list(df_old["path_to_ob"])
        path_news = list(df_new["path_to_ob"])

        # compute_ob(path_olds[0], path_news[0])

        ob_comps = p_map(compute_ob, path_olds, path_news)

        if sum(ob_comps) == 0:
            print("ALL TESTS ARE SUCCESS!")
        else:
            print(f"ALERT: {sum(ob_comps)}TEST FAILURE")
    df_old = df_old[varying_parameters + ["seed", "time"]].set_index(
        varying_parameters + ["seed"]
    )
    df_new = df_new[varying_parameters + ["seed", "time"]].set_index(
        varying_parameters + ["seed"]
    )
    df_diff = df_old - df_new  # /df_old
    df_results = df_diff.groupby(["config", "end-time"])["time"].describe()[
        ["mean", "std"]
    ]

    df_diff_pct = 100 * (df_old - df_new) / df_old
    df_results_pct = df_diff_pct.groupby(["config", "end-time"])["time"].describe()[
        ["mean", "std"]
    ]
    print("*********************************************")
    print("*********************************************")
    print("OLD RUNNING TIME")
    # with pd.option_context('display.float_format', '{:0.2f}'.format):
    print(df_old.groupby(["config", "end-time"])["time"].describe()[["mean", "std"]])
    print("*********************************************")
    print("*********************************************")
    print("NEW RUNNING TIME")
    with pd.option_context("display.float_format", "{:0.2f}".format):
        print(
            df_new.groupby(["config", "end-time"])["time"].describe()[["mean", "std"]]
        )
    print("*********************************************")
    print("*********************************************")
    print("TIME DIFFERENCE in seconds")
    with pd.option_context("display.float_format", "{:0.2f}".format):
        df_results["mean"] = df_results["mean"].dt.total_seconds()
        df_results["std"] = df_results["std"].dt.total_seconds()
    print(df_results)
    print("*********************************************")
    print("*********************************************")
    print("TIME DIFFERENCE in %")
    with pd.option_context("display.float_format", "{:0.2f}".format):
        print(df_results_pct)
