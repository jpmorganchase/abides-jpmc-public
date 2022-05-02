import psutil
import pathlib
import sys

import pandas as pd


def get_path(level):
    path = pathlib.Path(__file__).parent.absolute()
    path = str(path)
    if level == 0:
        return path
    else:
        path = path.split("/")[:-level]
        return ("/").join(path)


root_path_abides = get_path(1)
sys.path.insert(0, root_path_abides)
import version_testing.test_config as test_config


def generate_parameter_dict(
    seed, config, end_time, with_log
):  # can add for varying parameters

    if with_log:
        log_orders = True
        exchange_log_orders = True
        book_freq = 0
    else:
        log_orders = None
        exchange_log_orders = None
        book_freq = None

    parameters = {
        "old": {
            "sha": "f1968a56fdb55fd7c70be1db052be07cb701a5fb",
            "script": "abides_cmd.py",
            "config": config,
        },
        "new": {
            "sha": "f1968a56fdb55fd7c70be1db052be07cb701a5fb",  # CURRENT
            "script": "abides_cmd.py",
            "config": config,
        },
        "config_new": config,  # little hack for the analysis #TODO: find a better solution
        "end-time": end_time,  # little hack for the analysis #TODO: find a better solution
        "with_log": with_log,
        "shared": {
            "end-time": end_time,
            "end_time": end_time,
            "seed": seed,
            "verbose": 0,
            "log_orders": log_orders,
            "exchange_log_orders": exchange_log_orders,
            "book_freq": book_freq,
        },
    }

    parameters["command"] = generate_command(parameters)
    return parameters


def generate_command(parameters):

    specific_command_old = (
        f"{parameters['old']['script']} -config {parameters['old']['config']}"
    )
    specific_command_new = (
        f"{parameters['new']['script']} -config {parameters['new']['config']}"
    )

    shared_command = [f"--{key} {val}" for key, val in parameters["shared"].items()]
    shared_command = " ".join(shared_command)
    command_old = f"python3 -W ignore -u " + specific_command_old + " " + shared_command
    command_new = f"python3 -W ignore -u " + specific_command_new + " " + shared_command
    # f"python3 -u {parameter_dict['script']} -c {parameter_dict['config_old']} -t ABM -d 20200603 --end-time {parameters['end-time']}:00:00 -s {parameters['seed']} "
    return {"old": command_old, "new": command_new}


if __name__ == "__main__":

    with_log = (
        False  # if no log, then there is no checking of OB - only measure the timing
    )
    configs = ["rmsc04", "rmsc03"]  # , 'rmsc04_function']'rmsc03_aymeric',
    end_times = ["10:00:00", "12:00:00", "16:00:00"]  #'11, '12:00:00', '16:00:00'

    LIST_PARAMETERS = [
        generate_parameter_dict(seed, config, end_time, with_log)
        for seed in range(1, 41)
        for config in configs
        for end_time in end_times
    ]
    assert len(LIST_PARAMETERS) > 0, "Enter at least one parameters dictionary"

    # test_config.run_test(LIST_PARAMETERS[0])
    # result_list = test_config.run_imap_unordered_multiprocessing(func=test_config.run_test, argument_list=LIST_PARAMETERS)
    varying_parameters = ["config", "end-time"]  #'end-time'
    test_config.run_tests(LIST_PARAMETERS, varying_parameters)
