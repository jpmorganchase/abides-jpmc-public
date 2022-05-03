import psutil
import pathlib

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
import sys

sys.path.insert(0, root_path_abides)
import version_testing.test_config as test_config


def generate_parameter_dict(seed):  # can add for varying parameters
    parameters = {
        "sha_old": "8ab374e8d7c9f6fa6ab522502259e94e550e81b5",
        "sha_new": "ccdb7b3b0b099b89b86a6500e4f8f731a5dc6410",
        "script_old": "abides.py",
        "script_new": "abides_cmd.py",
        "config_old": "rmsc03",
        "config_new": "rmsc03_function",
        "end-time": "10",
        "seed": seed,
    }
    return parameters


if __name__ == "__main__":
    LIST_PARAMETERS = [generate_parameter_dict(seed) for seed in range(1, 3)]
    num_processes = (
        len(LIST_PARAMETERS)
        if len(LIST_PARAMETERS) < psutil.cpu_count()
        else psutil.cpu_count()
    )

    print(f"Total Number of Black-Box Tests: {num_processes}")

    func = test_config.run_test
    argument_list = LIST_PARAMETERS
    result_list = test_config.run_imap_unordered_multiprocessing(
        func=func, argument_list=argument_list, num_processes=num_processes
    )

    df_results = pd.DataFrame(result_list)

    test_config.analyse_results(df_results)
