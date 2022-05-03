import os
import subprocess
import shutil

import datetime as dt
import time


def version_greaterorequal(l1, l2):
    if l1[0] > l2[0]:
        return True
    elif l1[0] < l2[0]:
        return False
    elif l1[0] == l2[0]:
        if len(l1) == 1:
            return True
        else:
            return version_greaterorequal(l1[1:], l2[1:])


##this gets the version of git used on the machine (nothing to do with commit sha being used)
def get_git_version():
    result = subprocess.run(["git", "--version"], stdout=subprocess.PIPE).stdout.decode(
        "utf-8"
    )
    version = [
        int(c) for c in result.replace("git version ", "").replace("\n", "").split(".")
    ]
    return version


##run command as of a specified commit code version
##assumes current directory is git root of the main git repo
def run_command(
    command,
    commit_sha,
    specific_path_underscore="0",
    git_path=None,
    pass_logdir_sha=None,
    old_new_flag=None,
):
    """pass_logdir_sha is either null or tuple with arg name and function taking commit sha as input to produce arg value"""

    # first delete existing data in tmp file
    if pass_logdir_sha:
        shutil.rmtree(pass_logdir_sha, ignore_errors=True)

    if commit_sha == "CURRENT":
        if not pass_logdir_sha:
            simulation_start_time = dt.datetime.now()
            os.system(command)
            simulation_end_time = dt.datetime.now()
        else:
            simulation_start_time = dt.datetime.now()
            os.system(
                f"{command} {pass_logdir_sha[0]} {pass_logdir_sha[1](commit_sha)}"
            )
            simulation_end_time = dt.datetime.now()
    else:
        ##check git version to make sure it supports worktrees add and delete
        assert version_greaterorequal(
            get_git_version(), [2, 17]
        ), "git version needs to be >= 2.17"
        ##
        orig_pwd = os.getcwd()
        ##create tmp worktree with desired code version
        path_tmp_worktree = (
            "/".join(git_path.split("/")[:-1])
            + f"/tmp_{old_new_flag}_"
            + commit_sha
            + "_"
            + specific_path_underscore
        )

        subprocess.run(
            ["git", "worktree", "add", "--detach", path_tmp_worktree, commit_sha],
            stdout=subprocess.DEVNULL,
        )
        ##switch pwd to tmp work tree
        os.chdir(path_tmp_worktree)
        ##execute
        if not pass_logdir_sha:
            simulation_start_time = dt.datetime.now()
            os.system(command)
            simulation_end_time = dt.datetime.now()
        else:
            simulation_start_time = dt.datetime.now()
            os.system(
                f"{command} {pass_logdir_sha[0]} {pass_logdir_sha[1](commit_sha)}"
            )
            simulation_end_time = dt.datetime.now()
        ##clean tmp worktree
        subprocess.run(
            ["git", "worktree", "remove", path_tmp_worktree], stdout=subprocess.DEVNULL
        )
        os.chdir(orig_pwd)

    return simulation_end_time - simulation_start_time
