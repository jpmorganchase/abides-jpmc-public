import os
import pandas as pd

os.chdir("/home/ec2-user/project/abides_dev")
import sys

sys.path.insert(0, "/home/ec2-user/project/abides_dev")
import version_testing.runasof as runasof

## run rmsc03 config with ABIDES old version
sha_old = "8ab374e8d7c9f6fa6ab522502259e94e550e81b5"
runasof.run_command(
    "python3 -u abides.py -c rmsc03 -t ABM -d 20200603 --end-time 10:00:00 -s 1 ",
    commit_sha=sha_old,
    git_path="/home/ec2-user/project/abides_dev",
    pass_logdir_sha=("--log_dir", lambda x: "/home/ec2-user/tmp/" + x),
)

##run rmsc03 with aymericselim_dev
sha_new = "d9010d855f02678ab06d05f15ad4b9db2b93e5e6"  # "2e9aa10f51d2fafd9fe30fd537a7a157c27785d9"
runasof.run_command(
    "python3 abides_cmd.py -c rmsc03_aymeric -s 1 ",
    commit_sha=sha_new,
    git_path="/home/ec2-user/project/abides_dev",
    pass_logdir_sha=("--log_dir", lambda x: "/home/ec2-user/tmp/" + x),
)

## compare order book logs from the simulations
ob_old = pd.read_pickle(f"/home/ec2-user/tmp/{sha_old}/ORDERBOOK_ABM_FULL.bz2")
ob_new = pd.read_pickle(f"/home/ec2-user/tmp/{sha_new}/ORDERBOOK_ABM_FULL.bz2")

if ob_old.equals(ob_new):
    print("ORDER BOOKS ARE MATCHING")
else:
    print("ORDER BOOKS ARE NOT MATCHING")
