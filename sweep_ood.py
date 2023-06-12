import typing as th
from jsonargparse import ActionConfigFile
import os
from jsonargparse import ArgumentParser
from jsonargparse.actions import ActionConfigFile
from pathlib import Path
import traceback
from dysweep import dysweep_run_resume, ResumableSweepConfig
from pprint import pprint
from ood import run_ood


def build_args():
    parser = ArgumentParser()
    parser.add_class_arguments(
        ResumableSweepConfig,
        fail_untyped=False,
        sub_configs=True,
    )
    parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    args = parser.parse_args()

    # set the lightning logger to true
    args.use_lightning_logger = False

    # set the default root dir to the current working directory
    args.default_root_dir = Path.cwd() / args.default_root_dir
    # if the path does not exists then create it
    if not os.path.exists(args.default_root_dir):
        os.makedirs(args.default_root_dir)

    return args


def run(config, checkpoint_dir):
    try:
        run_ood(config)
    except Exception as e:
        print("Exception:\n", e)
        print(traceback.format_exc())
        print("-----------")
        raise e


if __name__ == "__main__":
    args = build_args()
    try:
        dysweep_run_resume(args, run)
    except Exception as e:
        print("Exception:\n", e)
        print(traceback.format_exc())
        print("-----------")
        raise e
