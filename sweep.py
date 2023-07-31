from dysweep import dysweep_run_resume
from dysweep import ResumableSweepConfig

from train import dysweep_compatible_run
from jsonargparse import ArgumentParser, ActionConfigFile

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

print(args)

dysweep_run_resume(
    conf=args,
    function=dysweep_compatible_run,
)