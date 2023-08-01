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

class NewModelB:
    def __init__(
        self,
        c_eff: int,
        d_eff: str,
    ):
        self.c_eff = c_eff
        self.d_eff = d_eff

class ModelA:
    def __init__(
        self,
        a: int,
        b: str,
        model_b: dict,
    ):
        self.a = a
        self.b = b
        import dypy as dy
        self.model_b = dy.eval(model_b['class_path']
                               )(**model_b['init_args'])