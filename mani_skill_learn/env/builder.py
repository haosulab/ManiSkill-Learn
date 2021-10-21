import contextlib, os

try:
    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    with contextlib.redirect_stdout(None):
        import mani_skill.env
        # import d4rl    # If you want to run on d4rl, you can add this line.
except ImportError as e:
    print("!!! WARNING: mani_skill.env import error (when executing mani_skill_learn/env/builder.py) !!!")
    print("Please check your installation. The error is:")
    print(e)
    print("")
    print("------ Continuing program execution, ignoring mani_skill.env ------")
    
from mani_skill_learn.utils.meta import Registry, build_from_cfg


ROLLOUTS = Registry('rollout')
EVALUATIONS = Registry('evaluation')
REPLAYS = Registry('replay')


def build_rollout(cfg, default_args=None):
    if cfg.get('num_procs', 1) > 1:
        cfg.type = 'BatchRollout'
    return build_from_cfg(cfg, ROLLOUTS, default_args)


def build_evaluation(cfg, default_args=None):
    if cfg.get('num_procs', 1) > 1:
        cfg.type = 'BatchEvaluation'
    return build_from_cfg(cfg, EVALUATIONS, default_args)


def build_replay(cfg, default_args=None):
    return build_from_cfg(cfg, REPLAYS, default_args)
