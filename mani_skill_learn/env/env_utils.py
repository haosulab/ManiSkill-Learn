from copy import deepcopy

from gym.core import gym
from gym.envs import registry

from mani_skill_learn.utils.data import get_shape
from mani_skill_learn.utils.meta import Registry, build_from_cfg
from .wrappers import MujocoWrapper, SapienRLWrapper, PendulumWrapper, build_wrapper

ENVS = Registry('env')


def get_gym_env_type(env_name):
    if env_name not in registry.env_specs:
        raise ValueError("No such env")
    entry_point = registry.env_specs[env_name].entry_point
    if entry_point.startswith("gym.envs."):
        type_name = entry_point[len("gym.envs."):].split(":")[0].split('.')[0]
    else:
        type_name = entry_point.split('.')[0]
    return type_name


def get_env_state(env):
    ret = {}
    if hasattr(env, 'get_state'):
        ret['env_states'] = env.get_state()
    if hasattr(env.unwrapped, '_scene'):
        ret['env_scene_states'] = env.unwrapped._scene.pack()
    if hasattr(env, 'level'):
        ret['env_levels'] = env.level
    return ret


def true_done(done, info):
    # Process gym standard time limit wrapper
    if done:
        if 'TimeLimit.truncated' in info and info['TimeLimit.truncated']:
            return False
        else:
            return True
    return False


def get_env_info(env_cfg):
    env = build_env(env_cfg)
    obs = env.reset()
    action = env.action_space.sample()
    action_space = deepcopy(env.action_space)
    return get_shape(obs), len(action), action_space


def make_gym_env(env_name, unwrapped=False, time_horizon_factor=1, stack_frame=1, **kwargs):
    tmp_kwargs = deepcopy(kwargs)

    # To handle a bug in mani skill
    tmp_kwargs.pop('obs_mode', None)
    tmp_kwargs.pop('reward_type', None)

    env = gym.make(env_name, **tmp_kwargs)
    if env is None:
        print(f"No {env_name} in gym")
        exit(0)
    if hasattr(env, '_max_episode_steps'):
        env._max_episode_steps *= time_horizon_factor

    if unwrapped:
        env = env.unwrapped if hasattr(env, 'unwrapped') else env

    env_type = get_gym_env_type(env_name)
    # Add our customed wrapper to support set_state which is needed in MPC
    if env_name == 'Pendulum-v0':
        env = PendulumWrapper(env)
    elif env_type == 'mujoco':
        env = MujocoWrapper(env)
    elif env_type == 'mani_skill':
        obs_mode = kwargs.get('obs_mode', None)
        reward_type = kwargs.get('reward_type', None)
        env.set_env_mode(obs_mode=obs_mode, reward_type=reward_type)
        env = SapienRLWrapper(env, stack_frame)
    else:
        print(f'Unsupported env_type {env_type}')

    extra_wrappers = kwargs.get('extra_wrappers', None)
    if extra_wrappers is not None:
        extra_wrappers.env = env
        env = build_wrapper(extra_wrappers)
    return env


ENVS.register_module('gym', module=make_gym_env)


def build_env(cfg, default_args=None):
    return build_from_cfg(cfg, ENVS, default_args)
