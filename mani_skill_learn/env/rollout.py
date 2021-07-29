import itertools
from collections import defaultdict
from copy import deepcopy
import numpy as np

from mani_skill_learn.utils.data import (to_np, concat_list_of_array, stack_dict_of_list_array, flatten_dict,
                             compress_size, unsqueeze, stack_list_of_array)
from mani_skill_learn.utils.meta import dict_of
from .builder import ROLLOUTS
from .env_utils import build_env, true_done


@ROLLOUTS.register_module()
class Rollout:
    def __init__(self, env_cfg, worker_id=None, use_cost=False, with_info=False, reward_only=True, **kwargs):
        self.n = 1
        self.worker_id = worker_id
        self.env = build_env(env_cfg)
        if hasattr(self.env, 'seed') and worker_id is not None:
            # Assume parallel-run is multi-process
            self.env.seed(np.random.randint(0, 10000) + worker_id)
        self.iscost = -1 if use_cost else 1
        self.reward_only = reward_only
        self.with_info = with_info
        self.recent_obs = None
        self.step = 0
        self.reset()

    def reset(self, **kwargs):
        self.step = 0
        reset_kwargs = {}
        kwargs = deepcopy(dict(kwargs))
        if 'level' in kwargs:
            reset_kwargs['level'] = kwargs['level']
        # flush_print(self.worker_id, 'Begin reset')
        self.recent_obs = compress_size(self.env.reset(**reset_kwargs))
        # flush_print(self.worker_id, 'End reset')

    def random_action(self):
        return self.env.action_space.sample()

    def forward_with_reset(self, states=None, actions=None):
        """
        :param states: [n, m] n different env states
        :param actions: [n, c, a] n sequences of actions
        :return: rewards [n, c]
        """
        # for CEM only
        assert self.reward_only
        rewards = []
        for s, a in zip(states, actions):
            self.env.set_state(s)
            reward_episode = []
            for action in a:
                ob, r, done, _ = self.env.step(action)
                reward_episode.append(r * self.iscost)
            rewards.append(reward_episode)
        rewards = np.array(rewards, dtype=np.float32)
        return rewards

    def forward_with_policy(self, pi=None, num=1, whole_episode=False):
        assert not self.reward_only and self.recent_obs is not None
        obs, next_obs, actions, rewards, dones, episode_dones = [], [], [], [], [], []
        infos = defaultdict(list)

        if pi is not None:
            import torch
            from mani_skill_learn.utils.data import to_torch
            device = pi.device

        for i in itertools.count(1):
            if pi is not None:
                with torch.no_grad():
                    recent_obs = to_torch(self.recent_obs, dtype='float32', device=device)
                    a = to_np(pi(unsqueeze(recent_obs, axis=0)))[0]
            else:
                a = self.random_action()
            ob, r, done, info = self.env.step(a)

            ob = compress_size(ob)
            self.step += 1
            episode_done = done
            # done = done if self.step < self.env._max_episode_steps else False
            done = true_done(done, info)

            obs.append(self.recent_obs)
            next_obs.append(ob)
            actions.append(a)
            rewards.append(compress_size(r * self.iscost))

            dones.append(done)
            episode_dones.append(episode_done)
            if self.with_info:
                info = flatten_dict(info)
                for key in info:
                    infos[key].append(info[key])
            self.recent_obs = ob
            if episode_done:
                self.reset()
            if i >= num and (episode_done or not whole_episode):
                break
        ret = dict_of(obs, actions, next_obs, rewards, dones, episode_dones)
        ret = stack_dict_of_list_array(ret)
        infos = stack_dict_of_list_array(dict(infos))
        return ret, infos

    def forward_single(self, action=None):
        """
        :param action: [a] one action
        :return: all information
        """
        assert not self.reward_only and self.recent_obs is not None
        if action is None:
            action = self.random_action()
        actions = action
        obs = self.recent_obs
        next_obs, rewards, dones, info = self.env.step(actions)
        rewards *= self.iscost
        episode_dones = dones

        next_obs = compress_size(next_obs)
        rewards = compress_size(rewards)

        self.step += 1
        dones = true_done(dones, info)
        ret = dict_of(obs, actions, next_obs, rewards, dones, episode_dones)
        self.recent_obs = next_obs
        if self.with_info:
            info = flatten_dict(info)
        else:
            info = {}
        if episode_dones:
            self.reset()
        return ret, info

    def close(self):
        if self.env:
            del self.env


@ROLLOUTS.register_module()
class BatchRollout:
    def __init__(self, env_cfg, num_procs=20, synchronize=True, reward_only=False, **kwargs):
        self.n = num_procs
        self.synchronize = synchronize
        self.reward_only = reward_only
        self.workers = []

        if synchronize:
            from ..env.parallel_runner import NormalWorker as Worker
        else:
            from ..env.torch_parallel_runner import TorchWorker as Worker
            print("This will consume a lot of memory due to cuda")
        for i in range(self.n):
            self.workers.append(Worker(Rollout, i, env_cfg, reward_only=reward_only, **kwargs))

    def reset(self, **kwargs):
        for i in range(self.n):
            self.workers[i].call('reset', **kwargs)
        for i in range(self.n):
            self.workers[i].get()

    @property
    def recent_obs(self):
        for i in range(self.n):
            self.workers[i].get_attr('recent_obs')
        return stack_list_of_array([self.workers[i].get() for i in range(self.n)])

    def random_action(self):
        for i in range(self.n):
            self.workers[i].call('random_action')
        return np.array([self.workers[i].get() for i in range(self.n)])

    def forward_with_reset(self, states=None, actions=None):
        from .parallel_runner import split_list_of_parameters
        paras = split_list_of_parameters(self.n, states=states, actions=actions)
        n = len(paras)
        for i in range(n):
            args_i, kwargs_i = paras[i]
            self.workers[i].call('forward_with_reset', *args_i, **kwargs_i)
        reward = [self.workers[i].get() for i in range(n)]
        reward = concat_list_of_array(reward)
        return reward

    def forward_with_policy(self, policy, num, whole_episode=False, merge=True):
        from mani_skill_learn.utils.math import split_num
        n, running_steps = split_num(num, self.n)
        batch_size = max(running_steps)
        if self.synchronize and policy is not None:
            """
            When the we run with random actions, it is ok to use asynchronizedly
            """
            device = policy.device
            trajectories = defaultdict(lambda: [[] for i in range(n)])
            infos = defaultdict(lambda: [[] for i in range(n)])
            for i in range(batch_size):
                current_n = 0
                for j in range(n):
                    if i < running_steps[j]:
                        current_n += 1
                assert current_n > 0
                if policy is None:
                    action = None
                else:
                    import torch
                    from mani_skill_learn.utils.data import to_torch
                    with torch.no_grad():
                        recent_obs = to_torch(self.recent_obs, dtype='float32', device=device)
                        action = to_np(policy(recent_obs))[:current_n]

                for j in range(current_n):
                    self.workers[j].call('forward_single', action=None if action is None else action[j])

                for j in range(current_n):
                    traj, info = self.workers[j].get()
                    for key in traj:
                        trajectories[key][j].append(traj[key])
                    for key in info:
                        infos[key][j].append(info[key])
            trajectories = [{key: stack_list_of_array(trajectories[key][i]) for key in trajectories} for i in range(n)]
            infos = [{key: stack_list_of_array(infos[key][i]) for key in infos} for i in range(n)]
        else:
            for i in range(n):
                if policy is not None:
                    assert not self.synchronize
                self.workers[i].call('forward_with_policy', pi=policy, num=running_steps[i],
                                     whole_episode=whole_episode)
            ret = [self.workers[i].get() for i in range(n)]
            trajectories = [ret[i][0] for i in range(n)]
            infos = [ret[i][1] for i in range(n)]

        if merge:
            trajectories = concat_list_of_array(trajectories)
            infos = concat_list_of_array(infos)
            """
            Concat: [Process 0, Process1, ..., Process n]
            """
        return trajectories, infos

    def close(self):
        for worker in self.workers:
            worker.call('close')
            worker.close()
