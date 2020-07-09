from collections import deque, OrderedDict

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.rollout_functions import rollout, multitask_rollout, ensemble_rollout, ensemble_eval_rollout
from rlkit.samplers.rollout_functions import ensemble_ucb_rollout
from rlkit.samplers.data_collector.base import PathCollector

class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            noise_flag=0,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._noise_flag = noise_flag
        
        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = rollout(
                self._env,
                self._policy,
                noise_flag=self._noise_flag,
                max_path_length=max_path_length_this_loop,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def collect_normalized_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            input_mean,
            input_std,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = normalized_rollout(
                self._env,
                self._policy,
                noise_flag=self._noise_flag,
                max_path_length=max_path_length_this_loop,
                input_mean=input_mean,
                input_std=input_std,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths
    
    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )
    
    
class EnsembleMdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            num_ensemble,
            noise_flag=0,
            ber_mean=0.5,
            eval_flag=False,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            critic1=None,
            critic2=None,
            inference_type=0.0,
            feedback_type=1,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self.num_ensemble = num_ensemble
        self.eval_flag = eval_flag
        self.ber_mean = ber_mean
        self.critic1 = critic1
        self.critic2 = critic2
        self.inference_type = inference_type
        self.feedback_type = feedback_type
        self._noise_flag = noise_flag
        
        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            if self.eval_flag:
                path = ensemble_eval_rollout(
                    self._env,
                    self._policy,
                    self.num_ensemble,
                    max_path_length=max_path_length_this_loop,
                )
            else:
                if self.inference_type > 0: # UCB
                    path = ensemble_ucb_rollout(
                        self._env,
                        self._policy,
                        critic1=self.critic1,
                        critic2=self.critic2,
                        inference_type=self.inference_type,
                        feedback_type=self.feedback_type,
                        num_ensemble=self.num_ensemble,
                        noise_flag=self._noise_flag,
                        max_path_length=max_path_length_this_loop,
                        ber_mean=self.ber_mean,
                    )
                else:
                    path = ensemble_rollout(
                        self._env,
                        self._policy,
                        self.num_ensemble,
                        noise_flag=self._noise_flag,
                        max_path_length=max_path_length_this_loop,
                        ber_mean=self.ber_mean,
                    )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )
