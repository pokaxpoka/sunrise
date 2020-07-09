import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
import numpy as np

class BatchNormalRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            save_frequency=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.save_frequency = save_frequency
        
        # normalization
        self._obs_mean = np.zeros_like(exploration_env.observation_space.low)
        self._obs_std = np.ones_like(exploration_env.observation_space.low)
        self.moving_avg = 0.9
        
    def estimate_obs_stats(self, obs_batch, init_flag=False):
#         if init_flag:
#             self._obs_mean = np.mean(obs_batch, axis=0)
#             self._obs_std = np.std(obs_batch, axis=0)
#         else:
        self._obs_mean = self.moving_avg * self._obs_mean + (1-self.moving_avg) * np.mean(obs_batch, axis=0)
        self._obs_std = self.moving_avg * self._obs_std + (1-self.moving_avg) * np.std(obs_batch, axis=0)

    def apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)
        
    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
            self.estimate_obs_stats(init_expl_paths[0]['observations'], init_flag=True)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_normalized_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
                input_mean=self._obs_mean,
                input_std=self._obs_std,
            )
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_normalized_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                    input_mean=self._obs_mean,
                    input_std=self._obs_std,
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.estimate_obs_stats(train_data['observations'], init_flag=False)
                    train_data['observations'] = self.apply_normalize_obs(train_data['observations'])
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)
            if self.save_frequency > 0:
                if epoch % self.save_frequency == 0:
                    self.trainer.save_models(epoch)
                    self.replay_buffer.save_buffer(epoch)