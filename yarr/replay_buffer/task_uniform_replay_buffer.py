import numpy as np
import os
from os.path import join
import pickle
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import invalid_range

ACTION = 'action'
REWARD = 'reward'
TERMINAL = 'terminal'
TIMEOUT = 'timeout'
INDICES = 'indices'
TASK = 'task'


class TaskUniformReplayBuffer(UniformReplayBuffer):
    """
    A uniform with uniform task sampling for each batch
    """

    def __init__(self, *args, **kwargs):
        """Initializes OutOfGraphPrioritizedReplayBuffer."""
        super(TaskUniformReplayBuffer, self).__init__(*args, **kwargs)
        self._task_idxs = dict()

    def _add(self, kwargs: dict):
        """Internal add method to add to the storage arrays.

        Args:
          kwargs: All the elements in a transition.
        """
        with self._lock:
            cursor = self.cursor()

            if self._disk_saving:
                term = self._store[TERMINAL]
                term[cursor] = kwargs[TERMINAL]
                self._store[TERMINAL] = term
                # self._store[TERMINAL][cursor] = kwargs[TERMINAL]

                with open(join(self._save_dir, '%d.replay' % cursor), 'wb') as f:
                    pickle.dump(kwargs, f)
                # If first add, then pad for correct wrapping
                if self._add_count.value == 0:
                    self._add_initial_to_disk(kwargs)
            else:
                for name, data in kwargs.items():
                    item = self._store[name]
                    item[cursor] = data
                    self._store[name] = item

                    # self._store[name][cursor] = data
            with self._add_count.get_lock():
                task = kwargs[TASK]
                if task not in self._task_idxs:
                    self._task_idxs[task] = [cursor]
                else:
                    self._task_idxs[task] =  self._task_idxs[task] + [cursor]
                self._add_count.value += 1

            self.invalid_range = invalid_range(
                self.cursor(), self._replay_capacity, self._timesteps,
                self._update_horizon)

    def sample_index_batch(self, batch_size):
        """Returns a batch of valid indices sampled uniformly.

        Args:
          batch_size: int, number of indices returned.

        Returns:
          list of ints, a batch of valid indices sampled uniformly across tasks.

        Raises:
          RuntimeError: If the batch was not constructed after maximum number of
            tries.
        """
        if self.is_full():
            # add_count >= self._replay_capacity > self._stack_size
            min_id = (self.cursor() - self._replay_capacity +
                      self._timesteps - 1)
            max_id = self.cursor() - self._update_horizon
        else:
            min_id = 0
            max_id = self.cursor() - self._update_horizon
            if max_id <= min_id:
                raise RuntimeError(
                    'Cannot sample a batch with fewer than stack size '
                    '({}) + update_horizon ({}) transitions.'.
                    format(self._timesteps, self._update_horizon))

        tasks = list(self._task_idxs.keys())
        attempt_count = 0
        found_indicies = False
        while not found_indicies and attempt_count < 1000:
            sampled_tasks = list(np.random.choice(tasks, batch_size, replace=(batch_size > len(tasks))))
            potential_indices = []
            for task in sampled_tasks:
                sampled_task_idx = np.random.choice(self._task_idxs[task], 1)[0]
                per_task_attempt_count = 0
                while not self.is_valid_transition(sampled_task_idx) and \
                    per_task_attempt_count < self._max_sample_attempts:
                    sampled_task_idx = np.random.choice(self._task_idxs[task], 1)[0]
                    per_task_attempt_count += 1

                if not self.is_valid_transition(sampled_task_idx):
                    attempt_count += 1
                    continue
                else:
                    potential_indices.append(sampled_task_idx)
            found_indicies = len(potential_indices) == batch_size
        indices = potential_indices

        if len(indices) != batch_size:
            raise RuntimeError(
                'Max sample attempts: Tried {} times but only sampled {}'
                ' valid indices. Batch size is {}'.
                    format(self._max_sample_attempts, len(indices), batch_size))

        return indices

    # def sample_transition_batch(self, batch_size=None, indices=None,
    #                             pack_in_dict=True):
    #     batched_arrays = super(TaskUniformReplayBuffer, self).sample_transition_batch(batch_size, indices, pack_in_dict)
    #
    #     # TODO: make a proper fix for this
    #     if 'task' in batched_arrays:
    #         del batched_arrays['task']
    #     if 'task_tp1' in batched_arrays:
    #         del batched_arrays['task_tp1']
    #
    #     return batched_arrays
