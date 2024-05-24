# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling RVT or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
#
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The standard DQN replay memory.

This implementation is an out-of-graph replay memory + in-graph wrapper. It
supports vanilla n-step updates of the form typically found in the literature,
i.e. where rewards are accumulated for n steps and the intermediate trajectory
is not exposed to the agent. This does not allow, for example, performing
off-policy corrections.
"""
import ctypes
import collections
import concurrent.futures
import os
from os.path import join
import pickle
from typing import List, Tuple, Type
import time
import math
# from threading import Lock
import multiprocessing as mp
from multiprocessing import Lock
import numpy as np
import logging

from natsort import natsort

from yarr.replay_buffer.replay_buffer import ReplayBuffer, ReplayElement
from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer

import sys

# Defines a type describing part of the tuple returned by the replay
# memory. Each element of the tuple is a tensor of shape [batch, ...] where
# ... is defined the 'shape' field of ReplayElement. The tensor type is
# given by the 'type' field. The 'name' field is for convenience and ease of
# debugging.


# String constants for storage
ACTION = 'action'
REWARD = 'reward'
TERMINAL = 'terminal'
TIMEOUT = 'timeout'
INDICES = 'indices'

import threading

class ReadWriteLock:
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self):
        with self._read_ready:
            self._readers += 1

    def release_read(self):
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self):
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        self._read_ready.release()


def invalid_range(cursor, replay_capacity, stack_size, update_horizon):
    """Returns a array with the indices of cursor-related invalid transitions.

    There are update_horizon + stack_size invalid indices:
      - The update_horizon indices before the cursor, because we do not have a
        valid N-step transition (including the next state).
      - The stack_size indices on or immediately after the cursor.
    If N = update_horizon, K = stack_size, and the cursor is at c, invalid
    indices are:
      c - N, c - N + 1, ..., c, c + 1, ..., c + K - 1.

    It handles special cases in a circular buffer in the beginning and the end.

    Args:
      cursor: int, the position of the cursor.
      replay_capacity: int, the size of the replay memory.
      stack_size: int, the size of the stacks returned by the replay memory.
      update_horizon: int, the agent's update horizon.
    Returns:
      np.array of size stack_size with the invalid indices.
    """
    assert cursor < replay_capacity
    return np.array(
        [(cursor - update_horizon + i) % replay_capacity
         for i in range(stack_size + update_horizon)])


class UniformReplayBufferOptimized(UniformReplayBuffer):
    """A simple out-of-graph Replay Buffer.

    Stores transitions, state, action, reward, next_state, terminal (and any
    extra contents specified) in a circular buffer and provides a uniform
    transition sampling function.

    When the states consist of stacks of observations storing the states is
    inefficient. This class writes observations and constructs the stacked states
    at sample time.

    Attributes:
      _add_count: int, counter of how many transitions have been added (including
        the blank ones at the beginning of an episode).
      invalid_range: np.array, an array with the indices of cursor-related invalid
        transitions
    """

    # def __init__(self,
    #             batch_size: int = 32,
    #             timesteps: int = 1,
    #             replay_capacity: int = int(1e6),
    #             update_horizon: int = 1,
    #             gamma: float = 0.99,
    #             max_sample_attempts: int = 10000,
    #             action_shape: tuple = (),
    #             action_dtype: Type[np.dtype] = np.float32,
    #             reward_shape: tuple = (),
    #             reward_dtype: Type[np.dtype] = np.float32,
    #             observation_elements: List[ObservationElement] = None,
    #             extra_replay_elements: List[ReplayElement] = None,
    #             disk_saving: bool = False,
    #             purge_replay_on_shutdown: bool = True,
    #             optimized_training=False,
    #             ):

    #     super(UniformReplayBufferOptimized, self).__init__(
    #         batch_size=batch_size,
    #         timesteps=timesteps,
    #         replay_capacity=replay_capacity,
    #         update_horizon=update_horizon,
    #         gamma=gamma,
    #         max_sample_attempts=max_sample_attempts,
    #         action_shape=action_shape,
    #         action_dtype=action_dtype,
    #         reward_shape=reward_shape,
    #         reward_dtype=reward_dtype,
    #         observation_elements=observation_elements,
    #         extra_replay_elements=extra_replay_elements,
    #         disk_saving=disk_saving,
    #         purge_replay_on_shutdown=purge_replay_on_shutdown,
    #         optimized_training=optimized_training,
    #     )

    #     self._rw_lock = ReadWriteLock()

    def _get_from_disk(self, start_index, end_index):
        """Returns the range of array at the index handling wraparound if necessary.

        Args:
          start_index: int, index to the start of the range to be returned. Range
            will wraparound if start_index is smaller than 0.
          end_index: int, exclusive end index. Range will wraparound if end_index
            exceeds replay_capacity.

        Returns:
          np.array, with shape [end_index - start_index, array.shape[1:]].
        """

        overall_time = time.time()

        assert end_index > start_index, 'end_index must be larger than start_index'
        assert end_index >= 0
        assert start_index < self._replay_capacity
        if not self.is_full():
            assert end_index <= self.cursor(), (
                'Index {} has not been added.'.format(start_index))

        # Here we fake a mini store (buffer)
        store = {store_element.name: {}
                 for store_element in self._storage_signature}

        if start_index % self._replay_capacity < end_index % self._replay_capacity:
            for i in range(start_index, end_index):
                task_replay_storage_folder = self._task_replay_storage_folders[self._index_mapping[i, 0]]
                task_index = self._index_mapping[i, 1]

                # NOTE: We need to skip loading the missing *.replay files as they were not saved during optimised replay bufffer generation
                try:
                    with open(join(task_replay_storage_folder, '%d.replay' % task_index), 'rb') as f:
                        # print("File name: ", join(task_replay_storage_folder, '%d.replay' % task_index))
                        d = pickle.load(f)
                        for k, v in d.items():
                            # print("#########: ", k, "       ", i)
                            store[k][i] = v # NOTE: potential bug here, should % self._replay_capacity
                    store['replay_file_name'][i] = str(join(task_replay_storage_folder, '%d.replay' % task_index))
                except:
                    # print("Cur i value: ", i, "     start idx: ", start_index, "     end_idx: ", end_index)
                    print("Failed to load: ", join(task_replay_storage_folder, '%d.replay' % task_index))
                    sys.stdout.flush()
                    
                    # since we encountered a corrupted replay file, we jsut load 0.replay corresponding to that task
                    with open(join(task_replay_storage_folder, '0.replay'), 'rb') as f:
                        # print("File name: ", join(task_replay_storage_folder, '%d.replay' % task_index))
                        d = pickle.load(f)
                        for k, v in d.items():
                            # print("#########: ", k, "       ", i)
                            store[k][i] = v # NOTE: potential bug here, should % self._replay_capacity
                    store['replay_file_name'][i] = str(join(task_replay_storage_folder, '%d.replay' % task_index))

        else:
            for i in range(end_index - start_index):
                idx = (start_index + i) % self._replay_capacity
                task_replay_storage_folder = self._task_replay_storage_folders[self._index_mapping[idx, 0]]
                task_index = self._index_mapping[idx, 1]
                with open(join(task_replay_storage_folder, '%d.replay' % task_index), 'rb') as f:
                    d = pickle.load(f)
                    for k, v in d.items():
                        store[k][idx] = v
        
        overall_time = time.time() - overall_time
        # print(f"Loading function in {overall_time:.4f} seconds")
        sys.stdout.flush()
        return store

    def sample_transition_batch(self, batch_size=None, indices=None,
                                pack_in_dict=True, distribution_mode = 'transition_uniform'):
        """Returns a batch of transitions (including any extra contents).

        If get_transition_elements has been overridden and defines elements not
        stored in self._store, an empty array will be returned and it will be
        left to the child class to fill it. For example, for the child class
        OutOfGraphPrioritizedReplayBuffer, the contents of the
        sampling_probabilities are stored separately in a sum tree.

        When the transition is terminal next_state_batch has undefined contents.

        NOTE: This transition contains the indices of the sampled elements.
        These are only valid during the call to sample_transition_batch,
        i.e. they may  be used by subclasses of this replay buffer but may
        point to different data as soon as sampling is done.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
          indices: None or list of ints, the indices of every transition in the
            batch. If None, sample the indices uniformly.

        Returns:
          transition_batch: tuple of np.arrays with the shape and type as in
            get_transition_elements().

        Raises:
          ValueError: If an element to be sampled is missing from the
            replay buffer.
        """

        # print("********************************************** \n executing sample_transition_batch... \n **************************************************")

        if batch_size is None:
            batch_size = self._batch_size

        # self._rw_lock.acquire_read()

        with self._lock:
            start_time = time.time()
            
            if indices is None:
                indices = self.sample_index_batch(batch_size, distribution_mode)
            assert len(indices) == batch_size
            # print(f"Index sampling time: {time.time() - start_time} seconds")
            sys.stdout.flush()

            # print("********************************************** \n Generated indices... : **************************************************", indices)

            transition_elements = self.get_transition_elements(batch_size)
            batch_arrays = self._create_batch_arrays(batch_size)
            task_name_arrays = []

            # print("********************************************** \n Entering sample transition for loop... \n **************************************************")

            for batch_element, state_index in enumerate(indices):

                if not self.is_valid_transition(state_index):
                    raise ValueError('Invalid index %d.' % state_index)

                task_name_arrays.append(self._task_names[self._index_mapping[state_index, 0]])

                trajectory_indices = [(state_index + j) % self._replay_capacity
                                      for j in range(self._update_horizon)]
                trajectory_terminals = self._store['terminal'][
                    trajectory_indices]
                is_terminal_transition = trajectory_terminals.any()
                if not is_terminal_transition:
                    trajectory_length = self._update_horizon
                else:
                    # np.argmax of a bool array returns index of the first True.
                    trajectory_length = np.argmax(
                        trajectory_terminals.astype(np.bool_),
                        0) + 1

                # next_state_index = state_index + trajectory_length
                next_state_index = state_index          # NOTE: Set next_state_index to the current state_index as the optimized replay buffer does not have the invalid transition states

                store = self._store

                # print("********************************************** \n Loading from disk... \n **************************************************")

                if self._disk_saving:
                    disk_start_time = time.time()
                    
                    # self._rw_lock.release_read()

                    store = self._get_from_disk(
                        state_index - (self._timesteps - 1),
                        next_state_index + 1)

                    # self._rw_lock.acquire_read()

                # print(f"Disk loading time: {time.time() - disk_start_time} seconds")
                sys.stdout.flush()

                # print("********************************************** \n Loaded from disk!! \n **************************************************")

                terminal_stack = self.get_terminal_stack(state_index)

                for element_array, element in zip(batch_arrays,
                                                transition_elements):
                    if element.is_observation:
                        if element.name.endswith('tp1'):
                            continue
                        else:
                            element_array[
                                batch_element] = self._get_element_stack(
                                store[element.name],
                                state_index, terminal_stack)

                # print("********************************************** \n Exiting sample transition for loop... \n **************************************************")

        # finally:
        #     self._rw_lock.release_read()
        
        if pack_in_dict:
            batch_arrays = self.unpack_transition(
                batch_arrays, transition_elements)

        # TODO: make a proper fix for this
        if 'task' in batch_arrays:
            del batch_arrays['task']
        if 'task_tp1' in batch_arrays:
            del batch_arrays['task_tp1']

        batch_arrays['tasks'] = task_name_arrays

        # print("********************************************** \n Returned Batch of Data ! \n **************************************************")
        # print(f"Total sample transition batch time: {time.time() - start_time} seconds")
        sys.stdout.flush()

        return batch_arrays