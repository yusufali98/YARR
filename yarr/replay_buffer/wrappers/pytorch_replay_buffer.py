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

# From: https://github.com/stepjam/YARR/blob/main/yarr/replay_buffer/wrappers/pytorch_replay_buffer.py

import time
from threading import Lock, Thread

import os
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader, Dataset

from yarr.replay_buffer.replay_buffer import ReplayBuffer
from yarr.replay_buffer.wrappers import WrappedReplayBuffer
import time
import sys

import pickle


class PyTorchIterableReplayDataset(IterableDataset):

    def __init__(self, replay_buffer: ReplayBuffer, sample_mode, sample_distribution_mode = 'transition_uniform'):
        self._replay_buffer = replay_buffer
        self._sample_mode = sample_mode
        if self._sample_mode == 'enumerate':
            self._num_samples = self._replay_buffer.prepare_enumeration()
        self._sample_distribution_mode = sample_distribution_mode

    def _generator(self):
        while True:
            start_time = time.time()
            if self._sample_mode == 'random':
                yield self._replay_buffer.sample_transition_batch(pack_in_dict=True, distribution_mode = self._sample_distribution_mode)
                print(f"Random Batch generation time: {time.time() - start_time} seconds")
                sys.stdout.flush()
            elif self._sample_mode == 'enumerate':
                yield self._replay_buffer.enumerate_next_transition_batch(pack_in_dict=True)
                print(f"Enumerate Batch generation time: {time.time() - start_time} seconds")
                sys.stdout.flush()

    def __iter__(self):
        return iter(self._generator())

    def __len__(self): # enumeration will throw away the last incomplete batch
        return self._num_samples // self._replay_buffer._batch_size


class PyTorchStaticReplayDataset(Dataset):

    def __init__(self, replay_buffer: ReplayBuffer):
        self._replay_buffer = replay_buffer
        self._num_samples = self._replay_buffer.prepare_enumeration()
        self._load_all_samples()

    def _load_all_samples(self):
        self._samples = []

        print("Loading all samples from .replay files....")
        sys.stdout.flush()

        start_time = time.time()
        # self._samples = self._replay_buffer.enumerate_next_transition_batch(batch_size=self._num_samples, pack_in_dict=True)
        with open('/workspace/RVT/rvt/big_replay.pkl', 'rb') as f:
            self._samples = pickle.load(f)
        print(f"Loading dataset in memory time: {time.time() - start_time} seconds")
        # self._samples.append(sample)
    
        print("Loaded all samples from memory !")
        sys.stdout.flush()

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self):
        return self._num_samples


class PyTorchReplayBuffer(WrappedReplayBuffer):
    """Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism.

    Usage:
      To add a transition:  call the add function.

      To sample a batch:    Construct operations that depend on any of the
                            tensors is the transition dictionary. Every sess.run
                            that requires any of these tensors will sample a new
                            transition.
      sample_mode: the mode to sample data, choose from ['random', 'enumerate']
    """

    def __init__(self, replay_buffer: ReplayBuffer, num_workers: int = 2, sample_mode = 'random', sample_distribution_mode = 'transition_uniform'):
        
        start_time = time.time()

        super(PyTorchReplayBuffer, self).__init__(replay_buffer)
        self._num_workers = num_workers
        self._sample_mode = sample_mode
        self._sample_distribution_mode = sample_distribution_mode

        print("PyTorch Replay Buffer Sampling Mode: ", self._sample_mode)
        print(f"Replay buffer initialization time: {time.time() - start_time} seconds")
        sys.stdout.flush()

    def dataset(self) -> DataLoader:
        start_time = time.time()

        # d = PyTorchIterableReplayDataset(self._replay_buffer, self._sample_mode, self._sample_distribution_mode)
        d = PyTorchStaticReplayDataset(self._replay_buffer)

        print(f"Internal IterableDataset creation time: {time.time() - start_time} seconds")
        sys.stdout.flush()
        
        # Batch size None disables automatic batching
        start_time = time.time()
        dataloader = DataLoader(d, batch_size=None, pin_memory=True, num_workers=self._num_workers)
        print(f"Internal DataLoader creation time: {time.time() - start_time} seconds")
        sys.stdout.flush()

        return dataloader    