import datetime
import functools
import os
import random
import shutil
import tempfile
from typing import Callable, Dict, List, Optional, Union

import gym
import numpy as np
import torch

from research.envs import PolicyID
from research.utils import utils

from . import sampling, storage, ReplayBuffer


class MultiReplayBuffer(torch.utils.data.IterableDataset):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **buffer_configs) -> None:
        self.buffers = {}
        self.weights = {}
        self.iters = {}
        for buffer_name, config in buffer_configs.items():
            self.buffers[buffer_name] = ReplayBuffer(observation_space, action_space, **config)
            self.weights[buffer_name] = 0

    def update_weights(self):
        total_episodes = sum(len(buffer.episode_filenames) for buffer in self.buffers.values())
        if total_episodes > 0:
            for buffer_name, buffer in self.buffers.items():
                self.weights[buffer_name] = len(buffer.episode_filenames) / total_episodes

    def add(self, buffer_name, **kwargs):
        assert buffer_name in self.buffers, f"'{buffer_name}' not a buffer"
        self.buffers[buffer_name].add(**kwargs)

    def extend(self, buffer_name, **kwargs):
        assert buffer_name in self.buffers, f"'{buffer_name}' not a buffer"
        self.buffers[buffer_name].extend(**kwargs)

    def sample(self, batch_size, *args, **kwargs):
        self.update_weights()
        concatenated_samples = None
        for buffer_name, weight in self.weights.items():
            buffer_batch_size = int(batch_size * weight)
            if buffer_batch_size > 0:
                sample = self.buffers[buffer_name].sample(*args, batch_size=buffer_batch_size, **kwargs)
                if concatenated_samples is None:
                    concatenated_samples = sample
                else:
                    concatenated_samples = utils.concatenate(concatenated_samples, sample, dim=0)
        return concatenated_samples

    def save(self, path):
        for _buffer_name, buffer in self.buffers.items():
            buffer.save(path, prefix=buffer.prefix)

    def __iter__(self):
        for buffer_name, buffer in self.buffers.items():
            self.iters[buffer_name] = iter(buffer)
        while True:
            concatenated_batch = None
            self.update_weights()
            empty_iters = 0
            for buffer_name in self.buffers.keys():
                try:
                    og_batch_size = self.buffers[buffer_name].sample_fn.keywords.get("batch_size")
                    new_batch_size = int(
                        self.buffers[buffer_name].sample_fn.keywords.get("batch_size") * self.weights[buffer_name]
                    )
                    self.buffers[buffer_name].sample_fn.keywords["batch_size"] = new_batch_size
                    new_batch = next(self.iters[buffer_name])
                    concatenated_batch = (
                        new_batch
                        if concatenated_batch is None
                        else utils.concatenate(concatenated_batch, new_batch, dim=0)
                    )
                    self.buffers[buffer_name].sample_fn.keywords["batch_size"] = og_batch_size
                except StopIteration:
                    empty_iters += 1
            if empty_iters == len(self.iters):
                break
            if concatenated_batch is None:
                break

            yield concatenated_batch
