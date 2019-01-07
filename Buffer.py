#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: xiaowei
"""
import pandas as pd
import numpy as np
from functools import lru_cache
import constants as c


class Buff:
    """
    Buffer for generating batches for Qnetwork modeling and prediction purposes
    """
    def __init__(self, pass_over, batch_size, path, buffer_size=200000, dev_size=0.01):
        """
        pass_over (int): on average how many times an experience gets trained on
        batch_size (int): number of obs in each batch
        path (str): location of buffer file
        buffer_size (int): # of experiences to store in buffer
        dev_size (float): proportion of buffer to set aside as validation set
        """
        self.path = path
        self.buffer_df = pd.read_pickle(path).sort_values('run_time_ms')
        self.num_samples = max(int((self.buffer_rows - buffer_size) / pass_over), 1)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.dev_size = dev_size

    def cut_old(self, env_path=c.env_path):
        """
        cut buffer_df to buffer_size if buffer bigger than buffer_size
            removes oldest experience and stores in a file
        env_path (str): location to load from/store buffer_df
        """
        if self.buffer_rows > self.buffer_size:
            print ('Cutting old rows: ', self.buffer_rows - self.buffer_size)
            old_df = self.buffer_df.iloc[:-self.buffer_size]
            min_dt = str(old_df.run_date.min())[:10]
            max_dt = str(old_df.run_date.max())[:10]
            old_df.to_pickle(env_path + 'usedUp_' + min_dt + '_' + max_dt + '.pkl')
            self.buffer_df = self.buffer_df.iloc[-self.buffer_size:]
            self.buffer_df.to_pickle(self.path)
        else:
            pass
            # print ('Buffer not filled. No rows cut')

    @property
    def buffer_rows(self):
        return self.buffer_df.shape[0]

    @property
    @lru_cache()
    def buffer_cols(self):
        """
        returns # of columns of data. This is needed to initialize the NN graph
        """
        return self.prep_state(self.buffer_df).shape[1]

    @property
    def epochs(self):
        return self.num_samples
    
    @property
    @lru_cache()
    def sorted_buffer_df(self):
        return self.buffer_df.sort_values(c.identification_vars)

    @staticmethod
    def prep_state(df):
        """
        return numpy matrix from dataframe ready for input into model
        df (pandas dataframe): dataframe 
        """
        to_drops = ['bank_code', 'reward', 'chunk']
        drops = [d for d in to_drops if d in df.columns]
        return df.set_index(c.identification_vars).drop(drops, axis=1).values/10

    def prep_df(self, idx):
        """
        returns a list of values serving as the experience replay for the model
        idx (iterable of int): the sequence of experiences to send to model
        """
        idx2 = [i for i in map(lambda v: v + 1 if v<self.buffer_rows -1 else 0, idx)]
        s0 = self.sorted_buffer_df.iloc[idx]
        s1 = self.sorted_buffer_df.iloc[idx2]
        done = (s0.app_code.values != s1.app_code.values) | (s0.chunk.values != s1.chunk.values)
        s_zero = self.prep_state(s0)
        s_one = self.prep_state(s1)
        return [s_zero, s0.actions.values, s0.reward.values, s_one, done]

    def one_sample(self):
        """
        returns one batch of experiences from buffer
        """
        idx = np.random.randint(np.int(self.buffer_rows * (1.0 - self.dev_size)), size=self.batch_size)
        return self.prep_df(idx)

    def samples(self, override=None):
        """
        generator that yields batches of experiences
        override (int): number of batches to yield. If none yields self.num_samples of batches
        """
        if override is None:
            iterator = range(self.num_samples)
        else:
            iterator = range(override)
        for num in iterator:
            yield self.one_sample()

    @property
    @lru_cache()
    def dev_sample(self):
        """
        returns the validation set as a batch of experiences
        """
        start = int(self.buffer_rows * (1 - self.dev_size))
        idx = [i for i in range(start, self.buffer_rows)]
        return self.prep_df(idx)
