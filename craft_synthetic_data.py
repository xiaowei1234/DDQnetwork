#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: xiaowei
This is NOT used in production. Used to help the the initial model
"""
import helpers
import constants as c
import numpy as np
from Buffer import Buff

#%%

def random_apps(df):
    idx = np.random.randint(df.shape[0], size=np.int32(df.shape[0]/4))
    apps_df = df.iloc[idx][['app_code']].drop_duplicates()
    return df.merge(apps_df, how='inner', on='app_code')


def make_synthetic_action(orig_df, action):
    df = random_apps(orig_df)
    df['actions'] = action
    val = c.action_nums[action]
    if val < 0.01:
        df['reward'] = 0
    else:
        retain = df.handset * 0.01 - c.min_amount > val
        df = df.loc[retain]
        pos = df.reward > 0
        df.loc[pos, 'reward'] = val
        df.loc[~pos, 'reward'] = -1
    return df

#%%
if __name__ == '__main__':
    buff = Buff(c.update_freq, c.batch_size, c.buffer_path, buffer_size=200000, dev_size=0.01)
    buff.cut_old()
    bdf = helpers.load_pickle(c.buffer_path)
    for act in range(1, len(c.action_nums)):
        df = make_synthetic_action(bdf, act)
        helpers.write_pickle(c.env_path + 'synth_' + str(act) + '.pkl', df)