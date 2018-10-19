#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: xiaowei
"""
import helpers
import constants as c
import numpy as np
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
    bdf = helpers.load_pickle(c.buffer_path)
    for act in range(1, len(c.action_nums)):
        df = make_synthetic_action(bdf, act)
        helpers.write_pickle(c.env_path + 'synth_' + str(act) + '.pkl', df)