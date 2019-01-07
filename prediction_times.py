#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 07:32:50 2018

@author: xiaowei

This tests the amount of time it takes to create predictions. Not used during production.
"""
from pipe_ce import ce_pipe_functions
import constants as c
from Model import Model
from Buffer import Buff
import pandas as pd
import time
#%%

def remove_control(df):
    make_control = df.app_code.apply(lambda v: 'W' in v)
    return df.loc[~make_control, :]

#%%
if __name__ == '__main__':
    print ('\nPiping in Collection Engine: ')
    df = pd.read_pickle('../data/raw_10k.pkl')#.iloc[:1]
    t0 = time.time()    
    ce_encoded = ce_pipe_functions(raw_df=df).drop('ranks', axis=1)#.pipe(remove_control)
    print ('Model prediction start')
    state_df = Buff.prep_state(ce_encoded)
    print (state_df.shape)
    mod = Model(state_df.shape[1], c.action_nums, c.params, c.mod_path, c.mod_records_path, False)
    json_str = mod.output_records_json(ce_encoded, c.random_prop)
    t1 = time.time()
    tot_time = t1 - t0
    print (tot_time)