#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: xiaowei
"""

from sklearn.model_selection import ParameterGrid
from Model import Model
from Buffer import Buff
import constants as c
import helpers
import numpy as np
import random
from train_initial import synthetic_lst

#%%
params = {'n_hidden_layers': [2], 'neuron_mult': [[0.5, 0.5]]
             , 'keep_prob': [0.65], 'lr': [0.0004]
             , 'q_discount': [0.98], 'tau': [0.0002]
             , 'update_freq': [4]
             , 'beta': [0.0002]
             }

#%%
batch_size = 32
update_freq = 4

#%%
buff = Buff(update_freq, batch_size, c.buffer_path)

#%%
buff.cut_old()
print (buff.buffer_df.shape)

#%%
lst_buff = synthetic_lst()

#%%
grid_save = []
save_freq = update_freq * 10
for num, param in enumerate(ParameterGrid(params)):
    generator = buff.samples(16000)
    mod = Model(buff.buffer_cols, c.action_nums, param, c.mod_path, c.mod_records_path, True)
    print (param)
    for experience in generator: #s0, actions, rewards, s1, done
        mod.train_on_sample(experience, buff.dev_sample)
        if random.random() < 0.25:
            tbuff = np.random.choice(lst_buff)
            mod.train_on_sample(tbuff.one_sample(), tbuff.dev_sample, toplot=False)
    mod.clean_up(save_mod=False)#save_mod=True
    mod.plot(helpers.get_plot_path() + str(num) + '.pdf')
    grid_save.append(mod)
#    break

        #%%
