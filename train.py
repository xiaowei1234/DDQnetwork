#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: xiaowei

train online model in production
"""
from Buffer import Buff
from Model import Model
import helpers
import constants as c

#%%
if __name__ == '__main__':
    buff = Buff(c.update_freq, c.batch_size, c.buffer_path, buffer_size=200000, dev_size=0.01)
    buff.cut_old()
    mod = Model(buff.buffer_cols, c.action_nums, c.params, c.mod_path, c.mod_records_path, True)
    for experience in buff.samples():
        mod.train_on_sample(experience, buff.dev_sample)
    mod.plot(helpers.get_plot_path() + 'plot.pdf')
    mod.clean_up(True)
