# Xiao Wei

from Buffer import Buff
from Model import Model
import helpers
import constants as c
import random
import numpy as np

#%%
def synthetic_lst():
    return [Buff(c.update_freq, c.batch_size, c.env_path + 'synth_1.pkl')
            , Buff(c.update_freq, c.batch_size, c.env_path + 'synth_2.pkl')
            , Buff(c.update_freq, c.batch_size, c.env_path + 'synth_3.pkl')
            ]


#%%
if __name__ == '__main__':
    buff = Buff(c.update_freq, c.batch_size, c.buffer_path, buffer_size=200000, dev_size=0.01)
    buff.cut_old()
    lst_buff = synthetic_lst
    mod = Model(buff.buffer_cols, c.action_nums, c.params, c.mod_path, c.mod_records_path, True)
    for experience in buff.samples(10000):
        mod.train_on_sample(experience, buff.dev_sample)
        if random.random() < 0.25:
            tbuff = np.random.choice(lst_buff)
            mod.train_on_sample(tbuff.one_sample(), tbuff.dev_sample)
    mod.plot(helpers.get_plot_path() + 'plot.pdf')
    mod.clean_up(True)
