#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: xiaowei
"""
import numpy as np
from Qnetwork import DoubleDuelQ
import matplotlib.pyplot as plt
import helpers
import pandas as pd
from Buffer import Buff
import constants as c


class Model(DoubleDuelQ):
    """
    instantiates or loads a Qnetwork model along with model diagnostics
    """
    def __init__(self, cols, actions, params, mod_path, mod_rec_path, new_mod):
        """
        also instantiates/loads the NN
        cols (int): number of columns of input data
        actions (iterable of float): reward for successful actions
        params (dict): hyperparameters for NN architecture
        mod_path (str): location to load/save model
        mod_rec_path (str): location to load/save model diagnostics
        new_mod (bool): whether to create a new NN model from scratch
        """
        self.new_mod = new_mod
        self.params = params
        self.mod_rec_path = mod_rec_path
        self.actions = actions
        super().__init__(cols, len(actions), params, mod_path, new_mod)
        self.initialize_diagnostics()
        self.this_run_iterations = 0
        self.save_freq = self.params['update_freq'] * 50
        self.iterations = self.sess.run(self.mainQN.global_step)

    def initialize_diagnostics(self):
        """
        initializes or loads model diagnostics
        """
        self.recordings = helpers.load_pickle(self.mod_rec_path, default={})
        if len(self.recordings) == 0 or self.new_mod:
            self.recordings['iter_array'] = np.array([], dtype=np.int32)
            self.recordings['loss_array'] = np.array([], dtype=np.float)
            self.recordings['grad_array'] = np.array([], dtype=np.float)
            self.recordings['grad_std_array'] = np.array([], dtype=np.float)
            self.recordings['action_list'] = []
            self.recordings['reward_array'] = np.array([], dtype=np.float)
            self.recordings['target_array'] = np.array([], dtype=np.float)
            self.recordings['valid_array'] = np.array([], dtype=np.float)
        else:
            for key in self.recordings:
                self.recordings[key] = self.recordings[key][-8000:]

    def train_on_sample(self, experience, validation_experience=None, toplot=True):
        """
        train model on experience and records in diagnostics
        experience (list returned by buffer.sample): training batch from buffer
        validation_experience (list returned by buffer.dev_sample): validation experience batch from buffer
        toplot (boolean): whether to record down and plot or not
        """
        loss, grads, targetQ = self.get_loss_grads(*experience, self.params['beta'])
        m_grad, m_std = self.summarize_grads(grads)
        self.iterations = self.sess.run(self.mainQN.global_step)
        self.this_run_iterations += 1
        if self.iterations < 100:
            return
        if self.iterations % self.params['update_freq'] == 0:
            self.update_target()
        if self.iterations < 2000:
            return
        if not toplot:
            return
        new_rec = self.this_run_iterations % self.save_freq == 0
        if new_rec:
            self.recordings['iter_array'] = np.append(self.recordings['iter_array'], self.iterations)
            if validation_experience is not None:
                self.test_on_validation(validation_experience)
        if len(self.recordings['iter_array']) > 0:
            self.recordings['grad_array'] = self.record_down(self.recordings['grad_array'], m_grad, new_rec)
            self.recordings['grad_std_array'] = self.record_down(self.recordings['grad_std_array'], m_std, new_rec)
            self.recordings['loss_array'] = self.record_down(self.recordings['loss_array'], loss, new_rec)
            self.recordings['reward_array'] = self.record_down(self.recordings['reward_array'], np.sum(experience[2]), new_rec)
            self.recordings['target_array'] = self.record_down(self.recordings['target_array'], np.mean(targetQ), new_rec)
            self.recordings['action_list'] = self.record_actions(self.recordings['action_list'], experience[1], new_rec)

    def test_on_validation(self, experience):
        """
        get model predictions and record down loss from validation set
        experience (list returned by buffer.dev_sample): validation experience batch from buffer 
        """
        loss, grads, targetQ = self.get_loss_grads(*experience, self.params['beta'], update_model=False)
        self.recordings['valid_array'] = self.record_down(self.recordings['valid_array']
            , loss, new_rec=True, use_mean=False)

    def output_records_json(self, ce_encoded, random_prop):
        """
        outputs model predictions of ce_encoded as a json object
        ce_encoded (pandas dataframe): output from ce_pipe_functions
        random_prop (float): proportion of records to move randomly
        """
        # to record how many training steps the model has taken
        steps = self.sess.run(self.mainQN.global_step)
        # proportion to assign random walk
        random_prop = max(random_prop - c.annealing * steps, c.min_rand_prop)
        # a data transformation
        state = Buff.prep_state(ce_encoded)
        # these columns are relevant to calculate the amount due (except for run_time_ms)
        df = ce_encoded.loc[:, ['app_code', 'handset', 'collect', 'run_time_ms']]
        # transform number back from 100x (reverse from ce_pipe_functions)
        df.loc[:, ['handset', 'collect']] = df[['handset', 'collect']] / 100
        # dollar amount of the service plan
        plan = df.collect - df.handset
        # get model recommended action
        df['actions'] = self.get_action(state, 1, random_prop)
        # based upon action determine the amount
        df['collect'] = df.apply(helpers.figure_out_success_rewards, axis=1)
        # if amount to collect is smaller than threshold collect all instead
        convert_to_zeros = df.collect * 1.02 + c.min_amount >= df.handset
        df.loc[convert_to_zeros, 'actions'] = 0
        df.loc[convert_to_zeros, 'collect'] = df.handset[convert_to_zeros]
        # total collect amount is handset amount + service plan
        df['collect'] = df.collect + plan
        return (df.set_index('app_code')[['actions', 'collect', 'run_time_ms']]
                .to_json(orient='index', double_precision=2))

    def record_actions(self, lst, actions, new_rec=False):
        """
        record model action diagnostics
        lst (list): list to append model action diagnostics values to
        actions (iterable of int): actions taken
        new_rec (boolean): whether or not to append or start new value
        """
        unique, counts = np.unique(actions, return_counts=True)
        ratios = (counts / np.sum(counts)) / self.save_freq
        use_vals = dict(zip(unique, ratios))
        if new_rec:
            lst.append(use_vals)
        else:
            for key in use_vals:
                if key not in lst[-1]:
                    lst[-1][key] = use_vals[key]
                else:
                    lst[-1][key] = lst[-1][key] + use_vals[key]
        return lst

    def record_down(self, array, new_val, new_rec=False, use_mean=True):
        """
        record model diagnostics
        array (numpy array): array containing values
        new_val (numeric): value to record
        new_rec (boolean): whether it's a new record ornot
        use_mean (boolean): whether to divide the value as to record the mean value
        """
        if use_mean:
            divisor = self.save_freq
        else:
            divisor = 1
        use_val = new_val / divisor
        if new_rec:
            return np.append(array, use_val)
        array[-1] += use_val
        return array

    @staticmethod
    def summarize_grads(grads):
        """
        summarize gradients of NN for diagnostics
        grads (numpy array of gradients): from NN
        """
        gradients = np.concatenate([np.reshape(np.array(i[0]), -1) for i in grads])
        return np.mean(gradients), np.std(gradients)

    def plot(self, plot_path):
        """
        plot out diagnostics
        plot_path (str): location of path to store plot
        """
        f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = \
            plt.subplots(5, 2, sharex=True, sharey=False, figsize=(8, 15))
        f.suptitle(helpers.format_title(self.params), fontsize=12)
        ax1.set_title("Train Loss", fontsize=8)
        ax1.plot(self.recordings['iter_array'][:-1], self.recordings['loss_array'][:-1])
        if 'valid_array' in self.recordings:
            ax2.set_title("Validation Loss", fontsize=8)
            ax2.plot(self.recordings['iter_array'][:-1], self.recordings['valid_array'][:-1])
        ax3.set_title("mean gradients", fontsize=8)
        ax3.plot(self.recordings['iter_array'][:-1], self.recordings['grad_array'][:-1])
        ax4.set_title("gradient std", fontsize=8)
        ax4.plot(self.recordings['iter_array'][:-1], self.recordings['grad_std_array'][:-1])
        ax5.set_title("Train Rewards", fontsize=8)
        ax5.plot(self.recordings['iter_array'][:-1], self.recordings['reward_array'][:-1])
        ax6.set_title("TargetQ", fontsize=8)
        ax6.plot(self.recordings['iter_array'][:-1], self.recordings['target_array'][:-1])
        action_df = pd.DataFrame(self.recordings['action_list'])
        for i, ax in enumerate([ax7, ax8, ax9, ax10]):
            if i in action_df.columns:
                ax.set_title('proportion of action {}'.format(i), fontsize=8)
                ax.plot(self.recordings['iter_array'][:-1], action_df[i][:-1])
        f.savefig(plot_path)
        plt.close(f)

    def clean_up(self, save_mod):
        """
        clean up after model run finishes
        save_mod (boolean): whether to save the current model or not
        """
        super().clean_up(save_mod)
        if save_mod:
            helpers.write_pickle(self.mod_rec_path, self.recordings)
