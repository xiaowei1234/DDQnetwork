#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: xiaowei

pipe data into model and output actions in json
"""

from pipe_ce import ce_pipe_functions
import constants as c
from Model import Model


def remove_control(df):
    """
    remove a portion from recommended actions to serve as control
    """
    make_control = df.app_code.apply(lambda v: 'W' in v)
    return df.loc[~make_control, :]


def pipe_create_predictions(adict):
    """
    takes raw data and outputs model recommended actions in a json file
    adict (dict): raw data
    """
    # loads TF package, instantiates tensorflow DAG. This line should only run once
    mod = Model(c.mod_feat_num, c.action_nums, c.params, c.mod_path, c.mod_records_path, False)
    # cleans and apply transformations to the data to prep for prediction. This needs to run everytime
    ce_encoded = ce_pipe_functions(adict=adict).drop('ranks', axis=1).pipe(remove_control)
    # calls method to create predictions. This needs to run everytime
    json_obj = mod.output_records_json(ce_encoded, c.random_prop)
    return json_obj
