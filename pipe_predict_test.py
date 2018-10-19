#!/usr/bin/env python3

#import pandas as pd
import sys
import bson.json_util as json_util
import numpy as np
import constants as c
import helpers
from pipe_ce import ce_pipe_functions
# chmod +x pipe_predict_test.py
# df in code refers to a pandas dataframe which is a two dimensional representation of a table of data


def output_records_json(ce_encoded, random_prop):
    """
    this is a close replica of the same class function in Model class of Model.py
        difference is that instead of picking the optimal action using the model is defaults to action 0
        which is a full collection attempt
    ce_encoded (pandas dataframe): output of ce_pipe_functions which is the fully numerical cleaned up
        version of collection_engine data
    random_prop (float): proportion of random walks
    """
    # only relevant columns
    df = ce_encoded.loc[:, ['app_code', 'handset', 'collect', 'run_time_ms']]
    df.loc[:, ['handset', 'collect']] = df[['handset', 'collect']] / 100
    # get plan $ amount
    plan = df.collect - df.handset
    # make random action proportions
    num_actions = len(c.action_nums)
    each_rand_action = list(np.repeat(random_prop / float(num_actions - 1), num_actions - 1))
    probs = [1 - random_prop] + each_rand_action
    # assign random actions
    df['actions'] = np.random.choice(c.action_lst, size=df.shape[0], p=probs)
    df['collect'] = df.apply(helpers.figure_out_success_rewards, axis=1)
    # if amount is close to or over the full amount set to full amount
    convert_to_zeros = df.collect * 1.02 + c.min_amount >= df.handset
    df.loc[convert_to_zeros, 'actions'] = 0
    df.loc[convert_to_zeros, 'collect'] = df.handset[convert_to_zeros]
    # add the plan amount back in
    df['collect'] = df.collect + plan
    return (df.set_index('app_code')[['actions', 'collect', 'run_time_ms']]
            .to_json(orient='index', double_precision=2))


if __name__ == '__main__':
    # read in json str and make json object
    json_str = sys.stdin.read()
    json_obj = json_util.loads(json_str)
    # make dataframe to point where it could be loaded into model even though we don't need it
    piped_df = ce_pipe_functions(json=json_obj)
    # make json
    json_records = output_records_json(piped_df, 0.1)
    json_records_str = json_util.dumps(json_records)
    sys.stdout.write(json_records_str)
    sys.stdout.flush()
