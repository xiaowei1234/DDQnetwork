#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: xiaowei
"""
import pandas as pd
import numpy as np
import re
import pickle
import os
import constants as c
from datetime import date


def empty_df():
    return pd.DataFrame([])


def run_ms_to_ts(epoch):
    return pd.to_datetime(epoch, unit='ms')# - pd.Timedelta(hours=7)


float_int = lambda series: (series.fillna(0) * 100).astype(np.int32)

is_iter = lambda thing: hasattr(thing, '__iter__') and not isinstance(thing, str)

make_list = lambda thing: thing if is_iter(thing) else [thing]

bool_conv = lambda v: 1 if v == True or (isinstance(v, str) and v.lower() == 'true') else 0

has_one_ele = lambda v: hasattr(v, '__iter__') and len(v) > 0

float_comp = lambda chr: bool(re.match(r"^[0-9]*\.[0-9]+$", chr)) if isinstance(chr, str) else False

int_comp = lambda chr: bool(re.match("^[0-9]+$", chr)) if isinstance(chr, str) else False

float_conv = lambda chr: np.float64(chr) if float_comp(chr) or int_comp(chr) or isinstance(chr, float) else np.nan


def action_int(v):
    try:
        return np.int32(v)
    except:
        return 0
    
def fail_code_map(code, fail_dict=c.failure_code_dict):
    if code in fail_dict:
        return fail_dict[code]
    return -1

def has_something(v):
    """
    returns True if v has some kind of value other than empty iterable or zero len str
    """
    if isinstance(v, str):
        return len(v.strip()) > 0
    if has_one_ele(v):
        return True
    if pd.isnull(v):
        return False
    if isinstance(v, (int, float)):
        return pd.notnull(v)


def cell_wrapper(df, func, field, drop=True, new_name=None):
    """
    decorator function for pandas pipe api
    takes func which applies function to one value in field
    returns modified dataframe
    """
    lst = list(df[field].apply(func))
    if isinstance(lst[0], dict):
        adf = pd.DataFrame(lst, index=df.index)
    else:
        adf = pd.DataFrame(lst, index=df.index, columns=[field])
    if new_name is not None:
        adf = adf.rename(columns={field: new_name})
    if drop:
        return pd.concat([df.drop(field, axis=1), adf], axis=1)
    return pd.concat([df, adf], axis=1)


def df_wrapper(df, func):
    """
    decorator function for pandas dataframe pipe api
    takes function that transforms df and then concats by column and returns both original and transformed
    """
    new_df = func(df)
    return pd.concat([df, new_df], axis=1)


def load_pickle(filepath, default=None):
    """
    loads pickled file specified by filepath or default if not found
    """
    if not os.path.isfile(filepath):
        return default
    if isinstance(default, pd.core.frame.DataFrame):
        return pd.read_pickle(filepath)
    with open(filepath, 'rb') as pfile:
        return pickle.load(pfile)


def write_pickle(filepath, thing):
    """
    write out thing to filepath
    """
    if isinstance(thing, pd.core.frame.DataFrame):
        thing.to_pickle(filepath)
    else:
        with open(filepath, 'wb') as pfile:
            pickle.dump(thing, pfile)


def sklearn_decorator(func, path, *args, **kwargs):
    """
    scikit learn ML fit transform decorator to fit into pandas dataframe pipe api
    """
    if c.create_new:
        transformer = None
    else:
        transformer = load_pickle(path)
    trans_df, transformer_ret = func(*args, **kwargs, trans=transformer)
    if transformer is None:
        write_pickle(path, transformer_ret)
    return trans_df


def sklearn_func(df, sk_func, trans=None):
    """
    similar to sklearn_decorator except no loading from path
    """
    if trans is None:
        trans = sk_func
        trans.fit(df)
    trans_array = trans.transform(df)
    return trans_array, trans


def index_series(df, name):
    return pd.Series(df.index.get_level_values(name))


def reshape_2d(array):
    """
    flatten to matrix with 1 column
    """
    return np.reshape(array, (-1, 1))


def memoize(func):
    dic = {}
    def helper(x, *args, **kwargs):
        if x not in dic:
            dic[x] = func(x, *args, **kwargs)
        return dic[x]
    return helper


def get_dir_from_path(path):
        sep = os.path.sep
        lst = path.split(sep)
        return sep.join(lst[:-1]) + sep


def make_folder(path):
    if os.path.exists(path):
        return
    if path.find('.') > 0:
        path = get_dir_from_path(path)
        make_folder(path)
    elif len(path) > 1:
        os.mkdir(path)


def get_plot_path(path=c.plot_path):
    dt = str(date.today())
    folder = path + dt
    make_folder(folder)
    return folder + os.path.sep


def success_collect_rows(df):
    return df.bank_code.isin(c.success_codes) & (df.actions != c.zero_action)


def figure_out_success_rewards(row):
    """
    given action nums and rewards figure out the correct amount to collect
    row (pandas series): one row in dataframe with actions and handset values
    reward_tups: action rewards if success
    """
    act = c.action_nums[row.actions]
    if act < 1.01:
        return row.handset * act #* 0.01
    return act


def format_title(dic):
    """
    model diagnostics title formatting for plot
    """
    line = str(dic)[1:-1]
    mid = int(len(line)/2) + 1
    return line[:mid] + '\n' + line[mid:]


def check_tf_model_exists(path):
    """
    checks if a tensorflow model exists
    TF model save is different from most
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    files = [f for f in os.listdir(path) if '.ckpt' in f]
    return len(files) > 0
