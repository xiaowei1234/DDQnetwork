# piping data into model
#import os
import pandas as pd
import sys
import constants as c
import clean_mongo as cm
import helpers
from functools import partial
from sklearn.preprocessing import StandardScaler
from get_collection_engine import get_input_date, get_ce_day
from pandas.io.json import json_normalize
#import bson.json_util as json_util


norm_func = partial(helpers.sklearn_func, sk_func=StandardScaler())

norm_pipe = partial(helpers.sklearn_decorator, norm_func, c.norm_path)


def get_right_columns(df, saved_path):
    """
    this ensures the right columns are passed into the model
        columns not found at saved_path are added with value 0
        extra columns are dropped
    df (pandas dataframe): the df in question
    saved_path (str): location of saved file
    """
    prev_cols = helpers.load_pickle(saved_path, default=None)
    if prev_cols is None:
        prev_cols = df.columns
        helpers.write_pickle(saved_path, prev_cols)
    return df.reindex(columns=prev_cols, fill_value=0)


def encode_df(df):
    dummy_df = get_right_columns(pd.get_dummies(df), c.col_path)
#    df2 = pd.concat([norm_df, dummy_df], axis=1)
    norm_mat = norm_pipe(dummy_df.fillna(0))
    norm_df = pd.DataFrame(norm_mat, columns=dummy_df.columns, index=df.index)
    return norm_df


def ce_pipe_functions(ts_ms=0, limit=None, json=None):
    """
    Piping functions for collection engine data
    if json is none it assumes one day worth of data piped from collection_engine mongodb
        otherwise json is assumed to be a raw json object
    ts_ms (int): epoch in milliseconds for start of day
    limit (int): max # of records to get. None for no limit
    json (json object): json object to use as data if not pulling from collection_engine
    """
    if json:
        raw_df = json_normalize(json)
    else:
        raw_df = get_ce_day(ts_ms, limit) 
    df_piped = (raw_df
                .pipe(cm.manual_remove)
                .pipe(cm.break_outs_df)
                .pipe(cm.install_counts)
                .pipe(cm.make_date_cols)
                .pipe(cm.payment_dates)
                .pipe(cm.make_payments)
                .pipe(cm.days_to_next_payment)
                # .pipe(cm.make_contains_var)
                .pipe(cm.convert_bool)
                .pipe(cm.convert_to_float, field='amount')
                .pipe(cm.convert_to_float, field='total_amount')
                .pipe(cm.rename_cols, c.collection_engine_renames)
                .pipe(cm.collect_columns)
                .pipe(cm.make_ranks, grpby=['app_code', 'collect'], rankby='run_time_ms')
                .drop(c.dict_fields + c.final_drop, axis=1)
                .set_index(c.identification_vars + ['ranks']).sort_index()
                .pipe(encode_df)
                )
    return df_piped.reset_index()


if __name__ == '__main__':
    # ssh -L 9998:10.128.1.43:27017 xwei@10.128.1.27
    # latest piped in date is 2018-07-16
    ts_ms = get_input_date(sys.argv[1])
    print ('\nPiping in Collection Engine: ' + sys.argv[1])
    limit = None if len(sys.argv) < 3 else int(sys.argv[2])
    print ('Cleaning Collection Engine data')
    ce_encoded = ce_pipe_functions(ts_ms, limit)


#from collections import defaultdict
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder


#def ddle_func(df, trans=None):
#    if trans is None:
#        trans = defaultdict(LabelEncoder)
#        df.apply(lambda col: trans[col.name].fit(col))
#    df_trans = df.apply(lambda col: trans[col.name].transform(col))
#    return df_trans, trans


#ddle_pipe = partial(sklearn_decorator, ddle_func, c.ddle_path)

#ohe_func = partial(sklearn_func, sk_func=OneHotEncoder(sparse=False, handle_unknown='ignore'))

#ohe_pipe = partial(sklearn_decorator, ohe_func, c.ohe_path)
    


#def encode_df(df):
#    obj_df = df.select_dtypes('object').fillna('NA')
#    ohe_array = ohe_pipe(ddle_pipe(obj_df))
#    norm_mat = norm_pipe(df.select_dtypes(exclude='object').fillna(0).values)/100
#    mat = np.concatenate((norm_mat, ohe_array), axis=1)
#    colnames = list(df.select_dtypes(exclude='object').columns) + \
#        [l for l in range(ohe_array.shape[1])]
#    return pd.DataFrame(mat, index=df.index, columns=colnames)