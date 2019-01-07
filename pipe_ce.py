# piping data into model
import pandas as pd
import sys
import constants as c
import clean_mongo as cm
import helpers
from functools import partial
from sklearn.preprocessing import StandardScaler
from get_collection_engine import get_input_date, get_ce_day

# wrapper for standardscaler
norm_func = partial(helpers.sklearn_func, sk_func=StandardScaler())

# wrapper to fit into pandas pipe API
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
    """
    takes a df and ensures it has the requisite columns and also scales numerical data
    df (dataframe): dataframe with unstandardized and also categorical columns
    """
    dummy_df = get_right_columns(pd.get_dummies(df), c.col_path)
    norm_mat = norm_pipe(dummy_df.fillna(0))
    norm_df = pd.DataFrame(norm_mat, columns=dummy_df.columns, index=df.index)
    return norm_df


def ce_pipe_functions(ts_ms=0, limit=None, adict=None, raw_df=None):
    """
    Piping functions for collection engine data
    if json is none it assumes one day worth of data piped from collection_engine mongodb
        otherwise json is assumed to be a raw json object
    ts_ms (int): epoch in milliseconds for start of day
    adict (dict): dictionary that can be formed into a dataframe
    limit (int): max # of records to get. None for no limit
    json (json object): json object to use as data if not pulling from collection_engine
    """
    if raw_df is not None:
        pass
    elif adict:
        raw_df = pd.DataFrame(data=adict).rename(columns=c.emit_dict)
    else:
        raw_df = get_ce_day(ts_ms, limit)
#    raw_df.iloc[:1000].to_csv('/Users/xiaowei/Desktop/raw_df.csv', index=False)
    df_piped = (raw_df
                .pipe(cm.manual_remove)
                .pipe(cm.break_outs_df)
                .pipe(cm.install_counts)
                .pipe(cm.make_date_cols)
                .pipe(cm.payment_dates)
                .pipe(cm.make_payments)
                .pipe(cm.days_to_next_payment)
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
    """
    main only used for debugging
    """
    # ssh -L 9998:10.128.1.43:27017 xwei@10.128.1.27
    # latest piped in date is 2018-07-16
    ts_ms = get_input_date(sys.argv[1])
    print ('\nPiping in Collection Engine: ' + sys.argv[1])
    limit = None if len(sys.argv) < 3 else int(sys.argv[2])
    print ('Cleaning Collection Engine data')
    ce_encoded = ce_pipe_functions(ts_ms, limit)
