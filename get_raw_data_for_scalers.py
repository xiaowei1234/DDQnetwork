#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 10:14:37 2018

@author: xiaowei

This script is run prior to creation of the initial model and not used in production.
    It is used to create the data
    that sklearn's standardscaler and pandas' get_dummies use to create the dataframe 
    that gets producted at the end of pipe_ce.py
"""

import pandas as pd
import constants as c
import clean_mongo as cm
import helpers
from get_collection_engine import get_input_date, get_ce_day
from pipe_ce import encode_df


def ce_pipe_functions(ts_ms, limit):
    df_piped = (get_ce_day(ts_ms, limit)
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
                .pipe(cm.make_ranks, grpby=['app_code', 'collect']
                    , rankby='run_time_ms')
                .drop(c.dict_fields + c.final_drop, axis=1)
                .set_index(c.identification_vars + ['ranks']).sort_index()
                    )
    return df_piped


if __name__ == '__main__':
#    ssh -L 9998:10.128.1.43:27017 xwei@10.128.1.27
    ts_ms = get_input_date('2018-08-01')
    lst_df = []
    for ts in range(ts_ms, 1541055600000, 24 * 3600 * 1000 * 12):
        print (ts)
        ce_clean = ce_pipe_functions(ts, limit=None)
        lst_df.append(ce_clean)
    standard_df = pd.concat(lst_df)
    helpers.write_pickle(c.data_path + 'norm_setter.pkl', standard_df)
    encoded_df = encode_df(standard_df)
    
