# piping card auth
#import pandas as pd
import sys
import constants as c
import clean_mongo as cm
import helpers
from get_card_auth import get_ca_day
#import numpy as np
    

def ca_pipe_functions(ts, limit=None):
    """
    """
    df_piped = (get_ca_day(ts, limit)
                .pipe(cm.manual_remove, keep=c.card_auth_keep)
                .pipe(cm.rename_cols, c.card_auth_renames)
                .pipe(cm.convert_to_float, field='amount_collect')
                .pipe(cm.make_run_date)
                )
    df_piped['collect'] = helpers.float_int(df_piped['amount_collect'])
    return (df_piped.loc[df_piped.amount_collect > 0, :]
            .pipe(cm.make_ranks, grpby=['app_code', 'collect']
            , rankby='run_time_ms')
            .drop(['amount_collect', 'run_time_ms'], axis=1)
            .reset_index()
            )

if __name__ == '__main__':
    df = helpers.read_pickle(c.repo_path + 'card_auth_' + sys.argv[1] + '.pkl')
    df_clean = ca_pipe_functions(df)
    df_clean.to_pickle(c.repo_path + 'card_auth_clean_' + sys.argv[1] + '.pkl')