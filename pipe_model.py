import sys
import pandas as pd
from pipe_ce import ce_pipe_functions
from pipe_ca import ca_pipe_functions
from get_collection_engine import get_input_date
import constants as c
#import helpers
from Env import Env
from datetime import timedelta


def merge_sources(ce_df, ca_df):
    zeros = ce_df['actions'] == c.zero_action
    zero_df = ce_df.loc[zeros, :]
    zero_df['bank_code'] = '99'
    joined_df = (ce_df.loc[~zeros, :].merge(ca_df.drop('run_date', axis=1), how='inner'
                      , on=['app_code', 'collect', 'ranks'])
                )
    return pd.concat([joined_df, zero_df]).drop('ranks', axis=1)


def get_one_day(dt, limit):
    ts = get_input_date(dt)
    print ('Piping Collection Engine data ' + dt)
    ce_encoded = ce_pipe_functions(ts, limit)
    print ('Piping Card Auth data')
    ca_clean = ca_pipe_functions(ts, limit)
    print ('Merging data sources')
    merged_df = merge_sources(ce_encoded, ca_clean)
    if limit is None:
        merged_df.to_pickle(c.repo_path + c.save_data_prefix + dt + '.pkl')
        print ('Piping into environment')
        env = Env()
        print ('unresolved: ', env.unresolved_df.shape, '\n', 'dead: ', len(env.dead_set), '\n'
           , 'buffer: ', env.buffer_df.shape)
        env.pipe_in(merged_df)
        print ('unresolved: ', env.unresolved_df.shape, '\n', 'dead: ', len(env.dead_set), '\n'
           , 'buffer: ', env.buffer_df.shape)
        env.clean_up()
        print ('piping finished')


if __name__ == '__main__':
#    ssh -L 9998:10.128.1.43:27017 xwei@10.128.1.27
    start = sys.argv[1]
    num_days = int(sys.argv[2])
    limit = None if len(sys.argv) < 4 else int(sys.argv[3])
    for day in range(num_days):
        dt = str(pd.to_datetime(start, infer_datetime_format=True) + timedelta(days=day))[:10]
        get_one_day(dt, limit)
