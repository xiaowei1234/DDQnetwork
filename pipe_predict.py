import sys
from pipe_ce import ce_pipe_functions
import constants as c
from Model import Model
from Buffer import Buff


def remove_control(df):
    make_control = df.app_code.apply(lambda v: 'W' in v)
    return df.loc[~make_control, :]


if __name__ == '__main__':
    with sys.stdin as f:
        json = f.read()
    print ('\nPiping in Collection Engine: ')
    ce_encoded = ce_pipe_functions(json=json).drop('ranks', axis=1).pipe(remove_control)
    print ('Model prediction start')
    state_df = Buff.prep_state(ce_encoded, c.identification_vars)
    mod = Model(state_df.shape[1], c.action_nums, c.params, c.mod_path, c.mod_records_path, False)
    json_str = mod.output_records_json(ce_encoded, c.random_prop)
    sys.stdout.write(json_str)
