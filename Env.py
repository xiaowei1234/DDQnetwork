import helpers
import constants as c
import time


class Env:
    """
    Class for translating collection_engine and card_auth data into experiences for the Buffer class
    """
    def __init__(self):
        """
        unresolved_df (pandas dataframe): current records that have not exited the environment
        dead_set (set): apps that exited the environment past the 33 days cutoff
        buffer_df (pandas dataframe): records in experience replay buffer
        """
        self.unresolved_df = helpers.load_pickle(c.unresolved_path, default=helpers.empty_df())
        self.dead_set = helpers.load_pickle(c.dead_path, default=set())
        self.buffer_df = helpers.load_pickle(c.buffer_path, default=helpers.empty_df())

    def make_failures(self, df, successes):
        """
        :return one column pandas dataframe of unique app_codes that exited the environment due to inability to collect
            the full amount after 33 days
        :param df (pandas dataframe): the piped in dataframe
        :param successes (iterable): successful app_codes in parameter df
        """
        latest_dt = df.run_date.max()
        first_dt_df = self.unresolved_df[c.identification_vars].groupby('app_code', as_index=False).min()
        latest_success = (self.unresolved_df
                          .loc[helpers.success_collect_rows(self.unresolved_df)
                               , ['app_code', 'run_date']]
                          .groupby('app_code', as_index=False).max()
                          .rename(columns={'run_date': 's_date'})
                          )
        dt_merged = first_dt_df.merge(latest_success, how='left', on='app_code')
        get_days = lambda d: d.days
        failures = dt_merged.app_code[((latest_dt - dt_merged.run_date).apply(get_days) > 33)
                                      & (~((latest_dt - dt_merged.s_date).apply(get_days) < 14))
                                      ]
        return failures[(~failures.isin(self.dead_set)) & (~failures.isin(successes))]

    def necromancer(self, fail, succeed):
        """
        creates the dead (fail) and also resurrects them (succeed)
        :param fail: app_codes that did not have a successful collection attempt before the environment cutoff
        :param succeed: app_codes that managed to have a full successful attempt
        """
        self.dead_set = self.dead_set.union(set(fail))
        self.dead_set -= set(succeed)

    def make_buffer_feed(self, df, failed):
        """
        determine whether to send rows from df to unresolved df or to buffer
        :param df: the dataframe to decide upon
        :param failed: app_codes to send to buffer because of failed collections past cutoff
        :return: tuple of app_codes for buffer and df to add to unresolved
        """
        through = df.bank_code.isin(c.success_codes)
        curr_unresolved = df.app_code.isin(self.unresolved_df.app_code)
        full = df.actions == 0
        always_success = through & (~curr_unresolved)
        to_unresolved = (~always_success) & (~df.app_code.isin(self.dead_set))
        to_buffer_codes_df_app_code = (df['app_code'][through & full & curr_unresolved]
                                       .append(failed, ignore_index=True).drop_duplicates()
                                       ).to_frame()
        to_buffer_codes_df_app_code['buff'] = 1
        return to_buffer_codes_df_app_code, df[to_unresolved]

    def make_buffer(self, df):
        """
        add in columns necessary for buffer
        df (pandas dataframe): the dataframe to add to the buffer
        """
        success = df.bank_code.isin(c.success_codes)
        df.loc[success, 'reward'] = df.loc[success, :].apply(helpers.figure_out_success_rewards, axis=1, div100=True)
        df.loc[~success, 'reward'] = df.bank_code.apply(helpers.fail_code_map)
        # chunk is created to differentiate different payment journeys for the same app
        df['chunk'] = int(time.time()) % 10000
        self.buffer_df = self.buffer_df.append(df.sort_values('run_time_ms'), ignore_index=True)

    def pipe_in(self, df):
        """
        get data and put into buffer_df for order numbers that completed journey
            otherwise put into unresolved_df
        env.buffer_df only exists as long as the environment so model will need to be
            trained same run as when new data is piped in
        """
        # if this is the first date that is piped in then just add to unresolved df
        if self.unresolved_df.shape[0] == 0:
            self.unresolved_df = df.loc[~df.bank_code.isin(c.success_codes), :]
            helpers.write_pickle(c.unresolved_path, self.unresolved_df)
            return None
        # get apps which successfully collected or timed out without success (failures)
        successes = df.app_code[helpers.success_collect_rows(df)]
        failures = self.make_failures(df, successes)
        self.necromancer(failures, successes)
        # decide whether collected data should remain unresolved or fed into buffer
        to_buffer_codes_df, add_unresolved = self.make_buffer_feed(df, failures)
        schrodinger_df = (self.unresolved_df.append(add_unresolved)
                        .merge(to_buffer_codes_df, on='app_code', how='left')
                        )
        # to_buffer_df contains payments that either completed or lapsed beyond the model cutoff date
        to_buffer_df = schrodinger_df.loc[schrodinger_df.buff == 1, :].drop('buff', axis=1)
        # unresolved are payments that haven't completed and are still within model cutoff
        self.unresolved_df = schrodinger_df.loc[~(schrodinger_df.buff == 1), :].drop('buff', axis=1)
        self.make_buffer(to_buffer_df)

    def clean_up(self):
        """
        writes out unresolved, dead, and buffer out to specified paths in constants file
        """
        helpers.write_pickle(c.unresolved_path, self.unresolved_df)
        helpers.write_pickle(c.dead_path, self.dead_set)
        helpers.write_pickle(c.buffer_path, self.buffer_df)
