import pandas as pd
import numpy as np


def lower_case_cols(df):
    # change all column names to lowercase and replace spaces with underscore
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def convert_rows_to_lower(df):
    df = df.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)
    return df

def convert_date_cols(df):
    # convert all date columns to date (currently objects)
    date_cols = ['effective_date_endorsement', 'claim_date_of_loss', 'policy_transaction_date',\
                 'policy_start_date', 'policy_end_date', 'original_inception_date', 'term_inception_date',\
                 'expiration_date', 'effective_date_new_business', 'effective_date_renewal', 'effective_date_cancel']
    for col in df.columns:
        if col in date_cols:
            df[col] = pd.to_datetime(df[col], format='%m/%d/%Y')
    return df

def calculate_datediff(df, colname, col1, col2):
    # create new column with the number of days between two columns
    df[colname] = (df[col1] - df[col2]).dt.days.astype(int)
    return df

def convert_to_int(df, cols_list):    
    for col in df.columns:
        if col in cols_list:
            df[col] = df[col].astype(int)
    return df

def create_claim_type(df):
    claim_df = df[~df['claim_date_of_loss'].isnull()]
    claim_df['transaction_type'] = 'claim'
    
    other_df = df[df['claim_date_of_loss'].isnull()]
    
    output_df = pd.concat([claim_df, other_df], axis=0)
    
    output_df = output_df.sort_values(by=['policy_id','policy_transaction_date'])
    
    return output_df


def set_endorsement_flag(df):
    endorsement_range = 7
    # get the difference in days between endorsement date and policy start date
    df['diff_effective_date_policy_start'] = (df['effective_date_endorsement'] - df['policy_start_date']).dt.days.astype('Int64')
    
    # filter dataset down to only the policy_id and diff_effective_date_policy_start, and get the min per policy_id
    endorse_df = df[~df['diff_effective_date_policy_start'].isnull()]\
                    .groupby(['policy_id'])\
                    .diff_effective_date_policy_start.min()\
                    .reset_index()
    
    # filter data down to only those policies that had a change in the date range
    endorse_df = endorse_df[endorse_df['diff_effective_date_policy_start'] <= endorsement_range]
    endorse_df['has_endorsement_first_7'] = 1
    
    endorse_df = endorse_df.drop('diff_effective_date_policy_start', axis=1)
    
    # merge with df, then return
    df = pd.merge(df, endorse_df, on = 'policy_id', how='left')
    df['has_endorsement_first_7'].fillna(0, inplace=True)
    
    del endorse_df
    
    return df

def set_claim_near_start_flag(df):
    claim_range = 14
    # get the difference in days between claim date and policy start date
    df['diff_claim_date_policy_start'] = (df['claim_date_of_loss'] - df['policy_start_date']).dt.days.astype('Int64')
    
    # filter dataset down to only the policy_id and diff_claim_date_policy_start, and get the min per policy_id
    claim_df = df[~df['diff_claim_date_policy_start'].isnull()]\
                    .groupby(['policy_id'])\
                    .diff_claim_date_policy_start.min()\
                    .reset_index()
    
    # filter data down to only those policies that had a claim in the date range
    claim_df = claim_df[claim_df['diff_claim_date_policy_start'] <= claim_range]
    claim_df['has_claim_first_14'] = 1
    
    claim_df = claim_df.drop('diff_claim_date_policy_start', axis=1)
    
    # merge with df, then return
    df = pd.merge(df, claim_df, on = 'policy_id', how='left')
    df['has_claim_first_14'].fillna(0, inplace=True)
    
    df = df.drop('diff_claim_date_policy_start', axis=1)
    
    del claim_df
    
    return df


def set_claim_near_end_flag(df):
    claim_range = 14
    # get the difference in days between claim date and policy end date
    df['diff_claim_date_policy_end'] = (df['policy_end_date'] - df['claim_date_of_loss']).dt.days.astype('Int64')
    
    # filter dataset down to only the policy_id and diff_claim_date_policy_end, and get the max per policy_id
    claim_df = df[~df['diff_claim_date_policy_end'].isnull()]\
                    .groupby(['policy_id'])\
                    .diff_claim_date_policy_end.max()\
                    .reset_index()
    
    # filter data down to only those policies that had a claim in the date range
    claim_df = claim_df[claim_df['diff_claim_date_policy_end'] <= claim_range]
    claim_df['has_claim_last_14'] = 1
    
    claim_df = claim_df.drop('diff_claim_date_policy_end', axis=1)
    
    # merge with df, then return
    df = pd.merge(df, claim_df, on = 'policy_id', how='left')
    df['has_claim_last_14'].fillna(0, inplace=True)
    
    df = df.drop('diff_claim_date_policy_end', axis=1)
    
    del claim_df
    
    return df

def create_territory_buckets(df):
    num_policies = df.policy_id.nunique()

    policy_agency = df[['policy_id','territory_code']].drop_duplicates()

    territory_df = policy_agency.territory_code.value_counts()\
                        .reset_index()\
                        .rename(columns={'index':'territory_code','territory_code':'counts'})

    territory_df['percent_vol'] = territory_df.counts / num_policies

    territory_df['decile'] = pd.cut(territory_df['percent_vol'], 5, labels=np.arange(5, 0, -1))

    territory_df['territory_group'] = territory_df.decile.apply(lambda x: 'decile_' + str(x))

    territory_df = territory_df.drop(['counts','percent_vol','decile'],axis=1)

    df = df.merge(territory_df, on='territory_code', how='left')
    
    #df = df.drop('territory_code', axis = 1)
    
    del policy_agency, territory_df
    
    return df

def create_age_bucket(val):
    if val < 30:
        return 'less_than_30'
    elif val >= 30 and val < 40:
        return '30_to_40'
    elif val >= 40 and val < 50:
        return '40_to_50'
    elif val >= 50 and val < 60:
        return '50_to_60'
    elif val >= 60 and val < 70:
        return '60_to_70'
    else:
        return 'greater_than_70'


def create_city_bucket(val):
    med_city_list = ['hialeah', 'homestead', 'miami gardens', 'hollywood', 'opa locka',\
                     'pembroke pines', 'miramar', 'fort lauderdale', 'coral springs']
    
    if val == 'miami':
        return 'miami'
    elif val in med_city_list:
        return 'medium_vol_city'
    else:
        return 'low_vol_city'

def has_garaging_address_2(df):
    address_df = df[~df['garaging_address_2'].isnull()][['policy_id']].drop_duplicates().reset_index()
    
    address_df['has_garaging_address_2'] = 1
    
    out_df = pd.merge(df, address_df, on='policy_id', how='left')
    
    return out_df

def create_prior_insurance_group(val):
    if val == 'windhaven':
        return 'windhaven'
    elif val == 'infinity':
        return 'infinity'
    elif val == 'united auto':
        return 'united_auto'
    elif val == 'responsive auto':
        return 'responsive_auto'
    elif val == 'ocean harbor':
        return 'ocean_harbor'
    elif val == 'progressive':
        return 'progressive'
    else:
        return 'other'

def create_agency_group(df):
    num_policies = df.policy_id.nunique()

    agency_df = df[['policy_id','agency_code']].drop_duplicates()

    agency_df = agency_df.agency_code.value_counts()\
                        .reset_index()\
                        .rename(columns={'index':'agency_code','agency_code':'counts'})

    agency_df['percent_vol'] = agency_df.counts / num_policies

    agency_df['decile'] = pd.cut(agency_df['percent_vol'], 5, labels=np.arange(5, 0, -1))

    agency_df['agency_code_group'] = agency_df.decile.apply(lambda x: 'decile_' + str(x))

    agency_df = agency_df.drop(['counts','percent_vol','decile'],axis=1)

    df = df.merge(agency_df, on='agency_code', how='left')
    
    #df = df.drop('agency_code', axis=1)
    
    del agency_df
    
    return df

def check_integer_val(val):
    if val >= 0:
        return 1
    else:
        return 0

    # create a count of endorsements prior to the date of claim date of loss
def create_min_endorsement_count(df):
    # first get the distinct policy_id and minimum claim_date_of_loss
    claim_df = df[(df['transaction_type'] == 'claim') & (df['has_claim_first_14'] == 1)]\
                    [['policy_id','claim_date_of_loss']]\
                        .groupby(['policy_id'])\
                        .claim_date_of_loss.min()\
                        .reset_index()
    
    # for each policy_id, get dataframe of all distinct endorsements (based on effective_date_endorsement), which is less than 
    # the min claim date
    endorse_df = df[~df['effective_date_endorsement'].isnull()][['policy_id','effective_date_endorsement']].drop_duplicates()
    
    joined_df = pd.merge(endorse_df, claim_df, on='policy_id', how='inner')
    
    joined_df['diff_days'] = (joined_df['claim_date_of_loss'] - joined_df['effective_date_endorsement']).dt.days.astype('Int64')
    
    joined_df['min_endorse_count'] = joined_df.diff_days.apply(check_integer_val)
    
    out_df = joined_df.groupby(['policy_id']).min_endorse_count.sum().reset_index()
    
    out_df = pd.merge(df, out_df, on='policy_id', how='left')
    
    return out_df


# create a count of endorsements prior to the date of claim date of loss
def create_max_endorsement_count(df):
    # first get the distinct policy_id and minimum claim_date_of_loss
    claim_df = df[(df['transaction_type'] == 'claim') & (df['has_claim_last_14'] == 1)]\
                    [['policy_id','claim_date_of_loss']]\
                        .groupby(['policy_id'])\
                        .claim_date_of_loss.max()\
                        .reset_index()
    
    # for each policy_id, get dataframe of all distinct endorsements (based on effective_date_endorsement), which is less than 
    # the min claim date
    endorse_df = df[~df['effective_date_endorsement'].isnull()][['policy_id','effective_date_endorsement']].drop_duplicates()
    
    joined_df = pd.merge(endorse_df, claim_df, on='policy_id', how='inner')
    
    joined_df['diff_days'] = (joined_df['claim_date_of_loss'] - joined_df['effective_date_endorsement']).dt.days.astype('Int64')
    
    joined_df['max_endorse_count'] = joined_df.diff_days.apply(check_integer_val)
    
    out_df = joined_df.groupby(['policy_id']).max_endorse_count.sum().reset_index()
    
    out_df = pd.merge(df, out_df, on='policy_id', how='left')
    
    return out_df

# create feature for count of distinct garaging addresses
def count_addresses(df):
    garaging_df = df.groupby(['policy_id'])\
                    .agg({'garaging_address': lambda x: x.nunique()})\
                    .reset_index()\
                    .rename(columns={'garaging_address': 'garaging_address_count'})
    
    return pd.merge(df, garaging_df, on = 'policy_id', how='inner') 


def filter_data(df):
    trans_types = ['new business', 'renewal']
    cancel_list = df[(df['transaction_type'] == 'cancel') &\
                     (df['policy_transaction_date'] == df['policy_start_date'])]\
                        .policy_id.unique()
    
    update_df = df[~df['policy_id'].isin(cancel_list)]
    
    return update_df[(update_df['transaction_type'].isin(trans_types))\
                   & (update_df['claim_date_of_loss'].isnull())\
                   & (update_df['policy_start_date'] != update_df['policy_end_date'])].sort_values('policy_id')


def get_policy_cost(written_total, written_incr):
    if written_incr == 0:
        return written_total
    elif written_total == written_incr:
        return written_total
    elif written_incr > 0:
        return written_incr
    

def agg_vals(df, cols):
    
    output_df = df[cols]
    
    output_df = output_df.groupby(['policy_id'], as_index=False).first()
    
    return output_df

def to_dummies(df, cols):
    return pd.get_dummies(df, prefix=cols, columns=cols, dtype=int)


def preprocess(raw_file):
    df = pd.read_csv(raw_file)

    df = df.drop_duplicates()
    
    df = df.fillna(0)
    
    df = lower_case_cols(df)
    
    df = convert_rows_to_lower(df)
    
    df = convert_date_cols(df)
    
    df = create_claim_type(df)
    
    df = set_endorsement_flag(df)
    
    df = set_claim_near_start_flag(df)
    
    df = set_claim_near_end_flag(df)
    
    # create new features
    df = create_territory_buckets(df)
    
    df['age_bucket'] = df.driver_one_age.apply(create_age_bucket)
    
    df['prior_insurance_group'] = df.prior_insurance_company.apply(create_prior_insurance_group)
    
    df = create_agency_group(df)
    
    df['garaging_city_bucket'] = df.garaging_city.apply(create_city_bucket)
    
    df = has_garaging_address_2(df)

    return df


def create_claim_features(df):
    df = create_min_endorsement_count(df)
    
    df = create_max_endorsement_count(df)
    
    df = count_addresses(df)

    return df


def filter_dataframe(df):
    # applies to both claims dataframe and endoresement dataframe
    # filter data down to the first row per policy_ids
    df = filter_data(df)
    
    # get the original policy amount
    df['written_premium_amt'] = df.apply(lambda x: get_policy_cost(x.written_premium_total, x.written_premium_incr), axis=1)
    
    # create date split cols
    df['policy_start_dow'] = df['policy_start_date'].dt.weekday_name
    df['policy_start_month'] = df['policy_start_date'].dt.month_name(locale = 'English')

    return df

def aggregate_claims_dataset(df):
    agg_cols = ['policy_id','transaction_type','policy_start_dow','policy_start_month','policy_coverages_desc',\
                    'policy_coverages_count','flag_transfer_discount','rating_program','term','term_length','is_renewal',\
                    'prior_insurance_limits','was_reinstated','flagged_for_non-renewal','non-renewal_reason',\
                    'payplan_description','payplan_autopay_required','has_garaging_address_2','has_endorsement_first_7',\
                    'autopay_enrolled','agency_type','commission_type','signing_method','policy_drivers','policy_drivers_excluded',\
                    'policy_drivers_unlicensed','policy_drivers_minor','flag_excluded_driver','flag_unlicensed','flag_minor',\
                    'driver_one_gender','driver_one_marital_status','policy_vehicles','usage_index_min','usage_index_max',\
                    'usage_index_avg','territory_group','age_bucket','prior_insurance_group','agency_code_group',
                    'garaging_city_bucket','written_premium_amt','min_endorse_count','max_endorse_count','garaging_address_count',\
                    'has_claim_first_14','has_claim_last_14']
    df = agg_vals(df, agg_cols)

    dummies_col_list = ['transaction_type','policy_start_dow','policy_start_month','policy_coverages_desc','rating_program',\
               'term','term_length','prior_insurance_limits','payplan_description','agency_type','commission_type',\
               'signing_method','driver_one_gender','driver_one_marital_status','territory_group','age_bucket',\
               'prior_insurance_group','agency_code_group','garaging_city_bucket','non-renewal_reason']
    
    df = to_dummies(df, dummies_col_list)

    cols_to_int_list = ['policy_coverages_count','flag_transfer_discount','is_renewal','was_reinstated','flagged_for_non-renewal',\
                'payplan_autopay_required','autopay_enrolled','policy_drivers','policy_drivers_excluded','policy_drivers_unlicensed',\
                'policy_drivers_minor','flag_excluded_driver','flag_unlicensed','flag_minor','policy_vehicles','usage_index_min',\
                'usage_index_max','usage_index_avg','has_endorsement_first_7','written_premium_amt','garaging_address_count',\
                'max_endorse_count','min_endorse_count','has_claim_last_14','has_claim_first_14','has_garaging_address_2',
                'non-renewal_reason']
    
    df = df.fillna(0)
    df = convert_to_int(df, cols_to_int_list)
    
    
    col_list = df.columns[df.dtypes.isin(['int64','uint8'])]
    df = convert_to_int(df, col_list)
    df = df.fillna(0)
    
    return df


def aggregate_endorsement_dataset(df):
    agg_cols = ['policy_id','transaction_type','policy_start_dow','policy_start_month','policy_coverages_desc',\
                 'policy_coverages_count','flag_transfer_discount','rating_program','term','term_length','is_renewal',\
                 'prior_insurance_limits','was_reinstated','flagged_for_non-renewal','payplan_description','payplan_autopay_required',\
                 'autopay_enrolled','agency_type','commission_type','signing_method','policy_drivers','policy_drivers_excluded',\
                 'policy_drivers_unlicensed','policy_drivers_minor','flag_excluded_driver','flag_unlicensed','flag_minor',\
                 'driver_one_gender','driver_one_marital_status','policy_vehicles','usage_index_min','usage_index_max',\
                 'usage_index_avg','has_endorsement_first_7','territory_group','age_bucket','prior_insurance_group','agency_code_group',
                 'garaging_city_bucket','written_premium_amt','has_garaging_address_2']

    df = agg_vals(df, agg_cols)

    dummies_col_list = ['transaction_type','policy_start_dow','policy_start_month','policy_coverages_desc','rating_program',\
               'term','term_length','prior_insurance_limits','payplan_description','agency_type','commission_type',\
               'signing_method','driver_one_gender','driver_one_marital_status','territory_group','age_bucket',\
               'prior_insurance_group','agency_code_group','garaging_city_bucket']
    
    df = to_dummies(df, dummies_col_list)

    cols_to_int_list = ['policy_coverages_count','flag_transfer_discount','is_renewal','was_reinstated','flagged_for_non-renewal',\
                'payplan_autopay_required','autopay_enrolled','policy_drivers','policy_drivers_excluded','policy_drivers_unlicensed',\
                'policy_drivers_minor','flag_excluded_driver','flag_unlicensed','flag_minor','policy_vehicles','usage_index_min',\
                'usage_index_max','usage_index_avg','has_endorsement_first_7','written_premium_amt']
    
    df = df.fillna(0)
    df = convert_to_int(df, cols_to_int_list)
    
    
    col_list = df.columns[df.dtypes.isin(['int64','uint8'])]
    df = convert_to_int(df, col_list)
    df = df.fillna(0)
    return df
    

def adjust_prediction(val, threshold):
    if val >= threshold:
        return 1
    else:
        return 0

class PreProcessData:
    def __init__(self, raw_file):
        self.raw_file = raw_file
        self.general_data_cleaning
        self.generate_claims_dataset
        self.generate_endorsement_dataset

    def general_data_cleaning(self):
        processed_df = preprocess(self.raw_file)
        processed_df = create_claim_features(processed_df)
        processed_df = filter_dataframe(processed_df)
        return processed_df


    def generate_claims_dataset(self, df):
        return aggregate_claims_dataset(df)

    def generate_endorsement_dataset(self, df):
        return aggregate_endorsement_dataset(df)

