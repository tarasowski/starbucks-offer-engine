import pandas as pd
import numpy as np
import math
import json
from functools import reduce

pipe = lambda fns: lambda x: reduce(lambda v, f: f(v), fns, x)


def load_dfs():
    portfolio = pd.read_json('../input/portfolio.json', 
            orient='records', lines=True)
    profile = pd.read_json('../input/profile.json', 
            orient='records', lines=True)
    transcript = pd.read_json('../input/transcript.json', 
            orient='records', lines=True)
    return (portfolio, profile, transcript)

def save_df(df, path):
    df.to_csv(path, index=false)

# ------------- start portfolio data preprocessing ----------------
def make_offer_name(portfolio):
    portfolio['offer_name'] = portfolio['offer_type'] +\
        '-' + portfolio['required_spend'].astype(str) +\
        'spend-' + + portfolio['reward'].astype(str) +\
        'reward-' + portfolio['duration'].astype(str) + 'days' 
    return portfolio

def unpack_channels(portfolio):
    dummies = pd.get_dummies(portfolio['channels']\
            .apply(pd.Series).stack(), prefix='channel').sum(level=0)
    portfolio = portfolio.merge(dummies, left_index=True, right_index=True)
    portfolio = portfolio.drop('channels', axis=1)
    return portfolio

def rn_portfolio(portfolio):
    portfolio = portfolio.rename(columns={'id': 'offer_id', 
            'difficulty': 'required_spend'})
    return portfolio

# -------------- end portfolio data preprocessing -----------------

# ------------- start profile data preprocessing -----------------

def membership(profile):
    profile['became_member_on'] = pd.to_datetime(profile['became_member_on'], 
                                                 format='%Y%m%d')
    profile['became_member_year'] = pd.DatetimeIndex(profile['became_member_on'])\
                                      .year
    profile['became_member_month'] = pd.DatetimeIndex(profile['became_member_on'])\
                                       .month
    profile['membership_duration'] = (profile['became_member_on'].max() - profile['became_member_on'])
    return profile

def replace_age(profile):
    profile['age'] = profile['age'].replace(118, np.nan)
    return profile

def drop_gender_O(profile):
    profile['gender'] = profile[profile['gender'] !='O']
    return profile

def encode_gender(profile):
    profile['gender'] = profile['gender'].map({'M': 0, 'F': 1})
    return profile

def drop_gender_null(profile):
    profile = profile.drop(profile[profile['gender'].isnull()].index)
    return profile

def age_categories(profile):
    bins = [10, 20, 30, 40, 50, 60, 70, 80, 120]
    group_names = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    profile['age_categories'] = pd.cut(profile['age'], bins, labels=group_names)
    return profile

def income_categories(profile):
    bins = [10, 20, 30, 40, 50, 60, 70, 80, 120]
    group_names = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    profile['age_categories'] = pd.cut(profile['age'], bins, labels=group_names)
    return profile

def rename_id(profile):
    profile = profile.rename(columns={'id': 'customer_id'})
    return profile

# ------------- end profile data preprocessing -------------------

# ------------- start transcript data preprocessing --------------

def parse_value(transcript):
    values_df = pd.DataFrame(transcript['value'].tolist())
    values_df['offerid'] = values_df['offer id'].combine_first(values_df['offer_id'])
    values_df.drop(['offer id', 'offer_id'], axis=1, inplace=True)
    index_df = pd.DataFrame({'idx': transcript.index.values.tolist()})
    combined = index_df.merge(values_df, left_index=True, right_index=True)
    combined_index = combined.set_index('idx')
    transcript = transcript.merge(combined_index, left_index=True, right_index=True)
    return transcript

def merge_portfolio(portfolio):
    def inner(transcript):
        transcript = transcript.merge(portfolio, left_on='offerid', right_on='offer_id', how='left')
        return transcript
    return inner

def transcript_rename(transcript):
    transcript = transcript.rename(columns={'reward_x': 'paid_reward', 
            'reward_y': 'planned_reward', 'person': 'customer_id'})
    return transcript

def convert_time(transcript):
    transcript['time'] = transcript['time'] / 24
    transcript = transcript.drop(columns=['value'])
    return transcript

def transcript_duplicates(transcript):
    transcript = transcript.drop_duplicates().reset_index(drop=True)
    return transcript

def create_transaction_df(transcript):
    transaction_df = transcript[transcript['event'] == 'transaction'].copy()
    transaction_df = transaction_df[['customer_id', 'time', 'amount']]
    return (transcript, transaction_df)

def create_offer_df(dfs):
    transcript, transaction_df = dfs
    offer_df = transcript[transcript['event'] != 'transaction'].copy()
    offer_df = pd.get_dummies(offer_df, columns=['event'], prefix=None)
    offer_df = offer_df.drop(columns=['amount', 'offerid', 'paid_reward'], axis=1)
    offer_df = offer_df.rename(columns={
            'event_offer completed': 'event_offer_completed',
            'event_offer received': 'event_offer_received',
            'event_offer viewed': 'event_offer_viewed'})
    offer_df['offer_ends'] = offer_df['time'] + offer_df['duration']
    return (offer_df, transaction_df)

# ------------- end transcript data preprocessing ----------------

# ------------- start metrics aggregation ------------------------

def get_customer_ids(offer_df):
    return offer_df['customer_id'].unique()

def filter_by_cid(df, cid):
    return df[df['customer_id'] == cid]

def get_received(df):
    return df[df['event_offer_received'] == 1]

def get_viewed(df):
    return df[df['event_offer_viewed'] == 1]

def get_completed(df):
    return df[df['event_offer_completed'] == 1]

def completed_lookup(row, completed_offers):
    row_len = row.shape[0]
    if row_len > 0:
        offers = completed_offers.loc[completed_offers['offer_id'] == row['offer_id'], :]
        time = offers.loc[offers['time'] >= row['time'], 'time']
        val = time.values
        if len(val) == 0:
            return -1
        else:
            return val[0]

def viewed_lookup(idx, row, viewed_offers):
    row_len = row.shape[0]
    if row_len > 0:
        offers = viewed_offers.loc[viewed_offers['offer_id'] == row['offer_id'], :]
        time = offers.loc[offers['time'] >= row['time'], 'time']
        val = time.values
        if len(val) == 0:
            return -1
        else:
            return val[0]

def transaction_lookup(row, transactions_df):
    start = row['time']
    end = row['offer_ends']
    cid = row['customer_id']
    condition = (transactions_df['customer_id'] == cid)
    time = transactions_df.loc[condition, 'time'].between(start, end)
    amount = transactions_df.loc[time, 'amount'].sum()
    return amount

def calc_offers(offer_df, transaction_df):
    def inner(cid):
        offers = filter_by_cid(offer_df, cid)
        transactions = filter_by_cid(transaction_df, cid)
        received_offers = get_received(offers)
        viewed_offers = get_viewed(offers)
        completed_offers = get_completed(offers)
        transactions = filter_by_cid(transaction_df, cid)
        received_offers.loc[:, 'completed_at'] = received_offers.apply(lambda row: completed_lookup(row, completed_offers), axis=1)
        received_offers.loc[:, 'viewed_at'] = received_offers.apply(lambda row: viewed_lookup(row.name, row, viewed_offers), axis=1)
        received_offers.loc[:, 'offer_success'] = (received_offers['offer_ends'] > received_offers['completed_at']) &\
                                                  (received_offers['viewed_at'] <= received_offers['completed_at']) &\
                                                  (received_offers['completed_at'] != -1) & (received_offers['viewed_at'] != -1)
        received_offers.loc[:, 'offer_success_no_view'] = (received_offers['completed_at'] > 0) &\
                                                          ((received_offers['viewed_at'] == -1) |\
                                                                  (received_offers['viewed_at'] > received_offers['completed_at']))
        received_offers.loc[:, 'amount'] = received_offers.apply(lambda row: transaction_lookup(row, transactions), axis=1)
        received_offers.loc[:, 'offer_success'] = received_offers['offer_success'].map({True: 1, False: 0})
        received_offers.loc[:, 'offer_success_no_view'] = received_offers['offer_success_no_view'].map({True: 1, False: 0})
        return received_offers
    return inner

def calculated_offers_df(customer_ids, offer_df, transaction_df):
    vfunc = np.vectorize(calc_offers(offer_df, transaction_df))
    calculated_offers_df = pd.concat(vfunc(customer_ids))
    return calculated_offers_df

# ------------- end metrics aggregation --------------------------


def program(portfolio, profile, transcript):
    portfolio = pipe([rn_portfolio, 
            unpack_channels, 
            make_offer_name])(portfolio)

    profile = pipe([membership,
            replace_age,
            drop_gender_O,
            encode_gender,
            drop_gender_null,
            age_categories,
            income_categories,
            rename_id
            ])(profile)

    offer_df, transaction_df = pipe([parse_value,
            merge_portfolio(portfolio),
            transcript_rename,
            convert_time,
            transcript_duplicates,
            create_transaction_df,
            create_offer_df
            ])(transcript)

    offers = calculated_offers_df(
            get_customer_ids(offer_df), 
            offer_df, transaction_df)

    return (offers, profile, portfolio)

def main():
    return program(*load_dfs())
    

if __name__ == '__main__':
    offers, profile, portfolio = main()
    #save_df(offers, '../input/offers.csv')
    #save_df(profile, '../input/profile.csv')
    #save_df(portfolio, '../input/portfolio.csv')

