import pandas as pd
from functools import reduce 
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

pipe = lambda fns: lambda x: reduce(lambda v, f: f(v), fns, x)

def load_df(path, filename):
    return pd.read_csv(path + filename) 

def merge_profile(dfs):
    offers, profile = dfs
    offers = offers.merge(profile, left_on='customer_id', right_on='customer_id', how='inner')
    return offers

def prepare_data(df):
    df['gender'] = df['gender'].map({0: 'Man', 1: 'Woman'})
    df = df.drop(columns=['customer_id', 
            'time', 
            'offer_id', 
            'offer_name', 
            'event_offer_completed', 
            'event_offer_received', 
            'event_offer_viewed',
            'offer_ends',
            'completed_at',
            'viewed_at',
            'became_member_on',
            'became_member_month',
            'membership_duration',
            'offer_success_no_view',
            'income_binned',
            'age_categories',
            'amount' # should be dropped, since it's a posterior value
            ], axis=1)
    df = pd.get_dummies(data=df, columns=['gender', 'offer_type'])
    return df

def remove_info_offer(df):
    df = df[df['offer_type_informational'] != 1]
    return df

def create_X_y(df):
    X = df.drop('offer_success', axis=1)
    y = df['offer_success']
    return (X, y)

def oversample(data):
    X, y = data
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return (X_resampled, y_resampled)

def feat_scaling(data):
    X, y = data
    scaler = MinMaxScaler()
    to_scale = ['planned_reward', 'required_spend', 'duration', 'age', 'income', 'became_member_year']
    X[to_scale] = scaler.fit_transform(X[to_scale])
    return (X, y)

def data_split(data):
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7, shuffle=True)
    return (X_train, X_test, y_train, y_test)



def program(offers, profile):
    X_train, X_test, y_train, y_test = pipe([merge_profile,
            prepare_data,
            remove_info_offer,
            create_X_y,
            oversample,
            feat_scaling,
            data_split,
            ])((offers, profile))
    return (X_train, X_test, y_train, y_test)

def main(offers, profile):
    return program(offers, profile)

if __name__ == '__main__':
    offers = load_df(load_path, 'offers.csv')
    profile = load_df(load_path, 'profile.csv')
    X_train, X_test, y_train, y_test = main(offers, profile)

