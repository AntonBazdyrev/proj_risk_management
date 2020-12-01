from ta import add_all_ta_features

def create_features_and_split(df, target, split_date='2018-12-31'):
    for i in range(1, 10):
        for column in ['Close', 'Low', 'High']:
            df[f'lag_{i}_d{column}'] = df[column].diff().shift(i).fillna(0.)

    df = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume").fillna(0)
    X = df.drop(columns=['Open', 'High', 'Low', 'Close'])
    return {
        'X_train': X[X.Date < split_date].drop(columns=['Date']),
        'y_train': target[X.Date < split_date],
        'X_test': X[X.Date >= split_date].drop(columns=['Date']),
        'y_test': target[X.Date >= split_date],
    }