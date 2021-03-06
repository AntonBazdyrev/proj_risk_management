from lightgbm import LGBMRegresssor

def fit_predict(data):
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    lgbm = LGBMRegresssor(
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=5,
        n_estimators=100,
        subsample=0.7,
        subsample_freq=2,
        colsample_bytree=0.5,
        random_state=42
    )
    lgbm.fit(X_train, y_train)
    return lgbm.predict(X_test)