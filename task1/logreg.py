from sklearn.linear_model import LogisticRegression

def fit_predict(data):
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    return lr.predict_proba(X_test)[:, 1]