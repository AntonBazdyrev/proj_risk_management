from sklearn.linear_model import ElasticNet

def fit_predict(data):
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    lr = ElasticNet(random_state=42, alpha=1, l1_ratio=0.1)
    lr.fit(X_train, y_train)
    return lr.predict(X_test)