import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data():
    model = joblib.load('../models/stacked-model.pkl')
    S_test = joblib.load('../models/S_test.pkl')
    y_test = joblib.load('../models/y_test.pkl')
    return (model, S_test, y_test)

def evaluate(data):
    clf, X_test, y_test = data
    y_pred = clf.predict(X_test)
    print(f'Accuracy: {round(accuracy_score(y_test, y_pred) * 100, 2)}%')
    print(f'F1 Score: {round(f1_score(y_test, y_pred) * 100, 2)}%')
    print(f'Recall Score: {round(precision_score(y_test, y_pred) * 100, 2)}%')
    print(f'Precision Score: {round(recall_score(y_test, y_pred) * 100, 2)}%')

def main():
    return load_data()


if __name__ == '__main__':
    evaluate(main())
