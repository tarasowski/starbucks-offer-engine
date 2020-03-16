from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from vecstack import stacking
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib

models = [DecisionTreeClassifier(), 
        RandomForestClassifier(random_state=7),
        XGBClassifier(objective='binary:logistic', random_state=7),
        AdaBoostClassifier(random_state=7),
        GradientBoostingClassifier(random_state=7)]

def prepare(data):
    X_train, X_test, y_train, y_test = data
    S_train, S_test = stacking(models,
            X_train, y_train, X_test, regression=False,
            mode='oof_pred_bag',
            needs_proba=False,
            save_dir=None,
            metric=accuracy_score,
            n_folds=7,
            stratified=True,
            shuffle=True,
            random_state=7,
            verbose=2)
    return (S_train, y_train, S_test, y_test)

def train(data):
    S_train, y_train, S_test, y_test = data
    model = RandomForestClassifier(random_state=7)
    model = model.fit(S_train, y_train)
    return (model, S_test, y_test)

def evaluate(data):
    model, S_test, y_test = data
    y_pred = model.predict(S_test)
    print(f'Accuracy: {round(accuracy_score(y_test, y_pred) * 100, 2)}%')
    print(f'F1 Score: {round(f1_score(y_test, y_pred) * 100, 2)}%')
    print(f'Recall Score: {round(precision_score(y_test, y_pred) * 100, 2)}%')
    print(f'Precision Score: {round(recall_score(y_test, y_pred) * 100, 2)}%')

def main(params):
    return train(prepare(params))

if __name__ == '__main__':
    X_train = joblib.load(load_path + 'X_train.pkl')
    X_test = joblib.load(load_path + 'X_test.pkl')
    y_train = joblib.load(load_path + 'y_train.pkl')
    y_test = joblib.load(load_path + 'y_test.pkl')
    model, S_test, y_test = main((X_train, X_test, y_train, y_test))
