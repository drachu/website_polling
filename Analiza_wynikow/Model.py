import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def CreateModel(database):
    feat_labels = database.columns[:-1]

    X = database.iloc[:, :-1].values
    y = database['WynikLudzki'].values

    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=101, stratify=y)

    forest = RandomForestClassifier(n_estimators=3703,
                                    random_state=101)

    forest.fit(X_train, y_train)
    print(forest.score(X_test, y_test))

    SaveModel(forest)


def SaveModel(model):
    joblib.dump(model, 'my_model.pkl')


def LoadModel():
    model = joblib.load('my_model.pkl')
    return model
