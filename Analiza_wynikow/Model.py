from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# stworzenie i zapisanie modelu


def CreateModel(database):

    X = database.iloc[:, :-1].values
    y = database['WynikLudzki'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=101, stratify=y)

    # Model bazujący na losowym drzewie (liczba drzew została określona metodą prób i błędów, dla tego n_estimators wynik modelu zaczął wynosić 82%)
    forest = RandomForestClassifier(n_estimators=3703, random_state=101)

    forest.fit(X_train, y_train)

    # Miara jakości moelu wynosi ok. 82%
    print(forest.score(X_test, y_test))

    SaveModel(forest)


def SaveModel(model):
    joblib.dump(model, 'my_model.pkl')
