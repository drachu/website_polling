
"""Kod w głównej mierze pochodzi z książki Sebastiana Raschka pt.Python Machine learning i deep learning Biblioteki scikit-learn i TensorFlow2 Wydanie III wydawnictwa Helion """

import numpy as np
import matplotlib.pyplot as plt

import os
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import ClassesClassifer as CC
from matplotlib.colors import ListedColormap
from sklearn.feature_selection import SelectFromModel
from scipy.cluster.hierarchy import dendrogram

"""Slowniki do mapowania zmiennych string na wartosci liczbowe"""
TakNieMap = {'tak': 1, 'nie': 0, '-': 2}
OnOffMap = {'online': 1, 'offline': 0}
PcMobileMap = {'pc': 1, 'mobile': 0}
PozaWCzasieMap = {'poza': 1, 'w_czasie': 0}
NigdyZawszeMap = {'nigdy': 1, 'rzadko': 2,
                  'czasami': 3, 'czesto': 4, 'zawsze': 5}


def MapStringToInt(database):
    """ Mapowanie wartosci typu string na int"""
    database['gra_prof'] = database['gra_prof'].map(TakNieMap)
    database['gra_typ'] = database['gra_typ'].map(OnOffMap)
    database['gra_platforma'] = database['gra_platforma'].map(PcMobileMap)
    database['gra_klan'] = database['gra_klan'].map(TakNieMap)
    database['gra_grind'] = database['gra_grind'].map(TakNieMap)
    database['pandemia_start'] = database['pandemia_start'].map(PozaWCzasieMap)
    database['pandemia_gra_wiecej'] = database['pandemia_gra_wiecej'].map(
        TakNieMap)
    database['pandemia_czas'] = database['pandemia_czas'].map(TakNieMap)
    database['pozytyw_ucieczka'] = database['pozytyw_ucieczka'].map(TakNieMap)
    database['pozytyw_samokontrola'] = database['pozytyw_samokontrola'].map(
        TakNieMap)
    database['negatyw_gra_pomimo_konsekwencji'] = database['negatyw_gra_pomimo_konsekwencji'].map(
        NigdyZawszeMap)
    database['negatyw_gra_dluzej'] = database['negatyw_gra_dluzej'].map(
        NigdyZawszeMap)
    database['negatyw_gra_nad_inne_aktywnosci'] = database['negatyw_gra_nad_inne_aktywnosci'].map(
        NigdyZawszeMap)
    database['negatyw_zaniedbania'] = database['negatyw_zaniedbania'].map(
        NigdyZawszeMap)
    database['negatyw_gniew'] = database['negatyw_gniew'].map(NigdyZawszeMap)
    database['tow_inni_sie_przejmuja'] = database['tow_inni_sie_przejmuja'].map(
        NigdyZawszeMap)
    database['tow_brak_towarzystwa'] = database['tow_brak_towarzystwa'].map(
        NigdyZawszeMap)
    database['tow_izolacja'] = database['tow_izolacja'].map(NigdyZawszeMap)
    database['tow_przytloczenie'] = database['tow_przytloczenie'].map(
        NigdyZawszeMap)
    database['zwiazek'] = database['zwiazek'].map(TakNieMap)
    database['praca'] = database['praca'].map(TakNieMap)

    return database

# Dodanie nowej kolumny bazującej na ilości godzin spędzonych przy komputerze. Kolumna ta służy w dalszej analizie


def AddBinaryClass(database, BitValue):
    ValueList = []
    for value in database['gra_tydz']:
        if value > BitValue:
            ValueList.append(1)
        else:
            ValueList.append(0)
    # axis=0 == rows, axis=1 == columns
    database = database.drop('gra_tydz', axis=1)
    database['WynikLudzki'] = np.array(ValueList)

    return database

# usunięcie niepotrzebnych  lub błędnych danych oraz kolumn


def PreProcess(database, columnsToDrop):
    database.drop_duplicates()
    database = database.dropna(axis=0)
    database = database.dropna(axis=1)
    database = database.drop(database[database.gra_tydz > 168].index)
    database = database.drop(columnsToDrop, axis=1)

    return database


# Wykorzystanie algorytmu KMeans do zbadania najbardziej optymalnej ilości klastrów
def skupieniaLokiec(database):
    distortions = []
    for i in range(1, 42):
        km = KMeans(n_clusters=i, init='k-means++',
                    n_init=10, max_iter=300, random_state=0)
        km.fit(database)
        # Zniekształcenie jako suma kwadratów odległości próbek do najbliższego środka klastra
        distortions.append(km.inertia_)

    plt.plot(range(1, 42), distortions, marker='o')
    plt.xlabel('Liczba Skupien')
    plt.ylabel('Znieksztalcenie')
    plt.tight_layout()
    plt.show()


# Stworzenie i zapisanie jako .png drzewa decyzyjnego powstałego na podstawie wprowadzonych danych
def DecisionTree(database, feature_names, class_names, image_name, criterion='gini', max_deph=5, random_state=1, max_features=None):
    tree_model = DecisionTreeClassifier(
        criterion=criterion, max_depth=max_deph, random_state=random_state, max_features=max_features)
    X = database.iloc[:, :-1].values
    y = database['WynikLudzki'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.025, random_state=0, stratify=y)
    tree_model.fit(X_train, y_train)

    SaveImageDecisionTree(tree_model, feature_names, class_names, image_name)

# Program graphviz generuje ładniejsze obrazy reprezentujące powstałe drzewo decyzyjne


def SaveImageDecisionTree(tree_model, feature_names, class_names, image_name):
    os.environ["PATH"] += os.pathsep + 'C:\Programy\Graphviz/bin/'

    dot_data = export_graphviz(tree_model,
                               filled=True,
                               rounded=True,
                               class_names=class_names,
                               feature_names=feature_names,
                               out_file=None)
    graph = graph_from_dot_data(dot_data)
    graph.write_png('Analiza_wynikow/Images/'+image_name+'.png')

# Sprawdzenie ilości istotnych cech poprzez wykorzystanie estymatora KNN do klasyfikatora SBS


def sprawdzenieIlosciIstnonychCech(database, n_neighbors=5):

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    X = database.iloc[:, :-1].values
    y = database['WynikLudzki'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.025, random_state=0, stratify=y)

    sbs = CC.SBS(knn, k_features=5)
    sbs.fit(X_train, y_train)

    k_feat = [len(k) for k in sbs.subsets_]

    knn.fit(X_train, y_train)

    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.02])
    plt.ylabel('Dokladnosc')
    plt.xlabel('Liczba cech')
    plt.grid()
    plt.tight_layout()
    plt.show()

# Wykorzystanie losowego lasu do określenia, które cechy są najważniejsze


def istotnoscCechLas(database, threshold=0.06):
    feat_labels = database.columns[:-1]

    X = database.iloc[:, :-1].values
    y = database['WynikLudzki'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.025, random_state=0, stratify=y)
    forest = RandomForestClassifier(n_estimators=500,
                                    random_state=1)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                feat_labels[indices[f]],
                                importances[indices[f]]))

    plt.title('Waznosc cech')
    plt.bar(range(X_train.shape[1]),
            importances[indices],
            align='center')

    plt.xticks(range(X_train.shape[1]),
               feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.show()
    SelectFromModelFunction(forest, X_train, feat_labels,
                            importances, indices, threshold)


def SelectFromModelFunction(model, X_train, feat_labels, importances, indices, threshold=0.1):
    sfm = SelectFromModel(model, threshold=threshold, prefit=True)
    X_selected = sfm.transform(X_train)
    print('Liczba cech spełniających podane kryterium progowe:',
          X_selected.shape[1])

    for f in range(X_selected.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                feat_labels[indices[f]],
                                importances[indices[f]]))


def Dendogram(database):

    labels = []
    for i in range(42):
        labels.append('Rekord: '+str(i))
    labels = np.array(labels)

    row_clusters = linkage(
        database.values, method='complete', metric='euclidean')

    row_dendr = dendrogram(row_clusters, labels=labels, color_threshold=np.inf)
    plt.tight_layout()
    plt.ylabel('odleglosc Euklidesowa')
    plt.show()


def plot_decision_regions(X, y, classifier, resolution=0.02):

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    color=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)


def saveDataFrame(df):
    df.to_pickle('DataFrame.pkl')
