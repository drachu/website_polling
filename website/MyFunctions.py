import joblib
import pandas as pd

# wczytanie modelu i predykcja wyniku na podstawie odpowiedzi użytkownika


def predictUser(array):
    model = LoadModel()
    return model.predict(array)

# Załadowanie wytrenowanego modelu


def LoadModel():
    model = joblib.load('website/my_model.pkl')
    return model

# Mapowanie wartości typu String na typ Int


def PreProccess(dataframe):
    """Slowniki do mapowania zmiennych string na wartosci liczbowe"""
    TakNieMap = {'tak': 1, 'nie': 0, '-': 2}
    OnOffMap = {'online': 1, 'offline': 0}
    PcMobileMap = {'pc': 1, 'mobile': 0, 'konsola': 2, 'konsola_mobilna': 3}
    PozaWCzasieMap = {'poza': 1, 'w_czasie': 0}
    NigdyZawszeMap = {'nigdy': 1, 'rzadko': 2,
                      'czasami': 3, 'czesto': 4, 'zawsze': 5}

    """ Mapowanie wartosci typu string na int"""
    dataframe['gra_prof'] = dataframe['gra_prof'].map(TakNieMap)
    dataframe['gra_typ'] = dataframe['gra_typ'].map(OnOffMap)
    dataframe['gra_platforma'] = dataframe['gra_platforma'].map(PcMobileMap)
    dataframe['gra_klan'] = dataframe['gra_klan'].map(TakNieMap)
    dataframe['gra_grind'] = dataframe['gra_grind'].map(TakNieMap)
    dataframe['pandemia_start'] = dataframe['pandemia_start'].map(
        PozaWCzasieMap)
    dataframe['pandemia_gra_wiecej'] = dataframe['pandemia_gra_wiecej'].map(
        TakNieMap)
    dataframe['pandemia_czas'] = dataframe['pandemia_czas'].map(TakNieMap)
    dataframe['pozytyw_ucieczka'] = dataframe['pozytyw_ucieczka'].map(
        TakNieMap)
    dataframe['pozytyw_samokontrola'] = dataframe['pozytyw_samokontrola'].map(
        TakNieMap)
    dataframe['negatyw_gra_pomimo_konsekwencji'] = dataframe['negatyw_gra_pomimo_konsekwencji'].map(
        NigdyZawszeMap)
    dataframe['negatyw_gra_dluzej'] = dataframe['negatyw_gra_dluzej'].map(
        NigdyZawszeMap)
    dataframe['negatyw_gra_nad_inne_aktywnosci'] = dataframe['negatyw_gra_nad_inne_aktywnosci'].map(
        NigdyZawszeMap)
    dataframe['negatyw_zaniedbania'] = dataframe['negatyw_zaniedbania'].map(
        NigdyZawszeMap)
    dataframe['negatyw_gniew'] = dataframe['negatyw_gniew'].map(NigdyZawszeMap)
    dataframe['tow_inni_sie_przejmuja'] = dataframe['tow_inni_sie_przejmuja'].map(
        NigdyZawszeMap)
    dataframe['tow_brak_towarzystwa'] = dataframe['tow_brak_towarzystwa'].map(
        NigdyZawszeMap)
    dataframe['tow_izolacja'] = dataframe['tow_izolacja'].map(NigdyZawszeMap)
    dataframe['tow_przytloczenie'] = dataframe['tow_przytloczenie'].map(
        NigdyZawszeMap)
    dataframe['zwiazek'] = dataframe['zwiazek'].map(TakNieMap)
    dataframe['praca'] = dataframe['praca'].map(TakNieMap)

    return dataframe

# wczytanie dataframu


def loadDataFrame():
    return pd.read_pickle('website/DataFrame.pkl')

# wczytanie dataframu i zwrócenie go


def getEmpytDataFrame():
    dataframe = loadDataFrame()
    return dataframe

# zapisanie odpowiedzi użytkownika do dataframu i obróbka ich


def getDataFrame(dataframe, array):

    dataframe.loc[0] = array
    dataframe.reset_index(drop=True)
    print(dataframe)
    dataframe = PreProccess(dataframe)

    return dataframe
