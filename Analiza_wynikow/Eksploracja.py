import pandas as pd
import MyFunctions as MF
import Model as Model


def run():
    """ Wczytanie bazy danych"""
    database = pd.read_csv(
        'Analiza_wynikow/DataBaseCSV.csv', on_bad_lines='skip')

    """Przygotowanie danych"""
    columnsToDrop = ['id', 'id_2', 'gra_ulubiona', 'gra_w_ciagu_12', 'user_id']
    database = MF.PreProcess(database, columnsToDrop)
    database = MF.AddBinaryClass(database, 20)

    database.to_csv('DatabaseCSVPythonNOTEncoding.csv')
    database = MF.MapStringToInt(database)

    # możliwe opcje przy klasyfikacji w ramach data miningu
    class_names = ['Nie_Uzal', 'Tak_Uzal']

    # nazwy kolumn
    feature_names = ['gra_prof', 'gra_tydz_weekend', 'gra_typ', 'gra_platforma', 'gra_klan',
                     'gra_grind', 'pandemia_start', 'pandemia_gra_wiecej', 'pandemia_czas',
                     'pozytyw_ucieczka', 'pozytyw_samokontrola', 'pozytyw_koncentracja',
                     'pozytyw_koordynacja', 'negatyw_gra_pomimo_konsekwencji',
                     'negatyw_gra_dluzej', 'negatyw_gra_nad_inne_aktywnosci',
                     'negatyw_zaniedbania', 'negatyw_gniew', 'tow_kontakty',
                     'tow_inni_sie_przejmuja', 'tow_brak_towarzystwa', 'tow_izolacja',
                     'tow_przytloczenie', 'rok', 'zwiazek', 'praca']

    # Stworzenie modelu
    Model.CreateModel(database)
    # Zapis dataframu, tylko nazwy kolumn bez ostatniej, który była wykorzystywana tylko podczas dataminingu
    MF.saveDataFrame(database.iloc[0:1, :-1])

    """Data Mining"""
    # Dla wszystkich cech
    MF.skupieniaLokiec(database)
    MF.istotnoscCechLas(database, 0.06)
    MF.Dendogram(database)
    MF.DecisionTree(database, feature_names, class_names, image_name='Entropy_10_0 ',
                    criterion='entropy', max_deph=10, random_state=0)

    # 6 najwazniejszych cech
    columnsToDrop = ['gra_prof', 'gra_typ', 'gra_platforma', 'gra_grind', 'pandemia_start', 'pandemia_gra_wiecej', 'pandemia_czas', 'pozytyw_ucieczka', 'pozytyw_samokontrola', 'pozytyw_koncentracja',
                     'pozytyw_koordynacja', 'negatyw_gra_pomimo_konsekwencji', 'negatyw_gra_dluzej', 'tow_kontakty', 'tow_inni_sie_przejmuja', 'tow_brak_towarzystwa', 'tow_izolacja', 'rok', 'zwiazek', 'praca']
    feature_names = ['negatyw_zaniedbania', 'gra_tydz_weekend',
                     'negatyw_gra_nad_inne_aktywnosci', 'tow_przytloczenie', 'negatyw_gniew', 'gra_klan']
    database = database.drop(columnsToDrop, axis=1)

    MF.DecisionTree(database, feature_names, class_names, image_name='6_Entropy_10_0 ',
                    criterion='entropy', max_deph=10, random_state=0)


if __name__ == "__main__":
    run()
