import numpy as np
import pandas as pd
import MyFunctions as MF
from sklearn.preprocessing import LabelEncoder


def run():
    """ Wczytanie bazy danych (wczesniejsza obrobka w Weka)"""
    database = pd.read_csv('DataBaseCSV.csv', on_bad_lines='skip')
   
    #MF.PrintHoursPerWeek(database)
    #print(round(np.average(database['gra_tydz'].values),2)*2)
    
    """Przygotowanie danych"""
    columnsToDrop = ['id', 'id_2', 'gra_ulubiona', 'gra_w_ciagu_12','user_id']
    database = MF.PreProcess(database, columnsToDrop)
    database = MF.AddBinaryClass(database, 20)

    database.to_csv('DatabaseCSVPythonNOTEncoding.csv')
    database = MF.MapStringToInt(database)

    """Mapowanie za pomoca LabelEncoder"""

    """LabelEncoder niepraktycznie kodował wartosci typu: nigdy, rzadko, czasami, czesto, zawsze"""
    #LabelEnc = LabelEncoder()
    #LabelEnc.inverse_transform(database['gra_prof'])

    #labelsToEncode = ['gra_prof', 'gra_typ', 'gra_platforma', 'gra_klan', 'gra_grind', 'pandemia_start', 'pandemia_gra_wiecej', 'pandemia_czas',
    #                'pozytyw_ucieczka', 'pozytyw_samokontrola', 'negatyw_gra_pomimo_konsekwencji', 'negatyw_gra_dluzej', 'negatyw_gra_nad_inne_aktywnosci', 'negatyw_zaniedbania',
    #               'negatyw_gniew', 'tow_inni_sie_przejmuja', 'tow_brak_towarzystwa', 'tow_izolacja', 'tow_przytloczenie', 'zwiazek', 'praca'
    #               ]
    #database = MF.EncodeLabel(database, labelsToEncode, LabelEnc)

    class_names=['Nie_Uzal', 'Tak_Uzal']
    # feature_names = ['gra_prof', 'gra_tydz_weekend', 'gra_typ', 'gra_platforma', 'gra_klan',
    #                  'gra_grind', 'pandemia_start', 'pandemia_gra_wiecej', 'pandemia_czas',
    #                  'pozytyw_ucieczka', 'pozytyw_samokontrola', 'pozytyw_koncentracja',
    #                  'pozytyw_koordynacja', 'negatyw_gra_pomimo_konsekwencji',
    #                  'negatyw_gra_dluzej', 'negatyw_gra_nad_inne_aktywnosci',
    #                  'negatyw_zaniedbania', 'negatyw_gniew', 'tow_kontakty',
    #                  'tow_inni_sie_przejmuja', 'tow_brak_towarzystwa', 'tow_izolacja',
    #                  'tow_przytloczenie', 'rok', 'zwiazek', 'praca']

    """Data Mining"""
    # MF.skupieniaLokiec(database)
    # MF.analizaSkupien(database,5)

    # MF.DecisionTree(database,feature_names,class_names,image_name='Gini_10_0_None', criterion='gini', max_deph=10,random_state=0,max_features=None)
    # MF.DecisionTree(database,feature_names,class_names, image_name='Gini_10_1_None',criterion='gini', max_deph=10,random_state=1,max_features=None)
    # MF.DecisionTree(database,feature_names,class_names,image_name='Entropy_10_0_None', criterion='entropy', max_deph=10,random_state=0,max_features=None)
    # MF.DecisionTree(database,feature_names,class_names, image_name='Entropy_10_1_None',criterion='entropy', max_deph=10,random_state=1,max_features=None)

    # MF.DecisionTree(database,feature_names,class_names, image_name='Gini_4_0_None',criterion='gini', max_deph=4,random_state=0,max_features=None)
    # MF.DecisionTree(database,feature_names,class_names,image_name='Gini_4_1_None' ,criterion='gini', max_deph=4,random_state=1,max_features=None)
    # MF.DecisionTree(database,feature_names,class_names,image_name='Entropy_4_0_None', criterion='entropy', max_deph=4,random_state=0,max_features=None)
    # MF.DecisionTree(database,feature_names,class_names,image_name='Entropy_4_1_None', criterion='entropy', max_deph=4,random_state=1,max_features=None)

    # MF.DecisionTree(database,feature_names,class_names, image_name='Gini_6_0_None',criterion='gini', max_deph=6,random_state=0,max_features=None)
    # MF.DecisionTree(database,feature_names,class_names,image_name='Gini_6_1_None', criterion='gini', max_deph=6,random_state=1,max_features=None)
    # MF.DecisionTree(database,feature_names,class_names,image_name='Entropy_6_0_None', criterion='entropy', max_deph=6,random_state=0,max_features=None)
    # MF.DecisionTree(database,feature_names,class_names,image_name='Entropy_6_1_None', criterion='entropy', max_deph=6,random_state=1,max_features=None)


    # MF.DecisionTree(database,feature_names,class_names,image_name='Gini_10_0_auto', criterion='gini', max_deph=10,random_state=0,max_features='auto')
    # MF.DecisionTree(database,feature_names,class_names, image_name='Gini_10_1_auto',criterion='gini', max_deph=10,random_state=1,max_features='auto')
    # MF.DecisionTree(database,feature_names,class_names,image_name='Entropy_10_0_auto', criterion='entropy', max_deph=10,random_state=0,max_features='auto')
    # MF.DecisionTree(database,feature_names,class_names, image_name='Entropy_10_1_auto',criterion='entropy', max_deph=10,random_state=1,max_features='auto')

    # MF.DecisionTree(database,feature_names,class_names, image_name='Gini_4_0_auto',criterion='gini', max_deph=4,random_state=0,max_features='auto')
    # MF.DecisionTree(database,feature_names,class_names,image_name='Gini_4_1_auto' ,criterion='gini', max_deph=4,random_state=1,max_features='auto')
    # MF.DecisionTree(database,feature_names,class_names,image_name='Entropy_4_0_auto', criterion='entropy', max_deph=4,random_state=0,max_features='auto')
    # MF.DecisionTree(database,feature_names,class_names,image_name='Entropy_4_1_auto', criterion='entropy', max_deph=4,random_state=1,max_features='auto')

    # MF.DecisionTree(database,feature_names,class_names, image_name='Gini_6_0_auto',criterion='gini', max_deph=6,random_state=0,max_features='auto')
    # MF.DecisionTree(database,feature_names,class_names,image_name='Gini_6_1_auto', criterion='gini', max_deph=6,random_state=1,max_features='auto')
    # MF.DecisionTree(database,feature_names,class_names,image_name='Entropy_6_0_auto', criterion='entropy', max_deph=6,random_state=0,max_features='auto')
    # MF.DecisionTree(database,feature_names,class_names,image_name='Entropy_6_1_auto', criterion='entropy', max_deph=6,random_state=1,max_features='auto')

    #6 najwazniejszych cech
    columnsToDrop = ['gra_prof','gra_typ','gra_platforma','gra_grind','pandemia_start','pandemia_gra_wiecej','pandemia_czas','pozytyw_ucieczka','pozytyw_samokontrola','pozytyw_koncentracja','pozytyw_koordynacja','negatyw_gra_pomimo_konsekwencji','negatyw_gra_dluzej','tow_kontakty','tow_inni_sie_przejmuja','tow_brak_towarzystwa','tow_izolacja','rok','zwiazek','praca']
    feature_names = ['negatyw_zaniedbania','gra_tydz_weekend','negatyw_gra_nad_inne_aktywnosci','tow_przytloczenie','negatyw_gniew','gra_klan']
    database = database.drop(columnsToDrop, axis=1)
    # MF.skupieniaLokiec(database)
    # MF.analizaSkupien(database,5)

    MF.DecisionTree(database,feature_names,class_names,image_name='6_Gini_10_0_None', criterion='gini', max_deph=10,random_state=0,max_features=None)
    MF.DecisionTree(database,feature_names,class_names, image_name='6_Gini_10_1_None',criterion='gini', max_deph=10,random_state=1,max_features=None)
    MF.DecisionTree(database,feature_names,class_names,image_name='6_Entropy_10_0_None', criterion='entropy', max_deph=10,random_state=0,max_features=None)
    MF.DecisionTree(database,feature_names,class_names, image_name='6_Entropy_10_1_None',criterion='entropy', max_deph=10,random_state=1,max_features=None)

    MF.DecisionTree(database,feature_names,class_names, image_name='6_Gini_4_0_None',criterion='gini', max_deph=4,random_state=0,max_features=None)
    MF.DecisionTree(database,feature_names,class_names,image_name='6_Gini_4_1_None' ,criterion='gini', max_deph=4,random_state=1,max_features=None)
    MF.DecisionTree(database,feature_names,class_names,image_name='6_Entropy_4_0_None', criterion='entropy', max_deph=4,random_state=0,max_features=None)
    MF.DecisionTree(database,feature_names,class_names,image_name='6_Entropy_4_1_None', criterion='entropy', max_deph=4,random_state=1,max_features=None)

    MF.DecisionTree(database,feature_names,class_names, image_name='6_Gini_6_0_None',criterion='gini', max_deph=6,random_state=0,max_features=None)
    MF.DecisionTree(database,feature_names,class_names,image_name='6_Gini_6_1_None', criterion='gini', max_deph=6,random_state=1,max_features=None)
    MF.DecisionTree(database,feature_names,class_names,image_name='6_Entropy_6_0_None', criterion='entropy', max_deph=6,random_state=0,max_features=None)
    MF.DecisionTree(database,feature_names,class_names,image_name='6_Entropy_6_1_None', criterion='entropy', max_deph=6,random_state=1,max_features=None)


    MF.DecisionTree(database,feature_names,class_names,image_name='6_Gini_10_0_auto', criterion='gini', max_deph=10,random_state=0,max_features='auto')
    MF.DecisionTree(database,feature_names,class_names, image_name='6_Gini_10_1_auto',criterion='gini', max_deph=10,random_state=1,max_features='auto')
    MF.DecisionTree(database,feature_names,class_names,image_name='6_Entropy_10_0_auto', criterion='entropy', max_deph=10,random_state=0,max_features='auto')
    MF.DecisionTree(database,feature_names,class_names, image_name='6_Entropy_10_1_auto',criterion='entropy', max_deph=10,random_state=1,max_features='auto')

    MF.DecisionTree(database,feature_names,class_names, image_name='6_Gini_4_0_auto',criterion='gini', max_deph=4,random_state=0,max_features='auto')
    MF.DecisionTree(database,feature_names,class_names,image_name='6_Gini_4_1_auto' ,criterion='gini', max_deph=4,random_state=1,max_features='auto')
    MF.DecisionTree(database,feature_names,class_names,image_name='6_Entropy_4_0_auto', criterion='entropy', max_deph=4,random_state=0,max_features='auto')
    MF.DecisionTree(database,feature_names,class_names,image_name='6_Entropy_4_1_auto', criterion='entropy', max_deph=4,random_state=1,max_features='auto')

    MF.DecisionTree(database,feature_names,class_names, image_name='6_Gini_6_0_auto',criterion='gini', max_deph=6,random_state=0,max_features='auto')
    MF.DecisionTree(database,feature_names,class_names,image_name='6_Gini_6_1_auto', criterion='gini', max_deph=6,random_state=1,max_features='auto')
    MF.DecisionTree(database,feature_names,class_names,image_name='6_Entropy_6_0_auto', criterion='entropy', max_deph=6,random_state=0,max_features='auto')
    MF.DecisionTree(database,feature_names,class_names,image_name='6_Entropy_6_1_auto', criterion='entropy', max_deph=6,random_state=1,max_features='auto')

    # MF.sprawdzenieIlosciIstnonychCech(database,6)
    # MF.istotnoscCechLas(database,0.06)
    
    # MF.Dendogram(database)
    # MF.PCAWymiar(database)
    # MF.TestPCA(database)
    # MF.TestSKF(database)
    # MF.LearningCurveTest(database)
    # MF.PozytywNegatywTest(database)
    # MF.ROCTest(database)

    database.to_csv('DatabaseCSVPythonEncoding.csv')

if __name__ == "__main__":
    run()
