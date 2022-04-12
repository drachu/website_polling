from website import db

class Poll(db.Model):
    #Informacje o osobie //mieszane//
    id = db.Column(db.Integer(), primary_key=True)
    rok = db.Column(db.Integer(), nullable=False) #listapytho
    zwiazek = db.Column(db.String(length=10), nullable=False) #lista
    praca = db.Column(db.String(length=10), nullable=False) #tak-nie

    # Czas gry //mieszane//
    gra_w_ciagu_12 = db.Column(db.String(length=10), nullable=False) #tak-nie
    gra_prof = db.Column(db.String(length=10), nullable=False) #tak-nie
    gra_tydz = db.Column(db.Integer(), nullable=False) #lista
    gra_tydz_weekend = db.Column(db.Integer(), nullable=False) #suwak

    #Rodzaj gier //mieszane//
    gra_typ = db.Column(db.String(length=30), nullable=False) #lista
    gra_ulubiona = db.Column(db.String(length=30), nullable=False) #textbox
    gra_platforma = db.Column(db.String(length=30), nullable=False) #lista
    gra_klan = db.Column(db.String(length=10), nullable=False) #tak-nie
    gra_grind = db.Column(db.String(length=10), nullable=False) #tak-nie

    #Gra w pandemii //mieszane//
    pandemia_start = db.Column(db.String(length=30), nullable=False) #lista
    pandemia_gra_wiecej = db.Column(db.String(length=10), nullable=False) #tak-nie
    pandemia_czas = db.Column(db.String(length=10), nullable=False) #tak-nie

    #Pozytywne odczucia //mieszane//
    pozytyw_ucieczka = db.Column(db.String(length=10), nullable=False) #tak-nie
    pozytyw_emocje = db.Column(db.String(length=30), nullable=False) #szczescie, gniew, smutek, strach
    pozytyw_samokontrola = db.Column(db.String(length=10), nullable=False) #tak-nie
    pozytyw_koncentracja = db.Column(db.Integer(), nullable=False) #1-5
    pozytyw_koordynacja = db.Column(db.Integer(), nullable=False) #1-5

    #Negatywne skutki //1-5//
    negatyw_gra_pomimo_konsekwencji = db.Column(db.String(length=30), nullable=False)
    negatyw_gra_dluzej = db.Column(db.String(length=30), nullable=False)
    negatyw_gra_nad_inne_aktywnosci = db.Column(db.String(length=30), nullable=False)
    negatyw_zaniedbania = db.Column(db.String(length=30), nullable=False)
    negatyw_gniew = db.Column(db.String(length=30), nullable=False)
    
    #Stosunki z innymi, deprecha //nigdy, rzadzko, czesto, zawsze//
    
    tow_kontakty = db.Column(db.Integer(), nullable=False) #1-5
    tow_inni_sie_przejmuja = db.Column(db.String(length=30), nullable=False)
    tow_brak_towarzystwa = db.Column(db.String(length=30), nullable=False)
    tow_izolacja = db.Column(db.String(length=30), nullable=False)
    tow_przytloczenie = db.Column(db.String(length=30), nullable=False)