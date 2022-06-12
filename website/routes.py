from run import app
from flask import render_template, redirect, request, url_for, flash
from website.forms import GamesForm, InfoForm
from website.models import Info, Games
from website import db
from flask_sqlalchemy import SQLAlchemy
import json
import website.MyFunctions as MF


@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')


@app.route('/form', methods=['GET', 'POST'])
def form_page():
    infoForm = InfoForm()
    gamesForm = GamesForm()
    if request.method == 'POST':

        if infoForm.validate_on_submit() and infoForm.gra_w_ciagu_12.data == 'nie':

            user_poll_info = Info(rok=infoForm.rok.data, zwiazek=infoForm.zwiazek.data,
                                  praca=infoForm.praca.data, gra_w_ciagu_12=infoForm.gra_w_ciagu_12.data,)

            db.session.add(user_poll_info)
            db.session.commit()
            return redirect(url_for('thanks_page'))
        elif gamesForm.validate_on_submit():

            # zapisanie odpowiedzi użytkownika do tablicy, aby móć później na nich wygodnie operować
            odp_list = [gamesForm.gra_prof.data, gamesForm.gra_tydz_weekend.data, gamesForm.gra_typ.data,
                        gamesForm.gra_platforma.data, gamesForm.gra_klan.data, gamesForm.gra_grind.data,
                        gamesForm.pandemia_start.data, gamesForm.pandemia_gra_wiecej.data, gamesForm.pandemia_czas.data,
                        gamesForm.pozytyw_ucieczka.data, gamesForm.pozytyw_samokontrola.data, gamesForm.pozytyw_koncentracja.data,
                        gamesForm.pozytyw_koordynacja.data, gamesForm.negatyw_gra_pomimo_konsekwencji.data, gamesForm.negatyw_gra_dluzej.data,
                        gamesForm.negatyw_gra_nad_inne_aktywnosci.data, gamesForm.negatyw_zaniedbania.data, gamesForm.negatyw_gniew.data,
                        gamesForm.tow_kontakty.data, gamesForm.tow_inni_sie_przejmuja.data, gamesForm.tow_brak_towarzystwa.data,
                        gamesForm.tow_izolacja.data, gamesForm.tow_przytloczenie.data, infoForm.rok.data, infoForm.zwiazek.data, infoForm.praca.data
                        ]

            # Wczytanie dataframu z ekploracji danych (rozkład i nazwy kolumn) i zapisanie do niego odpowiedzi użytkownika, aby następnie tak przygotowany dataframe wysłać do modelu
            emptyDataFrame = MF.getEmpytDataFrame()
            odp_array = MF.getDataFrame(emptyDataFrame, odp_list)

            # Przewidywanie modelu
            wynik = MF.predictUser(odp_array)

            # Dodanie odpowiedzi użytkowniaka do bazy danych + przewidywanie modelu

            user_poll_info = Info(rok=infoForm.rok.data, zwiazek=infoForm.zwiazek.data,
                                  praca=infoForm.praca.data, gra_w_ciagu_12=infoForm.gra_w_ciagu_12.data,
                                  uzaleznienie_model=int(wynik),)

            user_poll_info = Info(rok=infoForm.rok.data, zwiazek=infoForm.zwiazek.data,
                                  praca=infoForm.praca.data, gra_w_ciagu_12=infoForm.gra_w_ciagu_12.data,)
            db.session.add(user_poll_info)
            db.session.commit()
            user_poll_games = Games(gra_prof=gamesForm.gra_prof.data, gra_tydz=gamesForm.gra_tydz.data, gra_tydz_weekend=gamesForm.gra_tydz_weekend.data, gra_typ=gamesForm.gra_typ.data, gra_ulubiona=gamesForm.gra_ulubiona.data, gra_platforma=gamesForm.gra_platforma.data, gra_klan=gamesForm.gra_klan.data, gra_grind=gamesForm.gra_grind.data, pandemia_start=gamesForm.pandemia_start.data, pandemia_gra_wiecej=gamesForm.pandemia_gra_wiecej.data, pandemia_czas=gamesForm.pandemia_czas.data, pozytyw_ucieczka=gamesForm.pozytyw_ucieczka.data, pozytyw_samokontrola=gamesForm.pozytyw_samokontrola.data, pozytyw_koncentracja=gamesForm.pozytyw_koncentracja.data,
                                    pozytyw_koordynacja=gamesForm.pozytyw_koordynacja.data, negatyw_gra_pomimo_konsekwencji=gamesForm.negatyw_gra_pomimo_konsekwencji.data, negatyw_gra_dluzej=gamesForm.negatyw_gra_dluzej.data, negatyw_gra_nad_inne_aktywnosci=gamesForm.negatyw_gra_nad_inne_aktywnosci.data, negatyw_zaniedbania=gamesForm.negatyw_zaniedbania.data, negatyw_gniew=gamesForm.negatyw_gniew.data, tow_kontakty=gamesForm.tow_kontakty.data, tow_inni_sie_przejmuja=gamesForm.tow_inni_sie_przejmuja.data, tow_brak_towarzystwa=gamesForm.tow_brak_towarzystwa.data, tow_izolacja=gamesForm.tow_izolacja.data, tow_przytloczenie=gamesForm.tow_przytloczenie.data, user_id=user_poll_info.id)

            db.session.add(user_poll_games)
            db.session.commit()
            if (wynik == 1):
                return redirect(url_for('danger_page'))
            else:
                return redirect(url_for('thanks_page'))

        else:
            flash('Uzupełnij odpowiedź!', category='error')

    return render_template('form.html', infoForm=infoForm, gamesForm=gamesForm)


@app.route('/thanks')
def thanks_page():
    nigdy_zawsze_labels = ['nigdy', 'rzadko', 'czasami', 'czesto', 'zawsze']
    skala_labels = [1, 2, 3, 4, 5]

    gra_w_ciagu_12_db = db.session.query(db.func.count(
        Info.gra_w_ciagu_12)).group_by(Info.gra_w_ciagu_12).all()
    gra_w_ciagu_12 = []
    for value, in gra_w_ciagu_12_db:
        gra_w_ciagu_12.append(value)

    # NEGATYWNE CECHY
    gra_gniew_db = db.session.query(Games.negatyw_gniew, db.func.count(
        Games.negatyw_gniew)).group_by(Games.negatyw_gniew).all()
    gra_gniew_values = [0, 0, 0, 0, 0]
    for i in range(0, 5):
        for x in gra_gniew_db:
            if nigdy_zawsze_labels[i] == x[0]:
                gra_gniew_values[i] = x[1]

    negatyw_zaniedbania_db = db.session.query(Games.negatyw_zaniedbania, db.func.count(
        Games.negatyw_zaniedbania)).group_by(Games.negatyw_gniew).all()
    negatyw_zaniedbania_values = [0, 0, 0, 0, 0]
    for i in range(0, 5):
        for x in negatyw_zaniedbania_db:
            if nigdy_zawsze_labels[i] == x[0]:
                negatyw_zaniedbania_values[i] = x[1]

    negatyw_gra_nad_inne_aktywnosci_db = db.session.query(Games.negatyw_gra_nad_inne_aktywnosci, db.func.count(
        Games.negatyw_gra_nad_inne_aktywnosci)).group_by(Games.negatyw_gra_nad_inne_aktywnosci).all()
    negatyw_gra_nad_inne_aktywnosci_values = [0, 0, 0, 0, 0]
    for i in range(0, 5):
        for x in negatyw_gra_nad_inne_aktywnosci_db:
            if nigdy_zawsze_labels[i] == x[0]:
                negatyw_gra_nad_inne_aktywnosci_values[i] = x[1]

    # POZYTYWNE

    pozytyw_koordynacja_db = db.session.query(Games.pozytyw_koordynacja, db.func.count(
        Games.pozytyw_koordynacja)).group_by(Games.pozytyw_koordynacja).all()
    pozytyw_koordynacja_values = [0, 0, 0, 0, 0]
    for i in range(0, 5):
        for x in pozytyw_koordynacja_db:
            if skala_labels[i] == x[0]:
                pozytyw_koordynacja_values[i] = x[1]

    pozytyw_samokontrola_db = db.session.query(db.func.count(
        Games.pozytyw_samokontrola)).group_by(Games.pozytyw_samokontrola).all()
    pozytyw_samokontrola_values = []
    for value, in pozytyw_samokontrola_db:
        pozytyw_samokontrola_values.append(value)

    pozytyw_koncentracja_db = db.session.query(Games.pozytyw_koncentracja, db.func.count(
        Games.pozytyw_koncentracja)).group_by(Games.pozytyw_koncentracja).all()
    pozytyw_koncentracja_values = [0, 0, 0, 0, 0]
    for i in range(0, 5):
        for x in pozytyw_koncentracja_db:
            if skala_labels[i] == x[0]:
                pozytyw_koncentracja_values[i] = x[1]

    gra_prof_db = db.session.query(db.func.count(
        Games.gra_prof)).group_by(Games.gra_prof).all()
    gra_prof_values = []
    for value, in gra_prof_db:
        gra_prof_values.append(value)

    gra_typ_db = db.session.query(db.func.count(
        Games.gra_typ)).group_by(Games.gra_typ).all()
    gra_typ_values = []
    for value, in gra_typ_db:
        gra_typ_values.append(value)

    uzaleznienie_db = db.session.query(db.func.count(
        Info.uzaleznienie_model)).group_by(Info.uzaleznienie_model).all()
    uzaleznienie_values = []
    for value, in uzaleznienie_db:
        uzaleznienie_values.append(value)
    uzaleznienie_procent = round(
        (uzaleznienie_values[0]*100)/sum(uzaleznienie_values), 1)

    return render_template('thanks.html', gra_w_ciagu_12=json.dumps(gra_w_ciagu_12), gra_gniew_values=json.dumps(gra_gniew_values), negatyw_zaniedbania_values=json.dumps(negatyw_zaniedbania_values), negatyw_gra_nad_inne_aktywnosci_values=json.dumps(negatyw_gra_nad_inne_aktywnosci_values), pozytyw_koordynacja_values=json.dumps(pozytyw_koordynacja_values), pozytyw_koncentracja_values=json.dumps(pozytyw_koncentracja_values), pozytyw_samokontrola_values=json.dumps(pozytyw_samokontrola_values), gra_typ_values=json.dumps(gra_typ_values), uzaleznienie_values=json.dumps(uzaleznienie_values), uzaleznienie_procent=uzaleznienie_procent)


@app.route('/danger')
def danger_page():
    nigdy_zawsze_labels = ['nigdy', 'rzadko', 'czasami', 'czesto', 'zawsze']
    skala_labels = [1, 2, 3, 4, 5]

    gra_w_ciagu_12_db = db.session.query(db.func.count(
        Info.gra_w_ciagu_12)).group_by(Info.gra_w_ciagu_12).all()
    gra_w_ciagu_12 = []
    for value, in gra_w_ciagu_12_db:
        gra_w_ciagu_12.append(value)

    # NEGATYWNE CECHY
    gra_gniew_db = db.session.query(Games.negatyw_gniew, db.func.count(
        Games.negatyw_gniew)).group_by(Games.negatyw_gniew).all()
    gra_gniew_values = [0, 0, 0, 0, 0]
    for i in range(0, 5):
        for x in gra_gniew_db:
            if nigdy_zawsze_labels[i] == x[0]:
                gra_gniew_values[i] = x[1]

    negatyw_zaniedbania_db = db.session.query(Games.negatyw_zaniedbania, db.func.count(
        Games.negatyw_zaniedbania)).group_by(Games.negatyw_gniew).all()
    negatyw_zaniedbania_values = [0, 0, 0, 0, 0]
    for i in range(0, 5):
        for x in negatyw_zaniedbania_db:
            if nigdy_zawsze_labels[i] == x[0]:
                negatyw_zaniedbania_values[i] = x[1]

    negatyw_gra_nad_inne_aktywnosci_db = db.session.query(Games.negatyw_gra_nad_inne_aktywnosci, db.func.count(
        Games.negatyw_gra_nad_inne_aktywnosci)).group_by(Games.negatyw_gra_nad_inne_aktywnosci).all()
    negatyw_gra_nad_inne_aktywnosci_values = [0, 0, 0, 0, 0]
    for i in range(0, 5):
        for x in negatyw_gra_nad_inne_aktywnosci_db:
            if nigdy_zawsze_labels[i] == x[0]:
                negatyw_gra_nad_inne_aktywnosci_values[i] = x[1]

    # POZYTYWNE

    pozytyw_koordynacja_db = db.session.query(Games.pozytyw_koordynacja, db.func.count(
        Games.pozytyw_koordynacja)).group_by(Games.pozytyw_koordynacja).all()
    pozytyw_koordynacja_values = [0, 0, 0, 0, 0]
    for i in range(0, 5):
        for x in pozytyw_koordynacja_db:
            if skala_labels[i] == x[0]:
                pozytyw_koordynacja_values[i] = x[1]

    pozytyw_samokontrola_db = db.session.query(db.func.count(
        Games.pozytyw_samokontrola)).group_by(Games.pozytyw_samokontrola).all()
    pozytyw_samokontrola_values = []
    for value, in pozytyw_samokontrola_db:
        pozytyw_samokontrola_values.append(value)

    pozytyw_koncentracja_db = db.session.query(Games.pozytyw_koncentracja, db.func.count(
        Games.pozytyw_koncentracja)).group_by(Games.pozytyw_koncentracja).all()
    pozytyw_koncentracja_values = [0, 0, 0, 0, 0]
    for i in range(0, 5):
        for x in pozytyw_koncentracja_db:
            if skala_labels[i] == x[0]:
                pozytyw_koncentracja_values[i] = x[1]

    gra_prof_db = db.session.query(db.func.count(
        Games.gra_prof)).group_by(Games.gra_prof).all()
    gra_prof_values = []
    for value, in gra_prof_db:
        gra_prof_values.append(value)

    gra_typ_db = db.session.query(db.func.count(
        Games.gra_typ)).group_by(Games.gra_typ).all()
    gra_typ_values = []
    for value, in gra_typ_db:
        gra_typ_values.append(value)

    uzaleznienie_db = db.session.query(db.func.count(
        Info.uzaleznienie_model)).group_by(Info.uzaleznienie_model).all()
    uzaleznienie_values = []
    for value, in uzaleznienie_db:
        uzaleznienie_values.append(value)
    uzaleznienie_procent = round(
        (uzaleznienie_values[1]*100)/sum(uzaleznienie_values), 1)

    return render_template('danger.html', gra_w_ciagu_12=json.dumps(gra_w_ciagu_12), gra_gniew_values=json.dumps(gra_gniew_values), negatyw_zaniedbania_values=json.dumps(negatyw_zaniedbania_values), negatyw_gra_nad_inne_aktywnosci_values=json.dumps(negatyw_gra_nad_inne_aktywnosci_values), pozytyw_koordynacja_values=json.dumps(pozytyw_koordynacja_values), pozytyw_koncentracja_values=json.dumps(pozytyw_koncentracja_values), pozytyw_samokontrola_values=json.dumps(pozytyw_samokontrola_values), gra_typ_values=json.dumps(gra_typ_values), uzaleznienie_values=json.dumps(uzaleznienie_values), uzaleznienie_procent=uzaleznienie_procent)
