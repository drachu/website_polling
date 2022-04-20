from imp import reload
from unicodedata import name
from run import app
from flask import render_template, redirect, request, url_for, flash
from website.forms import GamesForm, InfoForm
from website.models import Info, Games
from website import db

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/form',methods=['GET', 'POST'])
def form_page():
    infoForm = InfoForm()
    gamesForm = GamesForm()
    if request.method == 'POST':
        
        if infoForm.validate_on_submit() and infoForm.gra_w_ciagu_12.data == 'nie':
            user_poll_info = Info(rok=infoForm.rok.data, zwiazek=infoForm.zwiazek.data, praca=infoForm.praca.data, gra_w_ciagu_12=infoForm.gra_w_ciagu_12.data,)
            db.session.add(user_poll_info)
            db.session.commit()
            return redirect(url_for('thanks_page'))
        elif gamesForm.validate_on_submit():
            user_poll_info = Info(rok=infoForm.rok.data, zwiazek=infoForm.zwiazek.data, praca=infoForm.praca.data, gra_w_ciagu_12=infoForm.gra_w_ciagu_12.data,)
            db.session.add(user_poll_info)
            db.session.commit()
            user_poll_games= Games(gra_prof=gamesForm.gra_prof.data, gra_tydz=gamesForm.gra_tydz.data, gra_tydz_weekend=gamesForm.gra_tydz_weekend.data,gra_typ=gamesForm.gra_typ.data ,gra_ulubiona=gamesForm.gra_ulubiona.data, gra_platforma=gamesForm.gra_platforma.data, gra_klan=gamesForm.gra_klan.data, gra_grind=gamesForm.gra_grind.data, pandemia_start=gamesForm.pandemia_start.data, pandemia_gra_wiecej=gamesForm.pandemia_gra_wiecej.data, pandemia_czas=gamesForm.pandemia_czas.data, pozytyw_ucieczka=gamesForm.pozytyw_ucieczka.data, pozytyw_samokontrola=gamesForm.pozytyw_samokontrola.data, pozytyw_koncentracja=gamesForm.pozytyw_koncentracja.data, pozytyw_koordynacja=gamesForm.pozytyw_koordynacja.data, negatyw_gra_pomimo_konsekwencji=gamesForm.negatyw_gra_pomimo_konsekwencji.data, negatyw_gra_dluzej=gamesForm.negatyw_gra_dluzej.data, negatyw_gra_nad_inne_aktywnosci=gamesForm.negatyw_gra_nad_inne_aktywnosci.data, negatyw_zaniedbania=gamesForm.negatyw_zaniedbania.data, negatyw_gniew=gamesForm.negatyw_gniew.data, tow_kontakty=gamesForm.tow_kontakty.data, tow_inni_sie_przejmuja=gamesForm.tow_inni_sie_przejmuja.data, tow_brak_towarzystwa=gamesForm.tow_brak_towarzystwa.data, tow_izolacja=gamesForm.tow_izolacja.data, tow_przytloczenie=gamesForm.tow_przytloczenie.data, user_id=user_poll_info.id)
            db.session.add(user_poll_games)
            print(user_poll_info.id)
            print(user_poll_games.user_id)
            db.session.commit()
            return redirect(url_for('thanks_page'))
        else:
            flash('Uzupełnij odpowiedź!', category='error')
        

        
    return render_template('form.html', infoForm=infoForm, gamesForm=gamesForm)


@app.route('/thanks')
def thanks_page():
    return render_template('thanks.html')