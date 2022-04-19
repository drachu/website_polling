from unicodedata import name
from run import app
from flask import render_template, redirect, request, url_for, flash
from website.forms import pytanieForm
from website.models import Poll
from website import db

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/form',methods=['GET', 'POST'])
def form_page():
    form = pytanieForm()
    if request.method == 'POST':
        if form.validate_on_submit():
            user_poll = Poll(rok=form.rok.data, zwiazek=form.zwiazek.data, praca=form.praca.data, gra_w_ciagu_12=form.gra_w_ciagu_12.data, gra_prof=form.gra_prof.data, gra_tydz=form.gra_tydz.data, gra_tydz_weekend=form.gra_tydz_weekend.data,gra_typ=form.gra_typ.data ,gra_ulubiona=form.gra_ulubiona.data, gra_platforma=form.gra_platforma.data, gra_klan=form.gra_klan.data, gra_grind=form.gra_grind.data, pandemia_start=form.pandemia_start.data, pandemia_gra_wiecej=form.pandemia_gra_wiecej.data, pandemia_czas=form.pandemia_czas.data, pozytyw_ucieczka=form.pozytyw_ucieczka.data, pozytyw_emocje=form.pozytyw_emocje.data, pozytyw_samokontrola=form.pozytyw_samokontrola.data, pozytyw_koncentracja=form.pozytyw_koncentracja.data, pozytyw_koordynacja=form.pozytyw_koordynacja.data, negatyw_gra_pomimo_konsekwencji=form.negatyw_gra_pomimo_konsekwencji.data, negatyw_gra_dluzej=form.negatyw_gra_dluzej.data, negatyw_gra_nad_inne_aktywnosci=form.negatyw_gra_nad_inne_aktywnosci.data, negatyw_zaniedbania=form.negatyw_zaniedbania.data, negatyw_gniew=form.negatyw_gniew.data, tow_kontakty=form.tow_kontakty.data, tow_inni_sie_przejmuja=form.tow_inni_sie_przejmuja.data, tow_brak_towarzystwa=form.tow_brak_towarzystwa.data, tow_izolacja=form.tow_izolacja.data, tow_przytloczenie=form.tow_przytloczenie.data)
            db.session.add(user_poll)
            db.session.commit()
       
            return redirect(url_for('thanks_page'))
        else:
            flash('Uzupełnij odpowiedź!', category='error')
    return render_template('form.html', form=form)


@app.route('/thanks')
def thanks_page():
    return render_template('thanks.html')