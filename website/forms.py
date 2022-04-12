from xml.dom import ValidationErr
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, RadioField
from wtforms.validators import Length, EqualTo, Email, DataRequired, ValidationError


class pytanieForm(FlaskForm):
    #Informacje o osobie
    rok = SelectField(label="Podaj rok studiów: ", choices=[("1", "1"), ("2", "2"), ("3", "3"), ("4","4"), ("5", "5")], validators=[DataRequired()])
    zwiazek = SelectField(label="Czy jesteś w związku: ", choices=[("tak", "Tak"), ("nie", "Nie"), ("-", "To skomplikowane"), ("-", "Nie chcę odpowiadać")], validators=[DataRequired()])
    praca = RadioField(label="Jesteś obecnie zatrudniona/y?", choices=[("tak", "Tak"), ("nie", "Nie")], validators=[DataRequired()])

    #Czas gry
    #gra_w_ciagu_12 = Integer()
    gra_prof = RadioField(label="Czy grasz profesjonalnie? (e-sport, streaming itp.)", choices=[("tak", "Tak"), ("nie", "Nie")], validators=[DataRequired()])
    gra_ulubiona = StringField(label="Jaka jest Twoja ostatnia ulubiona gra, w której najbardziej lubisz spędzać czas?", validators=[DataRequired()])
    gra_platforma = SelectField(label="Na jakiej platformie spędzasz najwięcej czasu?", choices=[("pc", "PC"), ("konsola", "Konsola (XBOX/PS)"), ("konsola_mobilna", "Konsola mobilna (np. Nintendo Switch)"), ("mobile", "Gry mobilne")], validators=[DataRequired()])
    gra_klan = RadioField(label="Czy jestes aktualnie w jakims klanie/drużynie/gildii gdzie regularnie grasz razem z innymi?", choices=[("tak", "Tak"), ("nie", "Nie")], validators=[DataRequired()])
    gra_grind = RadioField(label="W grach, w których spędzasz czas pojawia się zjawisko 'grindu'?", choices=[("tak", "Tak"), ("nie", "Nie")], validators=[DataRequired()])

    #Gra w pandemii
    pandemia_start = SelectField(label="Czy zacząłeś grać w okresie pandemii koronawirusa?", choices=[("poza", "Przed lub po pandemii"), ("w_czasie", "W czasie pandemii")], validators=[DataRequired()])
    pandemia_gra_wiecej = RadioField(label="Okres pandemii sprawił, że częściej grałeś/aś w gry?", choices=[("tak", "Tak"), ("nie", "Nie")], validators=[DataRequired()])
    pandemia_czas = RadioField(label="Spędzając ostatnie lata pandemii w domu miałeś/aś więcej wolnego czasu?", choices=[("tak", "Tak"), ("nie", "Nie")], validators=[DataRequired()])

    #Pozytywne odczucia
    pozytyw_ucieczka = RadioField(label="Gdy pozwalają Ci 'uciec' od problemów czy zmartwień? Potrafisz się dzięki nim w takich sytuacjach zrelaksować?", choices=[("tak", "Tak"), ("nie", "Nie")], validators=[DataRequired()])
    #pozytyw_emocje = RadioField(label="Jakie emocje zdarzało Ci się odczuć podczas rozgrywki?", choices=[(["radosc", "hej", "elo"], "Radność"), ("gniew", "Gniew"), ("smutek", "Smutek"), ("strach", "Strach")], validators=[DataRequired()])
    pozytyw_samokontrola = RadioField(label="Czy grajać w gry potrafileś/aś zachować samokontrolę?", choices=[("tak", "Tak"), ("nie", "Nie")], validators=[DataRequired()])
    pozytyw_koncentracja = RadioField(label="większą koncentrację:", choices=[("1", "1"), ("2", "2"), ("3", "3"), ("4","4"), ("5", "5")], validators=[DataRequired()])
    pozytyw_koordynacja = RadioField(label="polepszoną koordynację:", choices=[("1", "1"), ("2", "2"), ("3", "3"), ("4","4"), ("5", "5")], validators=[DataRequired()])

    #Negatywne odczucia
    negatyw_gra_pomimo_konsekwencji = RadioField(label="Zdarzało Ci się grać pomimo widocznych negatywnych konsekwencji?", choices=[("nigdy", "Nigdy"), ("rzadko", "Rzadko"), ("czasami", "Czasami"), ("czesto","Często"), ("zawsze", "Zawsze")], validators=[DataRequired()])
    negatyw_gra_dluzej = RadioField(label="Zdarzało Ci się grać dłużej niż planowałeś?", choices=[("nigdy", "Nigdy"), ("rzadko", "Rzadko"), ("czasami", "Czasami"), ("czesto","Często"), ("zawsze", "Zawsze")], validators=[DataRequired()])
    negatyw_gra_nad_inne_aktywnosci = RadioField(label="Więcej uwagi poświecasz grom niż innym zainteresowaniom?", choices=[("nigdy", "Nigdy"), ("rzadko", "Rzadko"), ("czasami", "Czasami"), ("czesto","Często"), ("zawsze", "Zawsze")], validators=[DataRequired()])
    negatyw_zaniedbania = RadioField(label="Czy grając w gry zaniedbywałeś/aś naukę lub inne obowiązki?", choices=[("nigdy", "Nigdy"), ("rzadko", "Rzadko"), ("czasami", "Czasami"), ("czesto","Często"), ("zawsze", "Zawsze")], validators=[DataRequired()])
    negatyw_gniew = RadioField(label="Zdarzało Ci sie wpadać w gniew podczas gry i porażki?", choices=[("nigdy", "Nigdy"), ("rzadko", "Rzadko"), ("czasami", "Czasami"), ("czesto","Często"), ("zawsze", "Zawsze")], validators=[DataRequired()])

    #Stosunki z innymi, deprecha
    
    tow_kontakty = RadioField(label="Zdarzalo Ci się grać pomimo widocznych negatywnych konsekwencji?", choices=[("1", "1"), ("2", "2"), ("3", "3"), ("4","4"), ("5", "5")], validators=[DataRequired()])
    tow_inni_sie_przejmuja = RadioField(label="Czujesz, że inni nie za bardzo się Tobą przejmują?", choices=[("nigdy", "Nigdy"), ("rzadko", "Rzadko"), ("czasami", "Czasami"), ("czesto","Często"), ("zawsze", "Zawsze")], validators=[DataRequired()])
    tow_brak_towarzystwa = RadioField(label="Brakuje Ci towarzystwa?", choices=[("nigdy", "Nigdy"), ("rzadko", "Rzadko"), ("czasami", "Czasami"), ("czesto","Często"), ("zawsze", "Zawsze")], validators=[DataRequired()])
    tow_izolacja = RadioField(label="Czujesz się odizolowany od innych?", choices=[("nigdy", "Nigdy"), ("rzadko", "Rzadko"), ("czasami", "Czasami"), ("czesto","Często"), ("zawsze", "Zawsze")], validators=[DataRequired()])
    tow_przytloczenie = RadioField(label="Czy czujesz się przytłoczony grając w gry?", choices=[("nigdy", "Nigdy"), ("rzadko", "Rzadko"), ("czasami", "Czasami"), ("czesto","Często"), ("zawsze", "Zawsze")], validators=[DataRequired()])



    submit = SubmitField(label="Zakończ")


