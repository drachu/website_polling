from xml.dom import ValidationErr
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, RadioField
from wtforms.validators import Length, EqualTo, Email, DataRequired, ValidationError


class pytanieForm(FlaskForm):
    #Informacje o osobie
    rok = SelectField(label="Podaj rok studiów: ", choices=[("1", "1"), ("2", "2"), ("3", "3"), ("4","4"), ("5", "5")], validators=[DataRequired()])
    zwiazek = SelectField(label="Czy jesteś w związku: ", choices=[("tak", "Tak"), ("nie", "Nie"), ("-", "To skomplikowane"), ("-", "Nie chcę odpowiadać")], validators=[DataRequired()])
    praca = RadioField(label="Jesteś obecnie zatrudniona/y?", choices=[("tak", "Tak"), ("nie", "Nie")], validators=[DataRequired()])

    submit = SubmitField(label="Zakończ")


