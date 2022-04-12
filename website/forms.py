from xml.dom import ValidationErr
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import Length, EqualTo, Email, DataRequired, ValidationError


class pytanieForm(FlaskForm):
    odpowiedz = StringField(label)


class end_polling_form(FlaskForm):

    submit = SubmitField(label="Zakończ")
