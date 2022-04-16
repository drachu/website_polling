from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = '64ceda6a59a769a1515de081'

#aby stworzyc baze: python: from website import db; db.create_all()
#aby usunac baze: python: from website import db; db.drop_all()

db = SQLAlchemy(app)

from website import routes