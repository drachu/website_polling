from run import app
from flask import render_template
from website.forms import pytanieForm
from website.models import Poll

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/form',methods=['GET', 'POST'])
def form_page():
    form = pytanieForm()
    if form.validate_on_submit():
        pass
    return render_template('form.html', form=form)


@app.route('/thanks')
def thanks_page():
    return render_template('thanks.html')