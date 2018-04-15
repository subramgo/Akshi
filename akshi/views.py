from flask import render_template, flash, redirect
from akshi import  app
from .forms import LoginForm


@app.route('/')
@app.route('/index')
def index():
	user ={'name':'Gopi'}
	return render_template('index.html', title='Akshi', user=user)


@app.route('/login',methods=['GET', 'POST'])
def login():
	form = LoginForm()
	if form.validate_on_submit():
		flash('Login requested for OpenId ="%s", remember_me=%s' %
			(form.openid.data, str(form.remember_me.data)))
		return redirect('/index')
	return render_template('login.html', title='sign in', form=form, providers=app.config['OPENID_PROVIDERS'])
