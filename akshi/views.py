from flask import render_template, flash, redirect
from akshi import  app


@app.route('/')
@app.route('/index')
def index():
	user ={'name':'Gopi'}
	return render_template('index.html', title='Akshi', user=user)


