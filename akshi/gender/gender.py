from flask import render_template, flash, redirect, Blueprint, request
import json
import numpy as np 
import cv2
from .Classifier import *
from .models import Gender, gender_db
from sqlalchemy import func
from flask import current_app as app



predictor = GenderClassifier()

gender_api = Blueprint('gender', __name__, url_prefix='/api/v1/gender', template_folder = 'templates', static_folder = 'static')


@gender_api.route('/', methods =['GET'])
def gender():
	return "Hello Gender"
 

@gender_api.route('/face', methods=['GET','POST'])
def from_face():
    data = request.data
    nparr = np.fromstring(data, np.uint8)
    image_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    pred_val = predictor.process(x_test=image_arr, y_test=None, batch_size=1)
    gen_obj = Gender(gender = pred_val)
    gender_db.session.add(gen_obj)
    gender_db.session.commit()


    return json.dumps({'gender': pred_val})


@gender_api.route('/face/cam1', methods=['GET','POST'])
def from_cam1():
    print(request.files)
    img_file = request.files.get('image_0')
    data = img_file.read()
    nparr = np.fromstring(data, np.uint8)
    image_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    pred_val = predictor.process(x_test=image_arr, y_test=None, batch_size=1)
    gen_obj = Gender(gender = pred_val)
    gender_db.session.add(gen_obj)
    gender_db.session.commit()


    return json.dumps({'gender': pred_val})





@gender_api.route('/list', methods=['GET'])
def list_genders():
	gends = Gender.query.all()
	output_dict = {}
	summ = gender_db.session.query(Gender.gender,func.count(Gender.id)).group_by(Gender.gender).all()
	wkly =gender_db.session.query(func.strftime('%W', Gender.date_created), Gender.gender, func.count(Gender.id)).group_by( Gender.gender,func.strftime('%W', Gender.date_created)).all()
	return render_template('list_gender.html', title='Gender Dashboard', genders = gends, summ = summ, weekly = wkly, liveurl = app.config['LIVE_CAMERA'])
