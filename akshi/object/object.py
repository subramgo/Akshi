from flask import render_template, flash, redirect, Blueprint, request, send_file
import json
import numpy as np 
import cv2
from .models import Person, object_db
from sqlalchemy import func
from flask import current_app as app
import random

from PIL import Image
import requests
from io import BytesIO
from .object_monitor import Yolo_detector
import scipy.misc

object_api = Blueprint('object', __name__, url_prefix='/api/v1/object', template_folder = 'templates', static_folder = 'static')
object_detector = Yolo_detector()


@object_api.route('/', methods =['GET'])
def object():
	pers = Person.query.order_by(Person.date_created.desc()).limit(50).all()

	return render_template('list_objects.html',  liveurl = app.config['LIVE_CAMERA'],  persons = pers, objecturl = '/api/v1/object/detect')


@object_api.route('/detect', methods=['GET'])
def detect():

	url = app.config['LIVE_CAMERA'] + str(random.randint(0,1000))
	response = requests.get(url,auth=("root", "pass"))
	data = response.content
	nparr = np.fromstring(data, np.uint8)
	image_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	# Run YOLO
	image_array = scipy.misc.toimage(Image.fromarray(image_arr))
	annotated = object_detector.process(image_array)





	bIO = BytesIO()
	annotated.save(bIO, 'PNG')

	per_obj = Person(person_count = object_detector.NoPersons)
	object_db.session.add(per_obj)
	object_db.session.commit()

	bIO.seek(0)
	return send_file(bIO, mimetype='image/png')
