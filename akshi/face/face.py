from flask import render_template, flash, redirect, Blueprint, request, send_from_directory
import json
import numpy as np 
import cv2
from .models import Face, face_db
from sqlalchemy import func
import sys
import dlib
from flask import current_app as app
import os
from .FaceReco import *


face_api = Blueprint('face', __name__, url_prefix='/api/v1/face', template_folder = 'templates', static_folder = 'static')
face_detector = dlib.get_frontal_face_detector()
face_recog = FaceReco()

@face_api.route('/', methods =['GET'])
def face_page():
    return render_template('face.html',title='Face Recogniton API',gender=1)
 
@face_api.route('/detect_faces',methods=['GET','POST'])
def detect_faces():
    data = request.data
    nparr = np.fromstring(data, np.uint8)
    image_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #
    dets = face_detector(image_arr, 1)
    no_faces = len(dets)
    face_obj = Face(no_faces = no_faces)
    face_db.session.add(face_obj)
    face_db.session.commit()
    return json.dumps({"no_faces":no_faces})



@face_api.route('/match', methods=['GET','POST'])
def match():
    if request.method == 'POST' and 'image-face' in request.files:
        print(request.files)
        files = request.files.getlist("image-face")
        
        face_recog.uploadpath = app.config['UPLOADED_PHOTOS_DEST']
        face_recog.process(files[0], files[1])
        similarity = face_recog.similarity
        filename1 =files[0].filename 
        filename2 =files[1].filename


    return render_template('face.html', filename1=filename1,filename2=filename2, face=1,similarity=similarity)

@face_api.route('/upload', methods=['GET', 'POST'])
def upload():
    filename =None
    rfilename =None

    file = None
    print(request.files)

    if request.method == 'POST' and 'image-gender' in request.files:
        file = request.files['image-gender']
        filename = file.filename
        
        # Save the file
        data = file.read()
        nparr = np.fromstring(data, np.uint8)

        o_image_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_arr = o_image_arr.copy()
        #
        dets = face_detector(image_arr, 1)
        no_faces = len(dets)

        for i, d in enumerate(dets):
            cv2.rectangle(image_arr, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 255), 2)
        
        rfilename = 'r' + filename
        
        cv2.imwrite(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], rfilename), image_arr)
        cv2.imwrite(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename), o_image_arr)

        #file.save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
        

    return render_template('face.html', filename=filename,rfilename=rfilename, gender=1)



@face_api.route('/uploads/<filename>')
def send_file(filename):
    full_path = app.config['UPLOADED_PHOTOS_DEST']
    return send_from_directory(full_path, filename)

