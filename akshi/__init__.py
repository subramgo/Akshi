from flask import Flask 
from flask_bootstrap import Bootstrap 
from akshi.gender.gender import gender_api
from akshi.gender.models import gender_db,Gender
from akshi.face.models import face_db, Face
from akshi.face.face import face_api
from akshi.object.object import object_api
from akshi.object.models import object_db, Person



from flask_uploads import UploadSet, configure_uploads, IMAGES



app = Flask(__name__)
app.config.from_object('config') 

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)


# Database
gender_db.init_app(app)
gender_db.create_all(app=app)

face_db.init_app(app)
face_db.create_all(app=app)

object_db.init_app(app)
object_db.create_all(app=app)

# Gender detection apis
app.register_blueprint(gender_api, url_prefix='/api/v1/gender')
# Face detection api
app.register_blueprint(face_api, url_prefix='/api/v1/face')
# object detection api
app.register_blueprint(object_api, url_prefix='/api/v1/object')


Bootstrap(app)



import akshi.views


