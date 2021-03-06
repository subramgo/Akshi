import os
basedir = os.path.abspath(os.path.dirname(__file__))

print(basedir)
WTF_CSRF_ENABLED = True
SECRET_KEY = 'you-will-never-guess'

OPENID_PROVIDERS = [
    {'name': 'Google', 'url': 'https://www.google.com/accounts/o8/id'},
    {'name': 'Yahoo', 'url': 'https://me.yahoo.com'},
    {'name': 'AOL', 'url': 'http://openid.aol.com/<username>'},
    {'name': 'Flickr', 'url': 'http://www.flickr.com/<username>'},
    {'name': 'MyOpenID', 'url': 'https://www.myopenid.com'}]

SQLALCHEMY_DATABASE_URI =  \
        'sqlite:///' + os.path.join(basedir, 'akshi.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False
UPLOADED_PHOTOS_DEST=os.path.join(basedir + '/uploads')
FACE_RECO_MODEL=os.path.join(basedir + '/akshi/face/bin/facerecog_2.h5')
