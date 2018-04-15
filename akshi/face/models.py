from flask_sqlalchemy import SQLAlchemy

face_db = SQLAlchemy()

class Face(face_db.Model):
    id = face_db.Column(face_db.Integer, primary_key=True, autoincrement=True)
    date_created = face_db.Column(face_db.DateTime, default=face_db.func.current_timestamp())
    no_faces = face_db.Column(face_db.Integer)


    def __repr__(self):
        return '<id {} date_created {} no_faces {}>'.format(self.id, self.date_created, self.no_faces)