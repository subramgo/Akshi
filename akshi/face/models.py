from flask_sqlalchemy import SQLAlchemy

face_db = SQLAlchemy()

class Face(face_db.Model):
    id = face_db.Column(face_db.Integer, primary_key=True, autoincrement=True)
    date_created = face_db.Column(face_db.DateTime, default=face_db.func.current_timestamp())
    no_faces = face_db.Column(face_db.Integer)

    def __repr__(self):
        return '<id {} date_created {} no_faces {}>'.format(self.id, self.date_created, self.no_faces)

class Name(face_db.Model):
    id = face_db.Column(face_db.Integer, primary_key=True, autoincrement=True)
    date_created = face_db.Column(face_db.DateTime, default=face_db.func.current_timestamp())
    Name_id = face_db.Column(face_db.Integer)
    Name_vector = face_db.Column(face_db.Float)

    def __repr__(self):
        return '<id {} date_created {} Name_id {} Name_vector {}>'.format(self.id, self.date_created, self.Name_id, self.Name_vector)



