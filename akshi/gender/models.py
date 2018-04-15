from flask_sqlalchemy import SQLAlchemy

gender_db = SQLAlchemy()

class Gender(gender_db.Model):
    id = gender_db.Column(gender_db.Integer, primary_key=True, autoincrement=True)
    date_created = gender_db.Column(gender_db.DateTime, default=gender_db.func.current_timestamp())
    gender = gender_db.Column(gender_db.String(6))


    def __repr__(self):

        return '<id {} date_created {} gender {}>'.format(self.id, self.date_created, self.gender)