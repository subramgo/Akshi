from flask_sqlalchemy import SQLAlchemy

object_db = SQLAlchemy()

class Person(object_db.Model):
    id = object_db.Column(object_db.Integer, primary_key=True, autoincrement=True)
    date_created = object_db.Column(object_db.DateTime, default=object_db.func.current_timestamp())
    person_count = object_db.Column(object_db.Integer)


    def __repr__(self):

        return '<id {} date_created {} gender {}>'.format(self.id, self.date_created, self.person_count)