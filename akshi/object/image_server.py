from flask import Flask
from flask import send_file

app = Flask(__name__)
current_frame_path = 'current_frame.png'

@app.route('/')
def get_image():
    return send_file(current_frame_path, mimetype='image/png')
 