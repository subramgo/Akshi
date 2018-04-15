import io
import os
import glob
from time import sleep
import requests




gender_url = "http://127.0.0.1:5000/api/v1/gender/face"
face_url = "http://127.0.0.1:5000/api/v1/face/detect_faces"

headers={'Content-Type': 'application/octet-stream'}

imagePath = '/Users/gsubramanian/Documents/Personel/Tech/blogs/Projects/DeepLearning/python/data/facerecog/*.jpg'


imageList = glob.glob(imagePath)
for file_name in imageList:
        with io.open(file_name, 'rb') as image_file:
            content = image_file.read()


        response = requests.post(url=gender_url, data = content, json = None, headers = headers)
        gender = response.json()['gender']
        no_faces = None
        response = requests.post(url=face_url, data = content, json = None, headers = headers)
        no_faces = response.json()['no_faces']


        print(file_name, gender, no_faces)




