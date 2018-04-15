import os
from keras.models import load_model
import glob
import cv2
import numpy as np
from .align_dlib import AlignDlib
from scipy.spatial.distance import cosine
from flask import current_app as app

crop_dim = (96,96)

def triplet_loss(y_true, y_pred):
  import tensorflow as tf
  anchor = y_pred[:,0]
  positive = y_pred[:,1]
  negative = y_pred[:,2]
  #anchor, positive, negative = y_pred
  alpha = 0.2
  pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
  neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
  basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
  loss = tf.maximum(tf.reduce_mean(basic_loss), 0.0)
  return loss




class FaceReco():
  def __init__(self, uploadpath=None):
    modelpath = './akshi/face/bin/facerecog_2.h5'
    full_model = load_model(modelpath,custom_objects={'triplet_loss': triplet_loss})
    print("Loaded Siamese Model")

    self.model = full_model.get_layer('model_1')
    self.similarity = None
    self.uploadpath = uploadpath
    self.align_dlib = AlignDlib()

  
  def process(self,file1,file2):
    face1 = file1.read()
    face2 = file2.read()

    f1arr = np.fromstring(face1, np.uint8)
    f2arr  = np.fromstring(face2, np.uint8)

    f1_image = cv2.imdecode(f1arr, cv2.IMREAD_COLOR)
    f2_image = cv2.imdecode(f2arr, cv2.IMREAD_COLOR)


    #Preprocess image
    f1_aligned = self.align_dlib.align(crop_dim, f1_image)
    f2_aligned = self.align_dlib.align(crop_dim, f2_image)

    f1_resized = np.expand_dims(f1_aligned, axis =0)
    f2_resized = np.expand_dims(f2_aligned, axis =0)



    f1_vector = self.model.predict_on_batch(f1_resized)
    f2_vector = self.model.predict_on_batch(f2_resized)


    distance = cosine(f1_vector, f2_vector, )
    self.similarity = np.round((1 - distance) * 100,2)

    cv2.imwrite(os.path.join(self.uploadpath, file1.filename), f1_image)
    cv2.imwrite(os.path.join(self.uploadpath, file2.filename), f2_image)






    




