from keras.models import load_model
import os
import cv2
import numpy as np

class  GenderClassifier():
    gender_filter = {0:'male',1:'female'}


    def __init__(self):
        self.x_test = None
        self.preprocessed = None
        self.y_test = None
        self.batch_size = 1
        self.model = None
        self.scores = None
        self.predictions = None
        target_size = (100,100)
        self.image_w = target_size[0]
        self.image_h = target_size[1]
        self.gender_session = None
        self.__load_model()

    def __load_model(self):
      self.model = load_model('./akshi/gender/bin/gender_1.h5')
      print("Gender Model Loaded")

    def __evaluate(self):
        self.scores = self.model.evaluate(self.preprocessed, self.y_test, batch_size = self.batch_size)
    
    def __predict(self):
        print(self.preprocessed.shape)
        #with self.gender_session.as_default():
        self.predictions = self.model.predict(self.preprocessed, batch_size=self.batch_size)

    def __cleanup(self):
        del self.model


    def input_preprocessing(self):
        """ Preprocessing to match the training conditions for this model. 
        Apply resize, reshape, other scaling/whitening effects.
        x_test can be any image size greater than 100x100 and it will be resized
        """
        resized = cv2.resize(self.x_test, (self.image_w, self.image_h)) 
        self.preprocessed = resized.reshape(1,self.image_w,self.image_h,3)



    def process(self, x_test, y_test , batch_size):
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.input_preprocessing()

        if y_test is not None:
            self.__evaluate()
            #print("Score {}".format(self.scores[1]))
            return None

        else:
            self.__predict()
            #print(self.predictions)
            idx = np.argmax(self.predictions)

            return self.gender_filter[idx]

    def process1(self, x_test, y_test , batch_size):
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.input_preprocessing()

        if y_test is not None:
            self.__evaluate()
            #print("Score {}".format(self.scores[1]))
            return None

        else:
            self.__predict()
            #print(self.predictions)
            idx = np.argmax(self.predictions)

            return self.gender_filter[idx],self.predictions
