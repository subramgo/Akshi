import argparse
import colorsys
import imghdr
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
#from DLUtils.cellar.yolo import yolokeras
#from DLUtils.cellar.yolo.yolokeras import yolo_eval, yolo_head
#from DLUtils import datafeed
import scipy.misc
import cv2
""" Implementation of the YOLO network using Keras """
from keras import backend as K
from keras.layers import Lambda
from keras.models import load_model
import tensorflow as tf
import requests


###########################################################
#####          Model Inference Parameters             #####
###########################################################
score_threshold = 0.3
iou_threshold = 0.5

anchors = np.array(
    [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]])

class_names = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
# Generate colors for drawing bounding boxes.
hsv_tuples = [(x / len(class_names), 1., 1.)
              for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))
random.seed(10101)  # Fixed seed for consistent colors across runs.
random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
random.seed(None)  # Reset seed to default.


class Yolo_detector():

    def __init__(self):
        self.yolo_model = None
        self.model_image_size = None 
        self.is_fixed_size = None

        self.NoPersons = 0

        self.sess = K.get_session()
        self.yolo_model = load_model('./akshi/object/bin/tiny_yolo.h5')

        num_classes = len(class_names)
        num_anchors = len(anchors)

        # TODO: Assumes dim ordering is channel last
        model_output_channels = self.yolo_model.layers[-1].output_shape[-1]
        assert model_output_channels == num_anchors * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes. ' \
            'Specify matching anchors and classes with --anchors_path and ' \
            '--classes_path flags.'
        print('Tiny Yolo model, anchors, and classes loaded.')

        ### Check if model is fully convolutional, assuming channel last order.
        self.model_image_size = self.yolo_model.layers[0].input_shape[1:3]
        self.is_fixed_size = self.model_image_size != (None, None)


    def yolo_head(self, feats, anchors, num_classes):
        """Convert final layer features to bounding box parameters.

        Parameters
        ----------
        feats : tensor
            Final convolutional layer features.
        anchors : array-like
            Anchor box widths and heights.
        num_classes : int
            Number of target classes.

        Returns
        -------
        box_xy : tensor
            x, y box predictions adjusted by spatial location in conv layer.
        box_wh : tensor
            w, h box predictions adjusted by anchors and conv spatial resolution.
        box_conf : tensor
            Probability estimate for whether each box contains any object.
        box_class_pred : tensor
            Probability distribution estimate for each box over class labels.
        """
        num_anchors = len(anchors)
        # Reshape to batch, height, width, num_anchors, box_params.
        anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])

        # Static implementation for fixed models.
        # TODO: Remove or add option for static implementation.
        # _, conv_height, conv_width, _ = K.int_shape(feats)
        # conv_dims = K.variable([conv_width, conv_height])

        # Dynamic implementation of conv dims for fully convolutional model.
        conv_dims = K.shape(feats)[1:3]  # assuming channels last
        # In YOLO the height index is the inner most iteration.
        conv_height_index = K.arange(0, stop=conv_dims[0])
        conv_width_index = K.arange(0, stop=conv_dims[1])
        conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

        # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
        # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
        conv_width_index = K.tile(
            K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
        conv_width_index = K.flatten(K.transpose(conv_width_index))
        conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
        conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
        conv_index = K.cast(conv_index, K.dtype(feats))

        feats = K.reshape(
            feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
        conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

        # Static generation of conv_index:
        # conv_index = np.array([_ for _ in np.ndindex(conv_width, conv_height)])
        # conv_index = conv_index[:, [1, 0]]  # swap columns for YOLO ordering.
        # conv_index = K.variable(
        #     conv_index.reshape(1, conv_height, conv_width, 1, 2))
        # feats = Reshape(
        #     (conv_dims[0], conv_dims[1], num_anchors, num_classes + 5))(feats)

        box_xy = K.sigmoid(feats[..., :2])
        box_wh = K.exp(feats[..., 2:4])
        box_confidence = K.sigmoid(feats[..., 4:5])
        box_class_probs = K.softmax(feats[..., 5:])

        # Adjust preditions to each spatial grid point and anchor size.
        # Note: YOLO iterates over height index before width index.
        box_xy = (box_xy + conv_index) / conv_dims
        box_wh = box_wh * anchors_tensor / conv_dims

        return box_xy, box_wh, box_confidence, box_class_probs


    def yolo_boxes_to_corners(self, box_xy, box_wh):
        """Convert YOLO box predictions to bounding box corners."""
        box_mins = box_xy - (box_wh / 2.)
        box_maxes = box_xy + (box_wh / 2.)

        return K.concatenate([
            box_mins[..., 1:2],  # y_min
            box_mins[..., 0:1],  # x_min
            box_maxes[..., 1:2],  # y_max
            box_maxes[..., 0:1]  # x_max
        ])

    def yolo_filter_boxes(self, boxes, box_confidence, box_class_probs, threshold=.6):
        """Filter YOLO boxes based on object and class confidence."""
        from keras import backend as K

        box_scores = box_confidence * box_class_probs
        box_classes = K.argmax(box_scores, axis=-1)
        box_classes = tf.cast(box_classes, tf.float32)
        box_class_scores = K.max(box_scores, axis=-1)

        prediction_mask = box_class_scores >= threshold

        # TODO: Expose tf.boolean_mask to Keras backend?
        boxes = tf.boolean_mask(boxes, prediction_mask)
        scores = tf.boolean_mask(box_class_scores, prediction_mask)
        classes = tf.boolean_mask(box_classes, prediction_mask)

        return boxes, scores, classes


    def yolo_eval(self, yolo_outputs,
                  image_shape,
                  max_boxes=10,
                  score_threshold=.6,
                  iou_threshold=.5):
        """Evaluate YOLO model on given input batch and return filtered boxes."""
        from keras import backend as K

        box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
        boxes = self.yolo_boxes_to_corners(box_xy, box_wh)
        boxes, scores, classes = self.yolo_filter_boxes(
            boxes, box_confidence, box_class_probs, threshold=score_threshold)

        # Scale boxes back to original image shape.
        height = image_shape[0]
        width = image_shape[1]
        image_dims = K.stack([height, width, height, width])
        image_dims = K.reshape(image_dims, [1, 4])
        boxes = boxes * image_dims

        # TODO: Something must be done about this ugly hack!
        max_boxes_tensor = K.variable(max_boxes, dtype='int32')
        K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
        nms_index = tf.image.non_max_suppression(
            boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
        boxes = K.gather(boxes, nms_index)
        scores = K.gather(scores, nms_index)
        classes = K.gather(classes, nms_index)
        return boxes, scores, classes





    ###########################################################
    #####         Iterative Prediction Execution          #####
    ###########################################################
    def yolo_predict(self, image):
        verbose=False

        if self.is_fixed_size:  # TODO: When resizing we can use minibatch input.
            new_image_size = tuple(reversed(self.model_image_size))
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))

        if image.size != new_image_size:
            if verbose: print("resizing input image from {} to {}".format(image.size,new_image_size))
            resized_image = image.resize(new_image_size, Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.



        ### Generate output tensor targets for filtered bounding boxes.
        # TODO: Wrap these backend operations with Keras layers.
        yolo_outputs = self.yolo_head(self.yolo_model.output, anchors, len(class_names))
        input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = self.yolo_eval(
            yolo_outputs,
            input_image_shape,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold)


        feed_dict={self.yolo_model.input: image_data,
                    input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0}

        out_boxes, out_scores, out_classes = self.sess.run(
            [boxes, scores, classes],
            feed_dict=feed_dict)


        return out_boxes,out_scores,out_classes

    def annotate(self, image,boxes,scores,classes):
        """ Add annotation of detected objects to an image """

        try:
            font = ImageFont.truetype(font='/usr/share/fonts/truetype/lato/Lato-Medium.ttf',
                                      size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        except OSError:
            font = ImageFont.truetype(font='/Library/Fonts/Arial.ttf',
                                      size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        thickness = (image.size[0] + image.size[1]) // 300

        NoPers = 0
        self.NoPersons = 0
        for i, c in reversed(list(enumerate(classes))):
            c = int(c)
            predicted_class = class_names[int(c)]
            if predicted_class == "person":
                NoPers+=1
            box = boxes[i]
            score = scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            # lol - michael
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        self.NoPersons = NoPers
        return image

    def process(self, frame):
        """
        Main Callling function
        """
        _boxes,_scores,_classes = self.yolo_predict(frame)
        annotated = self.annotate(frame,_boxes,_scores,_classes)
        return annotated


if __name__ == "__main__":
    obj = Yolo_detector()
    url = 'http://192.168.1.4/uapi-cgi/viewer/snapshot.fcgi?_=1525375407561' + str(random.randint(0,1000))
    response = requests.get(url)
    data = response.content
    nparr = np.fromstring(data, np.uint8)
    image_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_array = scipy.misc.toimage(Image.fromarray(image_arr))

    annotated = obj.process(image_array)
    annotated.save('test.png')







