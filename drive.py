import argparse
import base64
import json
import util

import numpy as np
import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# # change to grey scale & normalize (0-1)
# def change_to_gray_normalize (image) : # RGB image (not BGR)
#   return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]) / 255.0
#
#
# # crop 30 lines
# def crop_image(dataset) :
#     return dataset[14:34,:,  :].copy()

# def reformat(dataset):
#
#     dataset = util.crop_images_small_grayscale(dataset)
#     dataset= util.change_to_normalize(dataset)
#     dataset = np.reshape(dataset, (-1, 20, 80, 1)).astype(np.float32) # 일단 1채널만
#     return dataset

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


import cv2
@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]

    # The current speed of the car
    speed = float(data["speed"])

    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))

    image_array = np.asarray(image)
    image_array = cv2.cvtColor( image_array, cv2.COLOR_RGB2GRAY ) # change to GrayImage

    image_array= cv2.resize(image_array, (80,40))  # resize
    image_array = util.crop_driving_image(image_array) # crop
    image_array = cv2.equalizeHist(image_array)  # equalize Histogram
    transformed_image_array = util.reformat_driving_image(image_array)


    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))

    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    if( speed < 30) :
        throttle = 0.3
    else :
        throttle = 0.1

    print(steering_angle, throttle, speed)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)