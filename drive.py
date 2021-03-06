import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import matplotlib.pyplot as plt


import sys

import preprocess

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    """ Pulls telemetry data from car-driving game, processes it, and generates
    steering angle for feeding back to the game.
    """

    """Get telemetry data"""
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # BCJ note: The only image we get from this script is the center image. So the model
    # should only be trained on the center image.
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    transformed_image_array = image_array[None, :, :, :]
    single_im = transformed_image_array[0,:,:,:]

    """Preprocess the image before using it to get steering angle"""
    start_row = 60 # for trimming images
    stop_row = 140 # for trimming images
    cen_proc = preprocess.trim_image(single_im, start_row, stop_row)
    cen_proc = preprocess.grayscale(cen_proc)
    cen_proc = preprocess.normalize_image(cen_proc)
    cen_proc = cen_proc.astype('float32')  # Ensure data is float32 (Tensorflow expects this)
    shape_cen_proc = cen_proc.shape
    final_im = np.zeros((1, shape_cen_proc[0], shape_cen_proc[1], 1))
    final_im[0,:,:,0] = cen_proc
    transformed_image_array = final_im
    #print('Final image shape = ', transformed_image_array.shape)
    #print('Final image min, max = ', np.min(transformed_image_array), np.max(transformed_image_array))
    #plt.figure
    #plt.imshow(transformed_image_array[0,:,:,0], cmap='gray')
    #plt.show()

    """Get steering angle and feed back to driving game"""
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.1
    print(steering_angle, throttle)
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