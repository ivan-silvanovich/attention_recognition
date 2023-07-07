import json
import os

import numpy as np

from tools import camera_calibration

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CALIBRATION_CHECKERBOARD = 6, 9
ATTENTION_GAP = 1
ACCEPTABLE_FACE_DISTANCE = 0.6
INATTENTION_DETECTION_TIME = 3

# 3D model points.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),  # nose tip
    (0.0, -330.0, -65.0),  # chin
    (-225.0, 170.0, -135.0),  # left eye left corner
    (225.0, 170.0, -135.0),  # right eye right corner
    (-150.0, -150.0, -125.0),  # left mouth corner
    (150.0, -150.0, -125.0)  # right mouth corner
])

DATA_PATH = 'D:\Education\Programming\Python\Projects\\attention\data'
MODELS_PATH = os.path.join(DATA_PATH, 'models')
CALIBRATION_PATH = os.path.join(DATA_PATH, 'calibration')
USERS_PATH = os.path.join(DATA_PATH, 'users')

TENSORFLOW_MODEL_FILE = os.path.join(MODELS_PATH, 'opencv_face_detector_uint8.pb')
TENSORFLOW_CONFIG_FILE = os.path.join(MODELS_PATH, 'opencv_face_detector.pbtxt')
TENSORFLOW_LANDMARK_MODEL = os.path.join(MODELS_PATH, 'pose_model')
CAFFE_MODEL_FILE = os.path.join(MODELS_PATH, 'res10_300x300_ssd_iter_140000.caffemodel')
CAFFE_CONFIG_FILE = os.path.join(MODELS_PATH, 'deploy.prototxt')
DLIB_PREDICTOR_PATH = os.path.join(MODELS_PATH, 'shape_predictor_68_face_landmarks.dat')
DLIB_FACE_RECOGNITION_PATH = os.path.join(MODELS_PATH, 'dlib_face_recognition_resnet_model_v1.dat')

CALIBRATION_PHOTOS_PATH = os.path.join(CALIBRATION_PATH, 'photos')
CAMERA_CONFIG_FILE = os.path.join(CALIBRATION_PATH, 'config.json')

USERS_PHOTOS_PATH = os.path.join(USERS_PATH, 'photos')

SCREEN_RESOLUTIONS_URL = 'https://en.wikipedia.org/wiki/List_of_common_resolutions'

if not os.path.exists(CAMERA_CONFIG_FILE) or __name__ == '__main__':
    camera_calibration.store_calibration_data(
        CALIBRATION_CHECKERBOARD, CAMERA_WIDTH, CAMERA_HEIGHT, CALIBRATION_PHOTOS_PATH, CAMERA_CONFIG_FILE
    )

with open(CAMERA_CONFIG_FILE, 'r') as file:
    camera_config_data = json.load(file)

CAMERA_MATRIX = np.array(camera_config_data['camera_matrix'])
DISTORTION_COEFFICIENTS = np.array(camera_config_data['distortion_coefficients'])
