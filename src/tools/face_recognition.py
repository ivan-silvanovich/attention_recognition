import pickle

import dlib
from scipy.spatial import distance

from models import db_session, User
from settings import DLIB_PREDICTOR_PATH, DLIB_FACE_RECOGNITION_PATH, ACCEPTABLE_FACE_DISTANCE

db_session = db_session()
sp = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
facerec = dlib.face_recognition_model_v1(DLIB_FACE_RECOGNITION_PATH)
detector = dlib.get_frontal_face_detector()


def compute_face_descriptor(img):
    # win = dlib.image_window()
    # win.set_image(img)

    dets = detector(img)
    shape = None
    for k, d in enumerate(dets):
        shape = sp(img, d)
    #     win.clear_overlay()
    #     win.add_overlay(d)
    #     win.add_overlay(shape)
    #
    # win.wait_for_keypress('q')

    return facerec.compute_face_descriptor(img, shape) if shape else None


def recognize_face(image):
    users = db_session.query(User).all()
    image_descriptor = compute_face_descriptor(image)

    if image_descriptor is None:
        return None

    for user in users:
        user_descriptor = pickle.loads(user.descriptor)
        d = distance.euclidean(user_descriptor, image_descriptor)

        if d < ACCEPTABLE_FACE_DISTANCE:
            return user
