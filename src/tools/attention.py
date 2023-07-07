from time import time

import numpy as np

from models import Event, db_session
from settings import ATTENTION_GAP, INATTENTION_DETECTION_TIME

db_session = db_session()


def get_inattentive_users(faces_data):
    average_vector = sum(face_data['rotation_vector'] for face_data in faces_data) / len(faces_data)
    inattentive_list = []

    for face_data in faces_data:
        if (np.abs(face_data['rotation_vector'] - average_vector) > ATTENTION_GAP).any():
            inattentive_list.append(face_data['face'])

    return inattentive_list


def track_inattentive(track_list, user, idt=INATTENTION_DETECTION_TIME):
    current_time = time()
    user_data = track_list.setdefault(
        user,
        {'first_detection': current_time, 'last_detection': current_time}
    )

    inattentive_time = user_data['last_detection'] - user_data['first_detection']
    if current_time - user_data['last_detection'] > idt:
        if inattentive_time > INATTENTION_DETECTION_TIME:
            db_session.add(Event(user=user.id, duration=inattentive_time))
            db_session.commit()

        track_list[user]['first_detection'] = track_list[user]['last_detection'] = current_time
    else:
        track_list[user]['last_detection'] = current_time
