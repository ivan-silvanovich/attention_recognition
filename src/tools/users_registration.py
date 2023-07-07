import os
import pickle

from skimage import io
from sqlalchemy import select

from tools.face_recognition import compute_face_descriptor
from models import User, db_session

db_session = db_session()


def register_users(users_photos_path):
    images = os.listdir(users_photos_path)
    for image in images:
        image_name = image.split('.')[0]
        image_path = os.path.join(users_photos_path, image)

        name, surname, *_ = image_name.split('_') + ['']

        try:
            image_object = io.imread(image_path)
        except ValueError:
            print(f'{image} is not an image -- Skip')
            continue

        if not db_session.execute(select(User).where(User.picture == image_path)).first():
            db_session.add(
                User(
                    name=name, surname=surname, picture=image_path,
                    descriptor=pickle.dumps(compute_face_descriptor(image_object))
                )
            )
            db_session.commit()
        else:
            print(f'{name} {surname} already exists among users -- Skip')
            continue

        print(f'{image} -- Success', end='\n\n')
