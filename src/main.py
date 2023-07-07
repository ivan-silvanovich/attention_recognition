import cv2
import numpy as np

from models import db_session
from settings import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_MATRIX, DISTORTION_COEFFICIENTS, MODEL_POINTS, USERS_PHOTOS_PATH
from tools.attention import get_inattentive_users, track_inattentive
from tools.face_detector import get_face_detector, find_faces, draw_named_face_frame
from tools.face_landmarks import get_landmark_model, detect_marks, draw_marks
from tools.face_recognition import recognize_face
from tools.users_registration import register_users

face_model = get_face_detector()
landmark_model = get_landmark_model()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

register_users(USERS_PHOTOS_PATH)

user_tracking = {}
db_session = db_session()

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    if ret:
        faces = find_faces(img, face_model)
        faces_data = []

        for face in faces:
            marks = detect_marks(img, landmark_model, face)

            image_points = np.array([
                marks[30],  # Nose tip
                marks[8],  # Chin
                marks[36],  # Left eye left corner
                marks[45],  # Right eye right corner
                marks[48],  # Left Mouth corner
                marks[54]  # Right mouth corner
            ], dtype="double")

            success, rotation_vector, translation_vector = cv2.solvePnP(
                MODEL_POINTS, image_points, CAMERA_MATRIX, DISTORTION_COEFFICIENTS, flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                faces_data.append({
                    'face': face,
                    'rotation_vector': rotation_vector,
                    'translation_vector': translation_vector,
                })

                nose_end_point2D, jacobian = cv2.projectPoints(
                    np.array([(0.0, 0.0, 1000.0)]),
                    rotation_vector,
                    translation_vector,
                    CAMERA_MATRIX,
                    DISTORTION_COEFFICIENTS
                )

                draw_marks(img, image_points, 5, (0, 0, 255))

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                cv2.line(img, p1, p2, (255, 0, 0), 3)

        if len(faces_data) > 2:
            inattentive_list = get_inattentive_users(faces_data)

            for face in inattentive_list:
                x1, y1, x2, y2 = face
                face_image = cv2.cvtColor(img[y1-50:y2+50, x1-50:x2+50], cv2.COLOR_BGR2RGB)
                user = recognize_face(face_image) or 'Unknown'

                draw_named_face_frame(img, face, str(user))

                if user == 'Unknown':
                    continue
                else:
                    track_inattentive(user_tracking, user)

        cv2.imshow('Attention tracking', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

users = user_tracking.keys()
for user in users:
    track_inattentive(user_tracking, user, 0)

cv2.destroyAllWindows()
cap.release()
