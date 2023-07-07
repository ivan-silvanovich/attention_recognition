import cv2
import numpy as np

import settings


def get_face_detector(model_file=None, config_file=None, quantized=False):
    if quantized:
        return cv2.dnn.readNetFromTensorflow(
            model_file or settings.TENSORFLOW_MODEL_FILE,
            config_file or settings.TENSORFLOW_CONFIG_FILE
        )
    else:
        return cv2.dnn.readNetFromCaffe(
            config_file or settings.CAFFE_CONFIG_FILE,
            model_file or settings.CAFFE_MODEL_FILE
        )


def find_faces(img, model):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    res = model.forward()
    faces = []
    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]
        if confidence > 0.5:
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype('int')
            if x > w and y > h and x1 > w and y1 > h:
                print(f'look: {[x, y, x1, y1]}')
                continue
            faces.append([x, y, x1, y1])
    return faces


def draw_face_frame(img, face_frame, color=(0, 255, 0), width=3):
    x, y, x1, y1 = face_frame
    cv2.rectangle(img, (x, y), (x1, y1), color, width)


def draw_named_face_frame(img, face_frame, name, color=(0, 255, 0), width=3):
    x1, y1, x2, y2 = face_frame
    draw_face_frame(img, face_frame, color, width)
    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
