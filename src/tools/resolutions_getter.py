import cv2
import pandas as pd

from settings import SCREEN_RESOLUTIONS_URL

table = pd.read_html(SCREEN_RESOLUTIONS_URL)[0]
table.columns = table.columns.droplevel()
cap = cv2.VideoCapture(0)
resolutions = {}
for index, row in table[["W", "H"]][::-1].iterrows():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, row["W"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, row["H"])
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    resolutions[str(width) + "x" + str(height)] = "OK"
    print(index, row["W"], row["H"])
print(resolutions)
