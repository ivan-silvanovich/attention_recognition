import glob
import os.path
import json

import cv2
import numpy as np


def store_calibration_data(checkerboard, width, height, calibration_photos_path, camera_config_file):
    # Defining the dimensions of checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D, 2D points for each checkerboard image
    objpoints, imgpoints = [], []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    # prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    images = glob.glob(os.path.join(calibration_photos_path, '*.jpg'))

    if not images:
        camera_matrix = np.array(
            [[width, 0, width / 2],
             [0, width, height / 2],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))
    else:
        for image in images:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(gray, checkerboard,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            # If desired number of corner are detected,
            # we refine the pixel coordinates and display
            # them on the images of checker board
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                imgpoints.append(corners2)

                # Draw and display the corners
        #         img = cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
        #
        #     cv2.imshow('img', img)
        #     cv2.waitKey(0)
        #
        # cv2.destroyAllWindows()

        # h, w = img.shape[:2]

        # Performing camera calibration by
        # passing the value of known 3D points (objpoints)
        # and corresponding pixel coordinates of the
        # detected corners (imgpoints)
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],
                                                                            None, None, flags=cv2.CALIB_FIX_K3)

    dump_data = {'camera_matrix': camera_matrix.tolist(), 'distortion_coefficients': dist_coeffs.tolist()}

    with open(camera_config_file, 'w') as file:
        json.dump(dump_data, file)
