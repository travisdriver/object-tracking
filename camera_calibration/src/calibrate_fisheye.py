import cv2
assert cv2.__version__[0] == '3' # fisheye module requires opencv3
import numpy as np
import os
import glob

CHECKERBOARD = (7,9) # size of checkerboard used in cal images
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in space
imgpoints = [] # 2d points in image plane.

# main loop
images = glob.glob('calib_imgs/*.jpg')
count = 0
for fname in images:
    img = cv2.imread(fname)
    print(fname)

    # set img shape if None
    if _img_shape == None:
        _img_shape = img.shape[:2]
    # else assert img shape
    else:
        assert _img_shape == img.shape[:2] # all images must share same size

    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

    # if found add object points and image points is successful
    print(ret)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)
        count = count + 1

    # if set number found, break
    if count >= 10:
        break

# calculate camera and distortion matrices
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], K, D, rvecs, tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

# print out results in format for use in undistort.py
print("Used " + str(N_OK) + " images for calibration")
print("DIM = " + str(_img_shape[::-1]))
print("K   = np.array(" + str(K.tolist()) + ")")
print("D   = np.array(" + str(D.tolist()) + ")")
