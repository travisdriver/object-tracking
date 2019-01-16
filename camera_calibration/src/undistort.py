import sys
import numpy as np
import cv2

# camera resolution
DIM = (2448, 2048)

# camera matrix
K   = np.array([[1166.0997466247031, 0.0, 1251.2230433798386], [0.0, 1168.3935439304491, 993.4542506342376], [0.0, 0.0, 1.0]])

# distortion matrix
D   = np.array([[-0.0637447053999411], [-0.20366897141528187], [0.5539112758565363], [-0.5460778039790766]])

# undistort the passed image using the instrinsic camera parameters above
def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    print(undistorted_img.shape)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# main
if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)
