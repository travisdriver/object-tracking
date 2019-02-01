from collections import deque
from imutils.video import VideoStream
from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2
import imutils
import time

# calculate homography matrix between reference image and current frame
def calc_homography(kpref, kpfrm, matches):
    src_pts = np.float32([ kpref[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpfrm[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    return H, _

# apply homography matrix for perpective transformation
def output_perspective_transform(img_ref, M):
    h,w = img_ref.shape
    # get center and corners from the reference image
    corner_pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    center_pts = np.float32([ [w/2,h/2] ]).reshape(-1,1,2)
    corner_pts_3d = np.float32([ [-w/2,-h/2,0],[-w/2,(h-1)/2,0],[(w-1)/2,(h-1)/2,0],[(w-1)/2,-h/2,0] ])

    # calculate perspectiev transform
    corner_camera_coord = cv2.perspectiveTransform(corner_pts,M)
    center_camera_coord = cv2.perspectiveTransform(center_pts,M)
    return corner_camera_coord, center_camera_coord, corner_pts_3d, center_pts

# solving pnp using iterative LMA algorithm
def iterative_solve_pnp(object_points, image_points):
    image_points = image_points.reshape(-1,2)
    retval, rotation, translation = cv2.solvePnP(object_points, image_points, \
        kinect_intrinsic_param, kinect_distortion_param)
    return rotation, translation

# drawing box around object
def draw_bounding_box(corners):
    cv2.polylines(frame, [np.int32(corners)],True,255,3)

# draw lines between the corners and show matches
def draw_box_with_matches(scene_corners):
    cv2.line(img_matches, (int(scene_corners[0,0,0] + objImg.shape[1]), int(scene_corners[0,0,1])),\
        (int(scene_corners[1,0,0] + objImg.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
    cv2.line(img_matches, (int(scene_corners[1,0,0] + objImg.shape[1]), int(scene_corners[1,0,1])),\
        (int(scene_corners[2,0,0] + objImg.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
    cv2.line(img_matches, (int(scene_corners[2,0,0] + objImg.shape[1]), int(scene_corners[2,0,1])),\
        (int(scene_corners[3,0,0] + objImg.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
    cv2.line(img_matches, (int(scene_corners[3,0,0] + objImg.shape[1]), int(scene_corners[3,0,1])),\
        (int(scene_corners[0,0,0] + objImg.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)

# showing object position and orientation value to frame
def put_position_orientation_value_to_frame(translation, rotation):
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame,'position(cm)',(10,30), font, 0.7,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(frame,'x:'+str(np.round(translation[0],2)),(250,30), font, 0.7,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'y:'+str(np.round(translation[1],2)),(400,30), font, 0.7,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'z:'+str(np.round(translation[2],2)),(550,30), font, 0.7,(0,0,255),2,cv2.LINE_AA)

    cv2.putText(frame,'orientation(degree)',(10,60), font, 0.7,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(frame,'x:'+str(np.round(rotation[0],2)),(250,60), font, 0.7,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'y:'+str(np.round(rotation[1],2)),(400,60), font, 0.7,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'z:'+str(np.round(rotation[2],2)),(550,60), font, 0.7,(0,0,255),2,cv2.LINE_AA)

# draw axis lines on frame to show object orientation relative to reference
def draw_axes(rvec, tvec, center, mtx, dist):
    axis = np.float32([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]]).reshape(-1,3)
    rvec[2] = rvec[2]*-1.
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
    cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (255,0,0), 3) # x-axis red
    cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0,255,0), 3) # y-axis green
    cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 3) # z-axis blue

# perform optical flow on environment features
def perform_optical_flow(previous_frame, previous_fp, current_frame):
    current_fp, stat, err = cv2.calcOpticalFlowPyrLK(previous_frame, current_frame,
                            previous_fp, None, **lk_params)
    #previous_good = previous_fp[stat==1]
    #current_good = current_fp[stat==1]
    return previous_fp, current_fp, stat, err



#---------------------------------
#       main
#---------------------------------

# intrinsic camera parameters from calibration
kinect_intrinsic_param = np.array([[1158.03, 0., 540], [0., 1158.03, 360], [0., 0., 1.]])
kinect_distortion_param = np.array([0., 0., 0., 0., 0.])

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15), maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# initialize reference image
objImg = cv2.imread('../tracking_imgs/ids_temp2.jpg',0)

# create detetctor and matcher objects
detector_option = 'SIFT'
surf = cv2.xfeatures2d.SURF_create(500)
sift = cv2.xfeatures2d.SIFT_create()
orb = cv2.ORB_create()

# macthers
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
bf = cv2.BFMatcher()
orb_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)#, crossCheck=True)

# set
detector = sift
matcher = flann

# calculate keypoints and descriptors for reference image
kpo, deso = detector.detectAndCompute(objImg,None)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
        help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
        help="max buffer size")
args = vars(ap.parse_args())

pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab reference to webcam
if not args.get("video", False):
        vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
        vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# grab current frame
frame = vs.read()

# save first frame for optical flow and find features to track
previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
previous_fp = cv2.goodFeaturesToTrack(previous_frame, mask = None, **feature_params).reshape(-1,2)

# main loop
while True:
        # grab next frame
        frame = vs.read()

        # perform optical flow using previous and current frame
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p0, p1, stat, err = perform_optical_flow(previous_frame, previous_fp, current_frame)

        # calc avg change in displacement vector for checking
        #diff_p = np.subtract(p0,p1)
        #sum = 0
        #for i in range(len(diff_p)):
        #    curr = (diff_p[i,0]**2 + diff_p[i,1]**2)**(1/2)
        #    sum += curr
        #avg = sum/len(diff_p)
        #print(avg)

        # handle the frame from VideoCapture or VideoStream
        frame = frame[1] if args.get("video", False) else frame

        # exit if no more frames
        if frame is None:
                break

        # resize the frame
        kpf, desf = detector.detectAndCompute(frame,None)
        matches = matcher.knnMatch(deso,desf,k=2)
        print(len(matches))
        #good_matches = sorted(matches,key=lambda x:x.distance)

        # ratio test described in Lowe's paper
        good_matches = [m for m,n in matches if m.distance < 0.75*n.distance]
        print(len(good_matches))

        # draw matches
        #img_matches = np.empty((max(objImg.shape[0], frame.shape[0]), objImg.shape[1]+frame.shape[1], 3), dtype=np.uint8)
        #cv2.drawMatches(objImg,kpo,frame,kpf,good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # if enough good matches, perform homography
        if len(good_matches) > 100:

            # convert keypoints to 2D list for calcOpticalFlowPyrLK
            #pts = np.float([good_matches[idx].pt for idx in len(good_matches)]).reshape(-1, 2)
            #print(pts)

            # calculate homography
            H, _ = calc_homography(kpo,kpf,good_matches)
            if H is None:
                continue

            # break if could not find valid homography
            #if H == 0:
            #    break

            # perform perspective transform and draw bounding box
            corner_camera_coord, center_camera_coord, corner_pts_3d, center_pts = output_perspective_transform(objImg, H)
            draw_bounding_box(corner_camera_coord)

            # solve pnp using iterative LMA algorithm
            rvec, tvec = iterative_solve_pnp(corner_pts_3d, corner_camera_coord)
            rMat = np.identity(3)
            cv2.Rodrigues(rvec,rMat)
            rMat = np.transpose(rMat)
            rvec2 = rvec
            cv2.Rodrigues(rMat,rvec2)
            rvec = rvec * 180./np.pi
            draw_axes(rvec2, tvec, center_camera_coord, kinect_intrinsic_param, kinect_distortion_param)
            put_position_orientation_value_to_frame(tvec, rvec)

        # show frame
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF

        # if 'q' key is pressed stop the loop
        if key == ord("q"):
                break

        # update optical flow arrays
        p0 = p1
        previous_frame = current_frame

# if no video file stop the camera video stream
if not args.get("video", False):
        vs.stop()
# else release the camera
else:
        vs.release()

# close all windows
cv2.destroyAllWindows()
