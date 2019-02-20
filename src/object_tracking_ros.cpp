#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <math.h>
#include <vector>
#include <iostream>
#include <iomanip>

#include "../include/tracker.h"

using namespace cv;
using namespace std;

const double NN_MATCH_RATIO = 0.75f; // nearest-neighbour matching ratio
const double RANSAC_THRESH  = 2.5f;  // RANSAC inlier threshold
const Mat CAMERA_MAT = (Mat1d(3, 3) << 1158.03, 0.,      540.,
                                         0.,      1158.03, 360.,
                                         0.,      0.,      1.  );
const Mat DIST_COEFFS = (Mat1d(1, 4) << 0., 0., 0., 0.);

// apply homography matrix for perpective transformation
void framePerspectiveTransform(vector<Point2f> ref_corners,
    vector<Point2f> &frm_corners, vector<Point2f> ref_center,
    vector<Point2f> &frm_center, Mat H)
{
    perspectiveTransform(ref_corners, frm_corners, H);
    perspectiveTransform(ref_center, frm_center, H);
}

// drawing box around object
void drawMyBoundingBox(Mat& frm, Mat &ref_img, vector<Point2f> corners)
{
    line( frm, corners[0], corners[1], Scalar( 0, 255, 0), 4 );
    line( frm, corners[1], corners[2], Scalar( 0, 255, 0), 4 );
    line( frm, corners[2], corners[3], Scalar( 0, 255, 0), 4 );
    line( frm, corners[3], corners[0], Scalar( 0, 255, 0), 4 );
}

// draw axis lines on frame to show object orientation relative to reference
void drawMyAxes(Mat &frm, Mat rvec, Mat tvec)
{
    // idk why this makes it look right
    rvec.at<double>(0,0) = rvec.at<double>(0,0)*-1.;
    rvec.at<double>(1,0) = rvec.at<double>(1,0)*-1.;

    // reference axis
    vector<Point3f> axis(4);
    axis[0] = Point3f(0,  0,  0);
    axis[1] = Point3f(100, 0, 0);
    axis[2] = Point3f(0, 100, 0);
    axis[3] = Point3f(0, 0, 100);

    // project axis onto object in frame
    vector<Point2f> projectedPts;
    projectPoints(axis, rvec, tvec, CAMERA_MAT, DIST_COEFFS, projectedPts);

    // draw axis lines
    line( frm, projectedPts[0], projectedPts[1], Scalar( 255, 0, 0), 3 ); // x blue
    line( frm, projectedPts[0], projectedPts[2], Scalar( 0, 255, 0), 3 ); // y green
    line( frm, projectedPts[0], projectedPts[3], Scalar( 0, 0, 255), 3 ); // z red
}

/*
// perform optical flow on environment features
def perform_optical_flow(previous_frame, previous_fp, current_frame):
{

}
*/

//==============================================================================
//                                  MAIN
//==============================================================================

int main(int, char**)
{
    // create detector and orb_matcher
    //Ptr<Feature2D> detector = ORB::create();
    Ptr<Feature2D> detector = cv::xfeatures2d::SIFT::create(500);
    //Ptr<SURF> detector = SURF::create(500);
    //Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

    // initialize reference image
    Mat ref_img;
    ref_img = imread("ida_temp.jpg");
    imshow("edges", ref_img);

    Tracker tracker(detector, matcher);

    tracker.setReferenceImg(ref_img);

    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    for(;;)
    {
        Mat frame, H;
        cap >> frame; // get a new frame from camera

        tracker.calcMatches(frame);

        cout << tracker.getMatches() << endl;
        if (tracker.getMatches() > 50)
        {
            tracker.getRelativePose();

            cout << "rvec: " << tracker.getPoseRVec()*180./M_PI << endl;
            cout << "tvec: " << tracker.getPoseTVec() << endl;

            tracker.drawMyBoundingBox();
            tracker.drawFrameAxes();

        }

        imshow("edges", tracker.getCurrentFrame());
        if(waitKey(30) >= 0) break;
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
