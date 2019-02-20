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
//#include <opencv2/types.hpp>

using namespace cv;
//using namespace cv::xfeatures2d;
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
    //FlannBasedMatcher matcher;

    // initialize reference image
    Mat ref_img;
    ref_img = imread("ida_temp.jpg");
    imshow("edges", ref_img);

    // calculate keypoints and descriptors for reference image
    vector<KeyPoint> ref_kp;
    Mat ref_desc;
    detector->detectAndCompute(ref_img, Mat(), ref_kp, ref_desc);

    // corners of reference image
    vector<Point2f> ref_corners(4);
    ref_corners[0] = Point(0,0);
    ref_corners[1] = Point( ref_img.cols, 0 );
    ref_corners[2] = Point( ref_img.cols, ref_img.rows );
    ref_corners[3] = Point( 0, ref_img.rows );

    // coners of reference image in 3D
    vector<Point3f> ref_corners_3d(4);
    ref_corners_3d[0] = Point3f(-ref_img.cols/2.,-ref_img.rows/2., 0.);
    ref_corners_3d[1] = Point3f(-ref_img.cols/2., ref_img.rows/2., 0.);
    ref_corners_3d[2] = Point3f( ref_img.cols/2., ref_img.rows/2., 0.);
    ref_corners_3d[3] = Point3f( ref_img.cols/2.,-ref_img.rows/2., 0.);

    // center of reference image
    vector<Point2f> ref_center(1);
    ref_center[0] = Point(ref_img.cols/2., ref_img.rows/2.);

    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    for(;;)
    {
        Mat frame, H;
        cap >> frame; // get a new frame from camera

        vector<KeyPoint> kp;
        Mat desc;
        detector->detectAndCompute(frame, Mat(), kp, desc);

        vector<vector<DMatch>> matches;
        if (!ref_desc.empty() && !desc.empty())
            matcher->knnMatch(ref_desc, desc, matches, 2);

        vector<Point2f> ref_matched, frm_matched;
        for(unsigned i = 0; i < matches.size(); i++)
        {
            // perform ratio test described in Lowe's paper and store good matches
            if(matches[i][0].distance < NN_MATCH_RATIO * matches[i][1].distance)
            {
                ref_matched.push_back( ref_kp[matches[i][0].queryIdx].pt );
                frm_matched.push_back(     kp[matches[i][0].trainIdx].pt );
            }
        }

        cout << ref_matched.size() << endl;
        if (ref_matched.size() > 50)
        {
            // calculate homography transform
            H = findHomography(ref_matched, frm_matched, RANSAC, RANSAC_THRESH);
            if (H.empty()) { continue; }

            // perform perspective transform
            vector<Point2f> frm_corners, frm_center;
            framePerspectiveTransform(ref_corners, frm_corners, ref_center,
                frm_center, H);

            // draw bounding box
            drawMyBoundingBox(frame, ref_img, frm_corners);

            // solve PnP using iterative LMA and draw axes
            Mat rvec(3,1,cv::DataType<double>::type);
            Mat tvec(3,1,cv::DataType<double>::type);
            solvePnP(ref_corners_3d, frm_corners, CAMERA_MAT, DIST_COEFFS, rvec, tvec, 1, 1);
            cout << "rvec: " << rvec*180./M_PI << endl;
            cout << "tvec: " << tvec << endl;
            drawMyAxes(frame, rvec, tvec);

        }

        imshow("edges", frame);
        if(waitKey(30) >= 0) break;
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
