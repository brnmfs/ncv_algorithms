#include <iostream>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <geometry_utils.h>

using namespace std;
using namespace cv;

Mat compute_homography(const vector<DMatch> matches, const vector<KeyPoint> kp1, const vector<KeyPoint> kp2, const float ransac_thr, vector<uchar>& inliers)
{
	vector<Point2f> points1, points2;
	for(int i = 0; i < matches.size(); i++){
		int idx1 = matches[i].queryIdx;
		int idx2 = matches[i].trainIdx;
		Point2f pt1 = kp1[idx1].pt;
		Point2f pt2 = kp2[idx2].pt;
		points1.push_back(pt1);
		points2.push_back(pt2);
	}

	Mat H = findHomography(points1, points2, CV_RANSAC, ransac_thr, inliers);

	return H;
}

double compute_transfer_error(const vector<DMatch> matches, const vector<KeyPoint> kp1, const vector<KeyPoint> kp2, const Mat H, vector<double>& errors)
{
	double se = 0;
	for(int i = 0; i < matches.size(); i++){
		int idx1 = matches[i].queryIdx;
		int idx2 = matches[i].trainIdx;

		Mat tp(3, 1, CV_64F);
		tp.at<double>(0, 0) = kp1[idx1].pt.x;
		tp.at<double>(1, 0) = kp1[idx1].pt.y;
		tp.at<double>(2, 0) = 1;
		tp = H*tp;
		double err = sqrt((tp.at<double>(0,0)/tp.at<double>(0,2) - kp2[idx2].pt.x)*(tp.at<double>(0,0)/tp.at<double>(0,2) - kp2[idx2].pt.x) + 
                   (tp.at<double>(1,0)/tp.at<double>(0,2) - kp2[idx2].pt.y)*(tp.at<double>(1,0)/tp.at<double>(0,2) - kp2[idx2].pt.y));
		se += err;
		errors[i] = err;
	}

	return se/matches.size();
}
