#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <visualization_utils.h>

using namespace std;
using namespace cv;

void draw_keypoints(Mat& img, const vector<KeyPoint> keypoints, const int size, const Scalar color)
{
	for(int i = 0; i < keypoints.size(); i++)
	{
		circle(img, keypoints[i].pt, size, color);
	}
}

void draw_matches(Mat& img, const vector<DMatch> matches, const vector<KeyPoint> kp1, const vector<KeyPoint> kp2, const vector<uchar> inliers, const int offset, const bool inliers_only)
{
	for(int i = 0; i < matches.size(); i++)
	{	
		if(inliers_only)
		{
			if(inliers[i])
			{
				Point2f pt1 = kp1[matches[i].queryIdx].pt;
				Point2f pt2 = kp2[matches[i].trainIdx].pt;
				line(img, pt1, pt2 + Point2f(offset, 0), Scalar(0, 255, 0));
			}
		}
		else{
			Scalar color;
			if(inliers[i])
			{
				color = Scalar(0, 255, 0);	
			}
			else
			{
				color = Scalar(0, 0, 255);
			}
			Point2f pt1 = kp1[matches[i].queryIdx].pt;
			Point2f pt2 = kp2[matches[i].trainIdx].pt;
			line(img, pt1, pt2 + Point2f(offset, 0), Scalar(0, 255, 0));
		}
	}
}

void draw_planar_object(Mat& img, const Mat obj, const Mat H)
{
	vector<Point2f> obj_corners(4), scene_corners(4);
	obj_corners[0] = Point2f(0, 0);
	obj_corners[1] = Point2f(obj.cols, 0);
	obj_corners[2] = Point2f(obj.cols, obj.rows);
	obj_corners[3] = Point2f(0, obj.rows);
	
	//Find corners of the planar object after multiplying by H
	perspectiveTransform(obj_corners, scene_corners, H);

	scene_corners[0] += Point2f(obj.cols, 0);
	scene_corners[1] += Point2f(obj.cols, 0);
	scene_corners[2] += Point2f(obj.cols, 0);
	scene_corners[3] += Point2f(obj.cols, 0);

	circle(img, scene_corners[0], 5, Scalar(255, 0, 0), -1);
	circle(img, scene_corners[1], 5, Scalar(255, 0, 0), -1);
	circle(img, scene_corners[2], 5, Scalar(255, 0, 0), -1);
	circle(img, scene_corners[3], 5, Scalar(255, 0, 0), -1);

	line(img, scene_corners[0], scene_corners[1], Scalar(255, 0, 0));
	line(img, scene_corners[1], scene_corners[2], Scalar(255, 0, 0));
	line(img, scene_corners[2], scene_corners[3], Scalar(255, 0, 0));
	line(img, scene_corners[3], scene_corners[0], Scalar(255, 0, 0));
}
