#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

/*
 * Draws in img a circle at each keypoint position of keypoints with size sz and color c.
 */
void draw_keypoints(cv::Mat& img, const std::vector<cv::KeyPoint> keypoints, const int sz, const cv::Scalar c);

/*
 * Draws in img a line for each keypoint correspondence in matches and kp1/kp2.
 * It is assumed that img is a side-by-side image containing the planar object to be
 * recognized and an image of the scene.
 * For match[i], the line goes from kp1[i] to kp2[i] + offset, where offset is the width
 * of the planar object image.
 * If inliers_only is set, only the inlier correspondences in inliers are drawn.
 */
void draw_matches(cv::Mat& img, const std::vector<cv::DMatch> matches, const std::vector<cv::KeyPoint> kp1, const std::vector<cv::KeyPoint> kp2, const std::vector<uchar> inliers, const int offset, const bool inliers_only=false);

/*
 * Draws the four corners of the planar object obj in the scene image img
 * as the perspective transformation given by the homography matrix H.
 */
void draw_planar_object(cv::Mat& img, const cv::Mat obj, const cv::Mat H);
