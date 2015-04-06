#include <iostream>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/*
 * Computes the homography between matches of keypoints in kp1 and kp2.
 * RANSAC is used internally considering a correspondence as an inlier if the
 * error is below the ransac_thr threshold.
 * The function also returns a binary vector of the same size as matches:
 * if inliers[i] is 1, then the matches[i] is considered an inlier.  
 */
cv::Mat compute_homography(const std::vector<cv::DMatch> matches,
	                       const std::vector<cv::KeyPoint> kp1,
	                       const std::vector<cv::KeyPoint> kp2,
	                       const float ransac_thr,
	                       std::vector<uchar>& inliers);

/*
 * Computes the avg. transfer error between matches of keypoints in kp1 and kp2.
 * The error is the Euclidian distance between H*p1 (the transferred point in the
 * first image) and p2 (the same point in the second image).
 * The function also returns a vector of the same size as matches
 * with the transfer error for each match.
 */
double compute_transfer_error(const std::vector<cv::DMatch> matches,
	                          const std::vector<cv::KeyPoint> kp1,
	                          const std::vector<cv::KeyPoint> kp2,
	                          const cv::Mat H,
	                          std::vector<double>& errors);
