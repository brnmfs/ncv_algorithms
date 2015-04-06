#include <iostream>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>

#include <geometry_utils.h>
#include <visualization_utils.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv){

	Mat scene, scene_bw, cover, cover_bw, result;
	
	if(argc != 5){
		fprintf(stderr, "Usage: %s <cover_img> <scene_img> <surf_threshold> <ransac_threshold> \n", argv[0]);
		exit(-1);
	}
	
	//Load images
	cover = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
	//cvtColor(cover, cover_bw, CV_BGR2GRAY);
	cover_bw = cover;
	scene = imread(argv[2], CV_LOAD_IMAGE_UNCHANGED);
	scene_bw = scene;
	//cvtColor(scene, scene_bw, CV_BGR2GRAY);

	//Recognition parameters
	float surf_thr = atof(argv[3]); //default: 800
	float h_thr = atof(argv[4]); //default: 5.0
	vector<KeyPoint> kp_scene, kp_cover;
	Mat desc_scene, desc_cover;
	SurfFeatureDetector detector(surf_thr);
	SurfDescriptorExtractor extractor;
	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	vector<uchar> inliers;
	Mat H;
	
	//Detect and extract keypoints and descriptors
	detector.detect(scene_bw, kp_scene);
	extractor.compute(scene_bw, kp_scene, desc_scene);
	detector.detect(cover_bw, kp_cover);
	extractor.compute(cover_bw, kp_cover, desc_cover);
	
	//Match keypoints
	matcher.match(desc_cover, desc_scene, matches);
	printf("Number of matches: %i\n", (int) matches.size());
	
	//Compute homography between image of the CD cover and image of the scene
	H = compute_homography(matches, kp_cover, kp_scene, h_thr, inliers);
	
	//Count number of inlier matches
	int num_inliers = 0;
	for(int i = 0; i < inliers.size(); i++){
		if(inliers[i])
			num_inliers++;
	}
	printf("Inlier matches: %i (%f%% of %i)\n", num_inliers, (float)num_inliers/matches.size()*100, (int)matches.size());

	//Compute sum of errors:  Euc. distance between Hp1 (transformed point) and p2 (point in the second image)
	vector<double> errors(matches.size());
	double se2 = compute_transfer_error(matches, kp_cover, kp_scene, H, errors);
	printf("Avg. transfer error: %f\n", se2);

	//Draw cover keypoints
	draw_keypoints(cover, kp_cover, 2, Scalar(255, 0, 0));
	
	//Draw scene keypoints
	draw_keypoints(scene, kp_scene, 2, Scalar(255, 0, 0));
	
	//Build resulting image (cover + scene)
	result = Mat::zeros(max(cover.rows, scene.rows), cover.cols + scene.cols, cover.type());
	Mat roi = result(Rect(0, 0, cover.cols, cover.rows));
	cover.convertTo(roi, roi.type(), 1, 0);
	roi = result(Rect(cover.cols, 0, scene.cols, scene.rows));
	scene.convertTo(roi, roi.type(), 1, 0);

	//Draw matches
	draw_matches(result, matches, kp_cover, kp_scene, inliers, cover.cols, true);	

	//Draw CD cover
	draw_planar_object(result, cover, H);

	imshow("CD Cover Recognition", result);
	int key = waitKey(0);

	return 0;
}