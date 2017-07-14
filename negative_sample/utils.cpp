#include "head.h"
using namespace std;
using namespace cv;

// 读取 ground truth shapes
Mat_<double> LoadGroundTruthShape(string& filename){
	Mat_<double> shape(landmarks_num, 2);
	ifstream fin;
	string temp;

	fin.open(filename);
	getline(fin, temp);
	getline(fin, temp);
	getline(fin, temp);
	for (int i = 0; i<landmarks_num; i++){
		fin >> shape(i, 0) >> shape(i, 1);
	}
	fin.close();
	return shape;
}

// 将 ground truth shapes 的最小包围矩形作为目标框
BoundingBox CalculateBoundingBox(Mat_<double>& shape){
	BoundingBox bbx;
	double left_x = 10000;
	double right_x = 0;
	double top_y = 10000;
	double bottom_y = 0;
	for (int i = 0; i < shape.rows; i++){
		if (shape(i, 0) < left_x)
			left_x = shape(i, 0);
		if (shape(i, 0) > right_x)
			right_x = shape(i, 0);
		if (shape(i, 1) < top_y)
			top_y = shape(i, 1);
		if (shape(i, 1) > bottom_y)
			bottom_y = shape(i, 1);
	}
	bbx.start_x = left_x;
	bbx.start_y = top_y;
	bbx.height = bottom_y - top_y;
	bbx.width = right_x - left_x;
	bbx.centroid_x = bbx.start_x + bbx.width / 2.0;
	bbx.centroid_y = bbx.start_y + bbx.height / 2.0;
	return bbx;
}

// 计算IOU
double computIOU(const BoundingBox& A, const BoundingBox& B)
{
	double W = min(A.start_x + A.width, B.start_x + B.width) - max(A.start_x, B.start_x);
	double	H = min(A.start_y + A.height, B.start_y + B.height) - max(A.start_y, B.start_y);
	if (W <= 0 || H <= 0)
		return 0;
	double SA = A.width * A.height;
	double SB = B.width * B.height;
	double cross = W * H;
	return cross / (SA + SB - cross);
}


// 显示样本
void DrawROIImage(Mat &image, BoundingBox bbx, BoundingBox bbxNeg){
	
	cv::rectangle(image, cv::Rect(bbx.start_x, bbx.start_y, bbx.width, bbx.height),
		cv::Scalar(0, 0, 255), 2);
	cv::rectangle(image, cv::Rect(bbxNeg.start_x, bbxNeg.start_y, bbxNeg.width, bbxNeg.height),
		cv::Scalar(0, 255, 0), 2);
	imshow("show IOU", image);
}