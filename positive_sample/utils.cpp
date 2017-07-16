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

// 将 ground truth shapes 的最小包围矩形放大作为目标框
BoundingBox CalculateBoundingBox2(Mat_<double>& shape){
	BoundingBox bbx;
	double left_x = 10000;
	double right_x = 0;
	double top_y = 10000;
	double bottom_y = 0;
	double tmpX, tmpY, tmpH, tmpW;

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
	tmpX = left_x;
	tmpY = top_y;
	tmpH = bottom_y - top_y;
	tmpW = right_x - left_x;

	bbx.start_x = tmpX - tmpW * 0.2;
	bbx.width = tmpW * 1.4;
	bbx.start_y = top_y - (bbx.width - tmpH) / 2;
	bbx.height = tmpW * 1.4;
	bbx.centroid_x = bbx.start_x + bbx.width / 2.0;
	bbx.centroid_y = bbx.start_y + bbx.height / 2.0;
	return bbx;
}

// 显示样本
void DrawPredictedImage(Mat &image, cv::Mat_<double>& shape, BoundingBox bbx, BoundingBox bbx2){
	for (int i = 0; i < shape.rows; i++){
		circle(image, Point2d(shape(i, 0), shape(i, 1)), 2, Scalar(0, 0, 0), 5, 8, 0);
		circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 1, Scalar(255, 255, 255), 3, 8, 0);
	}
	cv::rectangle(image, cv::Rect(bbx.start_x, bbx.start_y, bbx.width, bbx.height),
		cv::Scalar(0, 0, 255), 2);
	cv::rectangle(image, cv::Rect(bbx2.start_x, bbx2.start_y, bbx2.width, bbx2.height),
		cv::Scalar(0, 255, 0), 2);
	imshow("show image", image);
}
void DrawROIImage(Mat &image)
{
	imshow("show ROI", image);
}

// 边界拓展
void PaddingImage(Mat &image, BoundingBox& bbx)
{
	copyMakeBorder(image, image, extRows, extRows, extCols, extCols, BORDER_CONSTANT, value);//四周拓展边界
	bbx.start_x = bbx.start_x + extCols;
	bbx.start_y = bbx.start_y + extRows;
	bbx.centroid_x = bbx.centroid_x + extCols;
	bbx.centroid_y = bbx.centroid_y + extRows;
}

// 缩放整个图片，将目标框固定为96*96大小
void ZoomImage(Mat &src, Mat &dst, BoundingBox& bbx, Mat_<double>& shape)
{
	double fx = limitBoundingBoxWidth / bbx.width;
	double fy = limitBoundingBoxHeight / bbx.height;
	resize(src, dst, Size(96, 96), 0, 0, INTER_LINEAR); // 双线性插值缩放
	for (int i = 0; i<landmarks_num; i++){
		shape(i, 0) = shape(i, 0) * fx;
		shape(i, 1) = shape(i, 1) * fy;
	}
	bbx.start_x = bbx.start_x * fx;
	bbx.start_y = bbx.start_y * fy;
	bbx.width = limitBoundingBoxWidth;
	bbx.height = limitBoundingBoxHeight;
	bbx.centroid_x = bbx.start_x + bbx.width / 2.0;
	bbx.centroid_y = bbx.start_y + bbx.height / 2.0;
}
