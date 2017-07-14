#ifndef PROCESS_H
#define PROCESS_H

#include <iostream>
#include <fstream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;

extern string dataPath; // 样本读取路径
extern string savePath; // 样本保存路径
extern int landmarks_num;
extern Scalar value; // 边界padding
extern int extRows; // 垂直padding大小
extern int extCols; // 水平padding大小
extern int limitBoundingBoxWidth; // 目标框限定宽度
extern int limitBoundingBoxHeight; // 目标框限定高度

/*****************************************************************
* 图像旋转：按标注中心随机旋转一个角度，角度取值范围：[-10°,+10°]
*****************************************************************/
void imageRotation(const Mat& srcImage, Mat& dstImage, Mat_<double>& shape, const float& angle);


// 读取 ground truth shapes
Mat_<double> LoadGroundTruthShape(string& filename);

// 围绕坐标原点旋转
Point2d rotationPoint(Point2d srcPoint, const double cosAngle, const double sinAngle);

// 显示样本
void DrawPredictedImage(Mat &image, cv::Mat_<double>& shape);

// 保存landmarks
void saveLandmarks(const Mat_<double>& shape, const string& file);

/*****************************************************************
* 平移：随机平移标注坐标最小包围盒的[-5 % , 5 % ]，x、y分别随机
* 随后截取正样本
*****************************************************************/
class BoundingBox{
public:
	double start_x;
	double start_y;
	double width;
	double height;
	double centroid_x;
	double centroid_y;
	BoundingBox(){
		start_x = 0;
		start_y = 0;
		width = 0;
		height = 0;
		centroid_x = 0;
		centroid_y = 0;
	};
};
void shiftSample(Mat& srcImage, Mat& ROI, Mat_<double>& shape, const int rateX, const int rateY);

// 随机平移后获得放大后的目标框
BoundingBox CalculateBoundingBox3(Mat_<double>& shape, const double rateX, const double rateY);

// 边界拓展
void PaddingImage(Mat &image, BoundingBox& bbx);

// 缩放整个图片，将目标框固定为96*96大小
void ZoomImage(Mat &src, Mat &dst, BoundingBox& bbx, Mat_<double>& ground_truth_shape); // 缩放样本

#endif