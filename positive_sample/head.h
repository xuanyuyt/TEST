#ifndef HEAD_H
#define HEAD_H
#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"

using namespace std;
using namespace cv;

extern string dataPath; // 样本读取路径
extern string savePath; // 样本保存路径
extern int landmarks_num; // 特征点数量
extern Scalar value; // 边界padding
extern int extRows; // 垂直padding大小
extern int extCols; // 水平padding大小
extern int limitBoundingBoxWidth; // 目标框限定宽度
extern int limitBoundingBoxHeight; // 目标框限定高度

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

Mat_<double> LoadGroundTruthShape(string& filename); // 读取landmarks坐标
BoundingBox CalculateBoundingBox(Mat_<double>& shape); // 获得最小包围框
BoundingBox CalculateBoundingBox2(Mat_<double>& shape); // 获得放大后的目标框
void DrawPredictedImage(Mat &image, cv::Mat_<double>& shape, BoundingBox bbx, BoundingBox bbx2);
void DrawROIImage(Mat &image);
void PaddingImage(Mat &image, BoundingBox& bbx); // 边界拓展
void ZoomImage(Mat &src, Mat &dst, BoundingBox& bbx, Mat_<double>& ground_truth_shape); // 缩放样本

#endif