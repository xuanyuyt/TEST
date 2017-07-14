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
	BoundingBox(double x, double y, double width, double height)
	{
		this->start_x = x;
		this->start_y = y;
		this->width = width;
		this->height = height;
		this->centroid_x = x + width/2.0;
		this->centroid_y = y + height/2.0;
	}
};

Mat_<double> LoadGroundTruthShape(string& filename); // 读取landmarks坐标
BoundingBox CalculateBoundingBox(Mat_<double>& shape); // 获得最小包围框
double computIOU(const BoundingBox& A, const BoundingBox& B); // 计算IOU
void DrawROIImage(Mat &image, BoundingBox bbx, BoundingBox bbxNeg); // 显示样本
#endif