#ifndef PROCESS2_H
#define PROCESS2_H

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

extern double rateX; // 目标框X方向随机位移
extern double rateY; // 目标框Y方向随机位移

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
BoundingBox CalculateBoundingBox2(Mat_<double>& shape); // 获得放大后的目标框
void PaddingImage(Mat &image, BoundingBox& bbx); // 边界拓展
void ZoomImage(Mat &src); // 缩放样本
void DrawPredictedImage(Mat &image, cv::Mat_<double>& shape, BoundingBox bbx);

/***********************************************************************
* 亮度：全图随机整体增加或减去固定像素值，范围为图像像素均值的±15%之间；
***********************************************************************/
void adjustBright(Mat& srcImage, Mat& dstImage,
	const double& alpha, const double& beta);

void computeMean(Mat& srcImage, Scalar& mean);


/***********************************************************************
* 分辨率：图像按随机比例下采样，再通过双线性插值的方法放大到原图大小，
* 下采样系数取值范围[0.4,1.0]
***********************************************************************/
void adjustResolution(Mat& srcImage, Mat& dstImage, const double& ratio);
void recoverSize(Mat& srcImage, Mat& dstImage, const double& ratio);
void average(const Mat &img, Point_<int> a, Point_<int> b, Vec3b &p);

#endif