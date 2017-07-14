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

extern string dataPath; // ������ȡ·��
extern string savePath; // ��������·��
extern int landmarks_num; // ����������
extern Scalar value; // �߽�padding
extern int extRows; // ��ֱpadding��С
extern int extCols; // ˮƽpadding��С
extern int limitBoundingBoxWidth; // Ŀ����޶����
extern int limitBoundingBoxHeight; // Ŀ����޶��߶�

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

Mat_<double> LoadGroundTruthShape(string& filename); // ��ȡlandmarks����
BoundingBox CalculateBoundingBox(Mat_<double>& shape); // �����С��Χ��
BoundingBox CalculateBoundingBox2(Mat_<double>& shape); // ��÷Ŵ���Ŀ���
void DrawPredictedImage(Mat &image, cv::Mat_<double>& shape, BoundingBox bbx, BoundingBox bbx2);
void DrawROIImage(Mat &image);
void PaddingImage(Mat &image, BoundingBox& bbx); // �߽���չ
void ZoomImage(Mat &src, Mat &dst, BoundingBox& bbx, Mat_<double>& ground_truth_shape); // ��������

#endif