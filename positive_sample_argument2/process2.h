#ifndef PROCESS2_H
#define PROCESS2_H

#include <iostream>
#include <fstream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;

extern string dataPath; // ������ȡ·��
extern string savePath; // ��������·��
extern int landmarks_num;
extern Scalar value; // �߽�padding
extern int extRows; // ��ֱpadding��С
extern int extCols; // ˮƽpadding��С
extern int limitBoundingBoxWidth; // Ŀ����޶����
extern int limitBoundingBoxHeight; // Ŀ����޶��߶�

extern double rateX; // Ŀ���X�������λ��
extern double rateY; // Ŀ���Y�������λ��

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
BoundingBox CalculateBoundingBox2(Mat_<double>& shape); // ��÷Ŵ���Ŀ���
void PaddingImage(Mat &image, BoundingBox& bbx); // �߽���չ
void ZoomImage(Mat &src); // ��������
void DrawPredictedImage(Mat &image, cv::Mat_<double>& shape, BoundingBox bbx);

/***********************************************************************
* ���ȣ�ȫͼ����������ӻ��ȥ�̶�����ֵ����ΧΪͼ�����ؾ�ֵ�ġ�15%֮�䣻
***********************************************************************/
void adjustBright(Mat& srcImage, Mat& dstImage,
	const double& alpha, const double& beta);

void computeMean(Mat& srcImage, Scalar& mean);


/***********************************************************************
* �ֱ��ʣ�ͼ����������²�������ͨ��˫���Բ�ֵ�ķ����Ŵ�ԭͼ��С��
* �²���ϵ��ȡֵ��Χ[0.4,1.0]
***********************************************************************/
void adjustResolution(Mat& srcImage, Mat& dstImage, const double& ratio);
void recoverSize(Mat& srcImage, Mat& dstImage, const double& ratio);
void average(const Mat &img, Point_<int> a, Point_<int> b, Vec3b &p);

#endif