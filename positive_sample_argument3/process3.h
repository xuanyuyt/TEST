#ifndef PROCESS_H
#define PROCESS_H

#include <iostream>
#include <fstream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <math.h>
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
extern double rateX;
extern double rateY;

/*****************************************************************
* ͼ����ת������ע���������תһ���Ƕȣ��Ƕ�ȡֵ��Χ��[-10��,+10��]
*****************************************************************/
void imageRotation(const Mat& srcImage, Mat& dstImage, Mat_<double>& shape, const float& angle);


// ��ȡ ground truth shapes
Mat_<double> LoadGroundTruthShape(string& filename);

// Χ������ԭ����ת
Point2d rotationPoint(Point2d srcPoint, const double cosAngle, const double sinAngle);

// ��ʾ����
void DrawPredictedImage(Mat &image, cv::Mat_<double>& shape);

// ����landmarks
void saveLandmarks(const Mat_<double>& shape, const string& file);

/*****************************************************************
* ƽ�ƣ����ƽ�Ʊ�ע������С��Χ�е�[-5 % , 5 % ]��x��y�ֱ����
* ����ȡ������
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


BoundingBox CalculateBoundingBox2(Mat_<double>& shape); // ��÷Ŵ���Ŀ���

void DrawPredictedImage(Mat &image, cv::Mat_<double>& shape, BoundingBox bbx2);

void shiftSample(Mat& srcImage, Mat& ROI, Mat_<double>& shape, const int rateX, const int rateY);

// ���ƽ�ƺ��÷Ŵ���Ŀ���
BoundingBox CalculateBoundingBox3(Mat_<double>& shape, const double rateX, const double rateY);

// �߽���չ
void PaddingImage(Mat &image, BoundingBox& bbx);

// ��������ͼƬ����Ŀ���̶�Ϊ96*96��С
void ZoomImage(Mat &src, Mat &dst, BoundingBox& bbx, Mat_<double>& ground_truth_shape); // ��������


/***********************************************************************
* ���ȣ�ȫͼ����������ӻ��ȥ�̶�����ֵ����ΧΪͼ�����ؾ�ֵ�ġ�15%֮�䣻
***********************************************************************/
void adjustBright(Mat& srcImage, Mat& dstImage,
	const double& alpha, const double& beta);

void computeMean(Mat& srcImage, Scalar& mean);
void ZoomImage(Mat &src); // ��������

/***********************************************************************
* �ֱ��ʣ�ͼ����������²�������ͨ��˫���Բ�ֵ�ķ����Ŵ�ԭͼ��С��
* �²���ϵ��ȡֵ��Χ[0.4,1.0]
***********************************************************************/
void adjustResolution(Mat& srcImage, Mat& dstImage, const double& ratio);
void recoverSize(Mat& srcImage, Mat& dstImage, const double& ratio);
void average(const Mat &img, Point_<int> a, Point_<int> b, Vec3b &p);


#endif