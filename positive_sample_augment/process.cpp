#include "process.h"
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

void imageRotation(const Mat& srcImage, Mat& dstImage, Mat_<double>& shape, const float& angle)
{
	const double cosAngle = cos(angle);
	const double sinAngle = sin(angle);
	Mat_<double> tmp = shape.clone();
	// 计算标注中心
	double center_x = 0;
	double center_y = 0;
	for (int i = 0; i < shape.rows; i++){
		center_x += shape(i, 0);
		center_y += shape(i, 1);
		/*cout << shape(i, 0) << " " << shape(i, 1) << endl;*/
	}
	center_x /= shape.rows;
	center_y /= shape.rows;

	//原图像四个角的坐标变为以旋转中心的坐标系
	Point2d leftTop(-center_x, center_y); //(0,0)
	Point2d rightTop(srcImage.cols - center_x, center_y); // (width,0)
	Point2d leftBottom(-center_x, -srcImage.rows + center_y); //(0,height)
	Point2d rightBottom(srcImage.cols - center_x, -srcImage.rows + center_y); //(width,height)

	//以center为中心旋转后四个角的坐标
	Point2d transLeftTop, transRightTop, transLeftBottom, transRightBottom;
	transLeftTop = rotationPoint(leftTop, cosAngle, sinAngle);
	transRightTop = rotationPoint(rightTop, cosAngle, sinAngle);
	transLeftBottom = rotationPoint(leftBottom, cosAngle, sinAngle);
	transRightBottom = rotationPoint(rightBottom, cosAngle, sinAngle);

	//计算旋转后图像的width，height
	double left = min({ transLeftTop.x, transRightTop.x, transLeftBottom.x, transRightBottom.x });
	double right = max({ transLeftTop.x, transRightTop.x, transLeftBottom.x, transRightBottom.x });
	double top = min({ transLeftTop.y, transRightTop.y, transLeftBottom.y, transRightBottom.y });
	double down = max({ transLeftTop.y, transRightTop.y, transLeftBottom.y, transRightBottom.y });

	int width = static_cast<int>(fabs(left - right) + 0.5);
	int height = static_cast<int>(fabs(top - down) + 0.5);
	
	// 分配内存空间
	dstImage.create(height, width, srcImage.type());

	double dx = -abs(left) * cosAngle - abs(down) * sinAngle + center_x;
	double dy = abs(left) * sinAngle - abs(down) * cosAngle + center_y;

	// landmarks 旋转后坐标
	for (int k = 0; k < shape.rows; k++){
		shape(k, 0) = (tmp(k, 0) - center_x)*cosAngle - (tmp(k, 1) - center_y)*sinAngle + abs(left);
		shape(k, 1) = (tmp(k, 0) - center_x)*sinAngle + (tmp(k, 1) - center_y)*cosAngle + abs(down);
	}

	// 像素旋转
	int x, y;
	for (int i = 0; i < height; i++)// y
	{
		for (int j = 0; j < width; j++)// x
		{
			//坐标变换
			x = float(j)*cosAngle + float(i)*sinAngle + dx + 0.5;
			y = float(-j)*sinAngle + float(i)*cosAngle + dy + 0.5;

			if ((x<0) || (x >= srcImage.cols) || (y<0) || (y >= srcImage.rows))
			{
				if (srcImage.channels() == 3)
				{
					dstImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
				}
				else if (srcImage.channels() == 1)
				{
					dstImage.at<uchar>(i, j) = 0;
				}
			}
			else
			{
				if (srcImage.channels() == 3)
				{
					dstImage.at<cv::Vec3b>(i, j) = srcImage.at<cv::Vec3b>(y, x);
				}
				else if (srcImage.channels() == 1)
				{
					dstImage.at<uchar>(i, j) = srcImage.at<uchar>(y, x);
				}
			}
		}
	}
	/*for (int i = 0; i < shape.rows; i++){
		cout << shape(i, 0) << " " << shape(i, 1) << "  " << tmp(i, 0) << " " << tmp(i, 1) << endl;;
	}*/
}

// 围绕坐标原点旋转
Point2d rotationPoint(Point2d srcPoint, const double cosAngle, const double sinAngle)
{
	Point2d dstPoint;
	dstPoint.x = srcPoint.x * cosAngle + srcPoint.y * sinAngle;
	dstPoint.y = -srcPoint.x * sinAngle + srcPoint.y * cosAngle;
	return dstPoint;
}

// 显示样本
void DrawPredictedImage(Mat &image, cv::Mat_<double>& shape){
	for (int i = 0; i < shape.rows; i++){
		circle(image, Point2d(shape(i, 0), shape(i, 1)), 2, Scalar(0, 0, 0), 5, 8, 0);
		circle(image, Point2d(shape(i, 0), shape(i, 1)), 1, Scalar(255, 255, 255), 3, 8, 0);
	}
	imshow("show dst", image);
}

// 保存landmarks
void saveLandmarks(const Mat_<double>& shape, const string& file)
{
	ofstream  OutFile(file, ios::out | ios::binary);
	OutFile << "version: 1" << endl;
	OutFile << " n_points: " << landmarks_num << endl;
	OutFile << "{" << endl;
	for (int i = 0; i < shape.rows; i++){
		OutFile << shape(i, 0) << "  " << shape(i, 1) << endl;
	}
	OutFile << "}" << endl;
	OutFile.close();
}

void shiftSample(Mat& srcImage, Mat& ROI, Mat_<double>& shape, const int rateX, const int rateY)
{
	Mat dstImage;
	// Get Shit Bounding box
	BoundingBox bbx2 = CalculateBoundingBox3(shape, rateX, rateY);

	// Extend boundary if need
	if (bbx2.start_x < 0 || bbx2.start_y < 0
		|| (bbx2.start_x + bbx2.width) > srcImage.cols || (bbx2.start_y + bbx2.height) > srcImage.rows)
	{
		PaddingImage(srcImage, bbx2); // 边界拓展
	}

	// Zoom Image to make BoundingBox size is 96*96
	ZoomImage(srcImage, dstImage, bbx2, shape);

	// Get sample
	ROI = dstImage(Range(int(bbx2.start_y), int(bbx2.start_y) + bbx2.height), Range(int(bbx2.start_x), int(bbx2.start_x) + bbx2.width));
}



BoundingBox CalculateBoundingBox3(Mat_<double>& shape, const double rateX, const double rateY){
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

	tmpH = bottom_y - top_y;
	tmpW = right_x - left_x;
	tmpX = left_x + rateX*tmpW;
	tmpY = top_y + rateY*tmpH;

	bbx.start_x = tmpX - tmpW * 0.2;
	bbx.width = tmpW * 1.4;
	bbx.start_y = top_y - (bbx.width - tmpH) / 2;
	bbx.height = tmpW * 1.4;
	bbx.centroid_x = bbx.start_x + bbx.width / 2.0;
	bbx.centroid_y = bbx.start_y + bbx.height / 2.0;
	return bbx;
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
	resize(src, dst, Size(), fx, fy, INTER_LINEAR); // 双线性插值缩放
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