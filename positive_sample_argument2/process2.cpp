#include "process2.h"
using namespace std;
using namespace cv;

///*****************************************************************************************************
// Method:    LoadGroundTruthShape
// Access:    public 
// Returns:   cv::Mat_<double>  特征点矩阵
// Parameter: string & filename 特征点文本路径
// Function:  读取 ground truth shapes
///*****************************************************************************************************
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

///*****************************************************************************************************
// Method:    CalculateBoundingBox2
// Access:    public 
// Returns:   BoundingBox
// Parameter: Mat_<double> & shape
// Function:  将 ground truth shapes 的最小包围矩形放大作为目标框
///*****************************************************************************************************
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

///*****************************************************************************************************
// Method:    DrawPredictedImage
// Access:    public 
// Returns:   void
// Parameter: Mat & image
// Parameter: cv::Mat_<double> & shape
// Parameter: BoundingBox bbx
// Function:  显示样本
///*****************************************************************************************************
void DrawPredictedImage(Mat &image, cv::Mat_<double>& shape, BoundingBox bbx){
	for (int i = 0; i < shape.rows; i++){
		circle(image, Point2d(shape(i, 0), shape(i, 1)), 2, Scalar(0, 0, 0), 5, 8, 0);
		circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 1, Scalar(255, 255, 255), 3, 8, 0);
	}
	cv::rectangle(image, cv::Rect(bbx.start_x, bbx.start_y, bbx.width, bbx.height),
		cv::Scalar(0, 0, 255), 2);
	imshow("show image", image);
	//waitKey(0);
}


///*****************************************************************************************************
// Method:    PaddingImage
// Access:    public 
// Returns:   void
// Parameter: Mat & image
// Parameter: BoundingBox & bbx
// Function:  边界拓展
///*****************************************************************************************************
void PaddingImage(Mat &image, BoundingBox& bbx)
{
	copyMakeBorder(image, image, extRows, extRows, extCols, extCols, BORDER_CONSTANT, value);//四周拓展边界
	bbx.start_x = bbx.start_x + extCols;
	bbx.start_y = bbx.start_y + extRows;
	bbx.centroid_x = bbx.centroid_x + extCols;
	bbx.centroid_y = bbx.centroid_y + extRows;
}


///*****************************************************************************************************
// Method:    adjustBright
// Access:    public 
// Returns:   void
// Parameter: Mat & srcImage
// Parameter: Mat & dstImage
// Parameter: const double & alpha
// Parameter: const double & beta
// Function:  调整图像亮度
///*****************************************************************************************************
void adjustBright(Mat& srcImage, Mat& dstImage,
	const double& alpha, const double& beta)
{
	dstImage.create(srcImage.size(), srcImage.type());
	Scalar mean(0,0,0,0);
	mean = cv::mean(srcImage); // 计算图像均值
	//computeMean( srcImage, mean);
	for (int y = 0; y < srcImage.rows; y++)
	{
		for (int x = 0; x < srcImage.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				dstImage.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(alpha*(srcImage.at<Vec3b>(y, x)[c]) + beta*mean[c]);
			}
		}
	}
}


///*****************************************************************************************************
// Method:    computeMean
// Access:    public 
// Returns:   void
// Parameter: Mat & image
// Parameter: Scalar & mean
// Function:  计算图像均值
///*****************************************************************************************************
void computeMean(Mat& image, Scalar& mean)
{

	for (int j = 0; j < image.rows; j++)
	{
		for (int i = 0; i < image.cols; i++)
		{
			mean[0] += image.at<cv::Vec3b>(j, i)[0];
			mean[1] += image.at<cv::Vec3b>(j, i)[1];
			mean[2] += image.at<cv::Vec3b>(j, i)[2];
			
		}
	}
	mean[0] = mean[0] / (image.rows * image.cols);
	mean[1] = mean[1] / (image.rows * image.cols);
	mean[2] = mean[2] / (image.rows * image.cols);
}


///*****************************************************************************************************
// Method:    adjustResolution
// Access:    public 
// Returns:   void
// Parameter: Mat & srcImage
// Parameter: Mat & dstImage
// Parameter: const double & ratio
// Function:  下采样调整图像分辨率
///*****************************************************************************************************
void adjustResolution(Mat& srcImage, Mat& dstImage, const double& ratio)
{
	// 缩小局部均值下采样
	Mat tmp;
	int rows = static_cast<int>(srcImage.rows * ratio);
	int cols = static_cast<int>(srcImage.cols * ratio);
	dstImage.create(rows, cols, srcImage.type());
	int lastRow = 0;
	int lastCol = 0;

	Vec3b *p;
	for (int i = 0; i < rows; i++) 
	{
		p = dstImage.ptr<Vec3b>(i);
		int row = static_cast<int>((i + 1) / ratio + 0.5) - 1;

		for (int j = 0; j < cols; j++) 
		{
			int col = static_cast<int>((j + 1) / ratio + 0.5) - 1;

			Vec3b pix;
			average(srcImage, Point_<int>(lastRow, lastCol), Point_<int>(row, col), pix);
			p[j] = pix;

			lastCol = col + 1; //下一个子块左上角的列坐标，行坐标不变
		}
		lastCol = 0; //子块的左上角列坐标，从0开始
		lastRow = row + 1; //子块的左上角行坐标
	}
	/*namedWindow("Scale Image", 0);
	imshow("Scale Image", tmp);*/
}


///*****************************************************************************************************
// Method:    recoverSize
// Access:    public 
// Returns:   void
// Parameter: Mat & srcImage
// Parameter: Mat & dstImage
// Parameter: const double & ratio
// Function:  缩放到目标图像大小，使用双线性插值
///*****************************************************************************************************
void recoverSize(Mat& srcImage, Mat& dstImage, const double& ratio)
{
	Vec3b *p;
	// 缩放, 双线性插值
	Vec3b *lastRows;
	Vec3b *nextRows;
	for (int i = 0; i < dstImage.rows; i++)
	{
		p = dstImage.ptr<Vec3b>(i);
		for (int j = 0; j < dstImage.cols; j++){
			{
				double row = i * ratio-0.5;
				double col = j * ratio-0.5;
				int lRow = static_cast<int>(row);
				int nRow = lRow + 1;
				int lCol = static_cast<int>(col);
				int rCol = lCol + 1;

				double u = row - lRow;
				double v = col - lCol;

				if ((row < srcImage.rows - 1) && (col < srcImage.cols - 1))
				{
					lastRows = srcImage.ptr<Vec3b>(lRow);
					nextRows = srcImage.ptr<Vec3b>(nRow);
					Vec3b f1 = v * lastRows[lCol] + (1 - v) * lastRows[rCol];
					Vec3b f2 = v * nextRows[lCol] + (1 - v) * lastRows[rCol];
					p[j] = u * f1 + (1 - u) * f2;      
				}
				//坐标在图像的右下角
				else if ((row >= srcImage.rows - 1) && (col >= srcImage.cols - 1))
				{
					lastRows = srcImage.ptr<Vec3b>(lRow);
					p[j] = lastRows[lCol];
				}
				//最后一行
				else if (row >= srcImage.rows - 1)
				{
					lastRows = srcImage.ptr<Vec3b>(lRow);
					p[j] = v * lastRows[lCol] + (1 - v) * lastRows[rCol];
				}
				//最后一列
				else if (col >= srcImage.cols - 1)
				{
					lastRows = srcImage.ptr<Vec3b>(lRow);
					nextRows = srcImage.ptr<Vec3b>(nRow);
					p[j] = u * lastRows[lCol] + (1 - u) * nextRows[lCol];
				}
			}
		}
	}
}


///*****************************************************************************************************
// Method:    average
// Access:    public 
// Returns:   void
// Parameter: const Mat & img
// Parameter: Point_<int> a
// Parameter: Point_<int> b
// Parameter: Vec3b & p
// Function:  计算局部图像块平均值
///*****************************************************************************************************
void average(const Mat &img, Point_<int> a, Point_<int> b, Vec3b &p)
{

	const Vec3b *pix;
	Vec3i temp;
	for (int i = a.x; i <= b.x; i++){
		pix = img.ptr<Vec3b>(i);
		for (int j = a.y; j <= b.y; j++){
			temp[0] += pix[j][0];
			temp[1] += pix[j][1];
			temp[2] += pix[j][2];
		}
	}

	int count = (b.x - a.x + 1) * (b.y - a.y + 1);
	p[0] = temp[0] / count;
	p[1] = temp[1] / count;
	p[2] = temp[2] / count;
}


///*****************************************************************************************************
// Method:    ZoomImage
// Access:    public 
// Returns:   void
// Parameter: Mat & src
// Function:  缩放整个图片，将目标框固定为96*96大小
///*****************************************************************************************************
void ZoomImage(Mat &src)
{
	double fx = double(limitBoundingBoxWidth) / src.cols;
	double fy = double(limitBoundingBoxHeight) / src.rows;
	resize(src, src, Size(96, 96), (0, 0), (0, 0), INTER_LINEAR); // 双线性插值缩放
}