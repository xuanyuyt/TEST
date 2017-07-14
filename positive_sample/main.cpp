/*******************************************************************************************************


  * @file   main.cpp

  * @brief  截取正样本

  * @date   2017:07:13 

  * @note
  * 1.得到包含10个特征点的外接矩形框，将矩形框左右分别外扩其宽度的20%，保持矩形框的中心不变，
	将矩形框的高度变为和宽度一致；
  * 2.将上一步得到的矩形框区域缩放至96*96大小的新图像上，缩放过程采用双线性插值方法。
  * 3.矩形框超出图像边界部分的区域用0值代替。

  * @version <1>

 *******************************************************************************************************/ 



#include "head.h"

using namespace std;
using namespace cv;

string dataPath = "D:/Other_Dataets/Car/Ts1/part5/";
string savePath = "D:/Other_Dataets/Car/Test/";
int landmarks_num = 10;
Scalar value(0, 0, 0); // 边界padding
int extRows = 300; // 垂直padding大小
int extCols = 300; // 水平padding大小
int limitBoundingBoxWidth = 96; // 目标框限定宽度
int limitBoundingBoxHeight = 96; // 目标框限定高度



int main()
{
	string path;
	path = dataPath + "Path_Images.txt";
	ifstream fin(path);
	string name;
	while (getline(fin, name))
	{
		name.erase(0, name.find_first_not_of("  ")); // 去前面空格
		name.erase(name.find_last_not_of("  ") + 1); // 去后面空格
		cout << "file:" << name << endl;
		string imageName = name.substr(name.find_last_of("/")+1, 5);
		// Read Image
		Mat image = cv::imread(name, 1);
		Mat dstImage, ROI;

		if (image.data == NULL){
			std::cerr << "could not load " << name << std::endl;
			continue;
		}
		// Read ground truth shapes
		name.replace(name.find_last_of("."), 4, ".txt"); // 替换后缀
		Mat_<double> ground_truth_shape = LoadGroundTruthShape(name);

		// Get Bounding box
		//BoundingBox bbx = CalculateBoundingBox(ground_truth_shape);
		BoundingBox bbx2 = CalculateBoundingBox2(ground_truth_shape);

		// Extend boundary if need
		if (bbx2.start_x < 0 || bbx2.start_y < 0
			|| (bbx2.start_x + bbx2.width) > image.cols || (bbx2.start_y + bbx2.height) > image.rows)
		{
			PaddingImage(image, bbx2); // 边界拓展
		}

		// Zoom Image to make BoundingBox size is 96*96
		ZoomImage(image, dstImage, bbx2, ground_truth_shape);

		// Get ROI
		ROI = dstImage.clone();
		ROI = ROI(Range(int(bbx2.start_y), int(bbx2.start_y) + bbx2.height), Range(int(bbx2.start_x), int(bbx2.start_x) + bbx2.width)); // 截取正样本

		// Test result
		/*namedWindow("show image", 1);
		DrawPredictedImage(dstImage, ground_truth_shape, bbx, bbx2);
		namedWindow("show ROI", 1);
		DrawROIImage(ROI);
		waitKey(0);*/
		imwrite(savePath + imageName + ".jpg", dstImage); // 保存正样本
	}

	return 0;
}
