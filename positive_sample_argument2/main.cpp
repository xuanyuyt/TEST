/*******************************************************************************************************


  * @file   main.cpp

  * @brief  正样本data argument

  * @date   2017:07:13 

  * @note
  * 图像处理扰动：
  * 1.亮度：全图随机整体增加或减去固定像素值，范围为图像像素均值的±15%之间；
  * 2.对比度：全图像素随机乘以一个系数，系数范围[0.9,1.1];
  * 3.分辨率：图像按随机比例下采样，再通过双线性插值的方法放大到原图大小，下采样系数取值范围[0.4,1.0]
  * 图像变换扰动:
  * 1.评议：随机平移标注坐标最小包围盒的[-5%,5%]，x、y分别随机

  * @version <1>

 *******************************************************************************************************/ 

#include "process2.h"
using namespace std;
using namespace cv;

string dataPath = "D:/Other_Dataets/Car/Tmp/part4/"; // part1~part8 是增广前的样本子集
string savePath = "D:/Other_Dataets/Car/Positive/part20/"; // part9~part16 是增广后的样本子集
int landmarks_num = 10;
Scalar value(0, 0, 0); // 边界padding
int extRows = 300; // 垂直padding大小
int extCols = 300; // 水平padding大小
int limitBoundingBoxWidth = 96; // 目标框限定宽度
int limitBoundingBoxHeight = 96; // 目标框限定高度
double rateX = 0; // 目标框X方向随机位移
double rateY = 0; // 目标框Y方向随机位移
int main(int argc, char** argv)
{
	string path;
	path = dataPath + "Path_Images.txt";
	ifstream fin(path);
	string name;
	while (getline(fin, name))
	{
		name.erase(0, name.find_first_not_of("  ")); // 去前面空格
		name.erase(name.find_last_not_of("  ") + 1); // 去后面空格
		//cout << "file:" << name << endl;
		string imageName = name.substr(name.find_last_of("/") + 1, 5);
		// Read Image
		Mat image = cv::imread(name, 1);
		if (image.data == NULL){
			cerr << "could not load " << name << endl;
			continue;
		}
		RNG random_generator(getTickCount());

		// 【1】Read ground truth shapes
		name.replace(name.find_last_of("."), 4, ".txt"); // 替换后缀
		Mat_<double> ground_truth_shape = LoadGroundTruthShape(name);

		// 【2】Get Bounding box(Random shift bbox)
		rateX = random_generator.uniform(-5, 5) / 100.0;
		rateY = random_generator.uniform(-5, 5) / 100.0;
		BoundingBox bbx2 = CalculateBoundingBox2(ground_truth_shape);
		//DrawPredictedImage(image, ground_truth_shape, bbx2);

		// 【3】Extend boundary if need
		if (bbx2.start_x < 0 || bbx2.start_y < 0
			|| (bbx2.start_x + bbx2.width) > image.cols || (bbx2.start_y + bbx2.height) > image.rows)
		{
			PaddingImage(image, bbx2); // 边界拓展
		}

		// 【4】Get ROI
		Mat  ROI;
		ROI = image(Range(int(bbx2.start_y), int(bbx2.start_y) + bbx2.height), Range(int(bbx2.start_x), int(bbx2.start_x) + bbx2.width)); // 截取目标
		/*namedWindow("Object Image", 0);
		imshow("Object Image", ROI);*/

		// 【5】Random Adjust Image Brightness and Contrast
		Mat midImage;
		double beta = random_generator.uniform(-15, 15) / 100.0;
		double alpha = random_generator.uniform(90, 110) / 100.0;
		//cout << "Random adjust color:" << alpha << "* Image + (" << beta << " * mean)" << endl;
		adjustBright(ROI, midImage, alpha, beta);
		/*namedWindow("New Brightness and Contrast Image", 0);
		imshow("New Brightness and Contrast Image", midImage);*/

		// 【6】Random Adjust Image Resolution
		double ratio = random_generator.uniform(4, 10) / 10.0;
		//cout << "Random adjust scale:" << ratio * 100 << "%" << endl;
		Mat dstImage, tmpImage;
		dstImage.create(midImage.size(), midImage.type());
		adjustResolution(midImage, tmpImage, ratio); // 随机下采样
		recoverSize(tmpImage, dstImage, ratio); // 双线性插值的方法放大

		// 【7】Zoom Image to make BoundingBox size is 96*96
		ZoomImage(dstImage);

		/*namedWindow("show dstImage", 0);
		imshow("show dstImage", dstImage);*/

		waitKey(0);
		imwrite(savePath + imageName + "_5.jpg", dstImage); // 保存正样本
	}

	return 0;
}