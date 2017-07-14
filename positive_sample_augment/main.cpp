/*******************************************************************************************************


  * @file   main.cpp

  * @brief  正样本data argument 2倍

  * @date   2017:07:13 

  * @note  * 图像变换扰动
		   * 1.按标注中心随机旋转一个角度，角度取值范围：[-10°,+10°]；
		   * 2 随机平移标注坐标最小包围盒的[-5%,5%]，x、y分别随机；
		   * 图像处理扰动：
		   * 1.亮度：全图随机整体增加或减去固定像素值，范围为图像像素均值的±15%之间；
		   * 2.对比度：全图像素随机乘以一个系数，系数范围[0.9,1.1];
		   * 3.分辨率：图像按随机比例下采样(范围[0.4,1.0]），再通过双线性插值的方法放大到原图大小

  * @version <1>

 *******************************************************************************************************/ 

#include "process.h"
using namespace std;
using namespace cv;

string dataPath = "D:/Other_Dataets/Car/Tr1/part2/"; // part1~part4 是增广前的样本子集
string savePath = "D:/Other_Dataets/Car/Tmp/part2/";// part5~part8 是增广后的样本子集
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
		Mat ROI;
		name.erase(0, name.find_first_not_of("  ")); // 去前面空格
		name.erase(name.find_last_not_of("  ") + 1); // 去后面空格
		cout << "file:" << name << endl;
		string imageName = name.substr(name.find_last_of("/") + 1, 5);
		// Read Image
		Mat image = cv::imread(name, 1);
		if (image.data == NULL){
			cerr << "could not load " << name << endl;
			continue;
		}
		// 1.Read ground truth shapes
		name.replace(name.find_last_of("."), 4, ".txt"); // 替换后缀
		Mat_<double> ground_truth_shape = LoadGroundTruthShape(name);

		// 2.Random rotation image&landmarks
		RNG random_generator(getTickCount());
		float degree = random_generator.uniform(-100, 100)/10.0; 
		cout << "rotation " << degree << ", ";
		float angle = degree * (CV_PI / 180);
		Mat rotaionImage;
		imageRotation(image, rotaionImage, ground_truth_shape, angle);

		/*namedWindow("show src", 0);
		imshow("show src", image);
		namedWindow("show dst", 0);
		DrawPredictedImage(rotaionImage, ground_truth_shape);
		waitKey(0)*/
		imwrite(savePath + imageName + "_1.jpg", rotaionImage); // 保存旋转后图片
		saveLandmarks(ground_truth_shape, savePath + imageName + "_1.txt"); // 保存landmarks

		//// 3.Random shift bbox and Extraction positive sample
		//double rateX = random_generator.uniform(-5, 5) / 100.0;
		//double rateY = random_generator.uniform(-5, 5) / 100.0;
		//cout << "shift X " << rateX * 100 << "%, Y " << rateY * 100 << "%" << endl;
		//shiftSample(rotaionImage, ROI, ground_truth_shape, rateX, rateY);

		///*namedWindow("show ROI", 0);
		//imshow("show ROI", ROI);
		//waitKey(0);*/
		//imwrite(savePath + imageName + "_1.jpg", ROI); // 保存旋转后图片

	}

	return 0;
}