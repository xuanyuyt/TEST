/*******************************************************************************************************


  * @file   main.cpp

  * @brief  正样本data argument 2倍

  * @date   2017:07:13 

  * @note   旋转+位移+亮度+对比度+分辨率

  * @version <version  number>

 *******************************************************************************************************/ 

#include "process3.h"
using namespace std;
using namespace cv;

string dataPath = "D:/Other_Dataets/Car/Tr1/part4/"; // part1~part4 是增广前的样本子集
string savePath = "D:/Other_Dataets/Car/Positive/part20/";// part17~part20 是正样本集
int landmarks_num = 10;
Scalar value(0, 0, 0); // 边界padding
int extRows = 300; // 垂直padding大小
int extCols = 300; // 水平padding大小
int limitBoundingBoxWidth = 96; // 目标框限定宽度
int limitBoundingBoxHeight = 96; // 目标框限定高度
double rateX = 0; // 目标框X方向随机位移
double rateY = 0; // 目标框Y方向随机位移

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
		float degree = random_generator.uniform(-100, 100) / 10.0;
		cout << "rotation " << degree << ", ";
		float angle = degree * (CV_PI / 180);
		Mat rotaionImage;
		imageRotation(image, rotaionImage, ground_truth_shape, angle);
		/*namedWindow("show dst", 0);
		DrawPredictedImage(rotaionImage, ground_truth_shape);
		waitKey(0);*/

		// 3.Random shift bbox
		rateX = random_generator.uniform(-5, 5) / 100.0;
		rateY = random_generator.uniform(-5, 5) / 100.0;
		cout << "shift X " << rateX * 100 << "%, Y " << rateY * 100 << "%, ";
		BoundingBox bbx2 = CalculateBoundingBox2(ground_truth_shape);
		/*DrawPredictedImage(rotaionImage, ground_truth_shape, bbx2);
		waitKey(0);*/

		// 4.Extend boundary if need
		if (bbx2.start_x < 0 || bbx2.start_y < 0
			|| (bbx2.start_x + bbx2.width) > rotaionImage.cols || (bbx2.start_y + bbx2.height) > rotaionImage.rows)
		{
			PaddingImage(rotaionImage, bbx2); // 边界拓展
		}

		// 5.Get ROI
		Mat  ROI;
		ROI = rotaionImage(Range(int(bbx2.start_y), int(bbx2.start_y) + bbx2.height), Range(int(bbx2.start_x), int(bbx2.start_x) + bbx2.width)); // 截取目标
		
		// 6.Random Adjust Image Brightness and Contrast
		Mat midImage;
		double beta = random_generator.uniform(-15, 15) / 100.0;
		double alpha = random_generator.uniform(90, 110) / 100.0;
		//cout << "Random adjust color:" << alpha << "* Image + (" << beta << " * mean)" << endl;
		adjustBright(ROI, midImage, alpha, beta);

		// 7.Random Adjust Image Resolution
		int time = random_generator.uniform(0, 8);
		double factor = 0.793700526;
		double ratio = pow(factor, time); // [1, 0.7937, 0.62996, 0.5, 0.391498, 0.25, 0.1984, 0.15749, 0.125]
		cout << "scale: " << ratio * 100 << "%" << endl;
		Mat dstImage, tmpImage;
		dstImage.create(midImage.size(), midImage.type());
		//adjustResolution(midImage, tmpImage, ratio); // 随机下采样
		//recoverSize(tmpImage, dstImage, ratio); // 双线性插值的方法放大
		resize(midImage, tmpImage, cv::Size(ratio*midImage.cols, ratio*midImage.rows), (0, 0), (0, 0), cv::INTER_AREA);
		resize(tmpImage, dstImage, cv::Size(midImage.cols, midImage.rows), (0, 0), (0, 0), cv::INTER_LINEAR);
		// 8.Zoom Image to make BoundingBox size is 96*96
		//ZoomImage(dstImage);
		resize(tmpImage, dstImage, cv::Size(96, 96), (0, 0), (0, 0), cv::INTER_LINEAR);

		/*namedWindow("show dstImage", 0);
		imshow("show dstImage", dstImage);
		waitKey(0);*/
		imwrite(savePath + "Pos" + imageName + "_7.jpg", dstImage); // 保存正样本
	}

	return 0;
}
