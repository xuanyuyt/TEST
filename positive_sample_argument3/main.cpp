/*******************************************************************************************************


  * @file   main.cpp

  * @brief  ������data argument 2��

  * @date   2017:07:13 

  * @note   ��ת+λ��+����+�Աȶ�+�ֱ���

  * @version <version  number>

 *******************************************************************************************************/ 

#include "process3.h"
using namespace std;
using namespace cv;

string dataPath = "D:/Other_Dataets/Car/Tr1/part4/"; // part1~part4 ������ǰ�������Ӽ�
string savePath = "D:/Other_Dataets/Car/Positive/part20/";// part17~part20 ����������
int landmarks_num = 10;
Scalar value(0, 0, 0); // �߽�padding
int extRows = 300; // ��ֱpadding��С
int extCols = 300; // ˮƽpadding��С
int limitBoundingBoxWidth = 96; // Ŀ����޶����
int limitBoundingBoxHeight = 96; // Ŀ����޶��߶�
double rateX = 0; // Ŀ���X�������λ��
double rateY = 0; // Ŀ���Y�������λ��

int main()
{
	string path;
	path = dataPath + "Path_Images.txt";
	ifstream fin(path);
	string name;
	while (getline(fin, name))
	{
		name.erase(0, name.find_first_not_of("  ")); // ȥǰ��ո�
		name.erase(name.find_last_not_of("  ") + 1); // ȥ����ո�
		cout << "file:" << name << endl;
		string imageName = name.substr(name.find_last_of("/") + 1, 5);
		// Read Image
		Mat image = cv::imread(name, 1);
		if (image.data == NULL){
			cerr << "could not load " << name << endl;
			continue;
		}
		// 1.Read ground truth shapes
		name.replace(name.find_last_of("."), 4, ".txt"); // �滻��׺
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
			PaddingImage(rotaionImage, bbx2); // �߽���չ
		}

		// 5.Get ROI
		Mat  ROI;
		ROI = rotaionImage(Range(int(bbx2.start_y), int(bbx2.start_y) + bbx2.height), Range(int(bbx2.start_x), int(bbx2.start_x) + bbx2.width)); // ��ȡĿ��
		
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
		//adjustResolution(midImage, tmpImage, ratio); // ����²���
		//recoverSize(tmpImage, dstImage, ratio); // ˫���Բ�ֵ�ķ����Ŵ�
		resize(midImage, tmpImage, cv::Size(ratio*midImage.cols, ratio*midImage.rows), (0, 0), (0, 0), cv::INTER_AREA);
		resize(tmpImage, dstImage, cv::Size(midImage.cols, midImage.rows), (0, 0), (0, 0), cv::INTER_LINEAR);
		// 8.Zoom Image to make BoundingBox size is 96*96
		//ZoomImage(dstImage);
		resize(tmpImage, dstImage, cv::Size(96, 96), (0, 0), (0, 0), cv::INTER_LINEAR);

		/*namedWindow("show dstImage", 0);
		imshow("show dstImage", dstImage);
		waitKey(0);*/
		imwrite(savePath + "Pos" + imageName + "_7.jpg", dstImage); // ����������
	}

	return 0;
}
