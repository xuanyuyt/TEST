/*******************************************************************************************************


  * @file   main.cpp

  * @brief  ������data argument 2��

  * @date   2017:07:13 

  * @note  * ͼ��任�Ŷ�
		   * 1.����ע���������תһ���Ƕȣ��Ƕ�ȡֵ��Χ��[-10��,+10��]��
		   * 2 ���ƽ�Ʊ�ע������С��Χ�е�[-5%,5%]��x��y�ֱ������
		   * ͼ�����Ŷ���
		   * 1.���ȣ�ȫͼ����������ӻ��ȥ�̶�����ֵ����ΧΪͼ�����ؾ�ֵ�ġ�15%֮�䣻
		   * 2.�Աȶȣ�ȫͼ�����������һ��ϵ����ϵ����Χ[0.9,1.1];
		   * 3.�ֱ��ʣ�ͼ����������²���(��Χ[0.4,1.0]������ͨ��˫���Բ�ֵ�ķ����Ŵ�ԭͼ��С

  * @version <1>

 *******************************************************************************************************/ 

#include "process.h"
using namespace std;
using namespace cv;

string dataPath = "D:/Other_Dataets/Car/Tr1/part2/"; // part1~part4 ������ǰ�������Ӽ�
string savePath = "D:/Other_Dataets/Car/Tmp/part2/";// part5~part8 �������������Ӽ�
int landmarks_num = 10;
Scalar value(0, 0, 0); // �߽�padding
int extRows = 300; // ��ֱpadding��С
int extCols = 300; // ˮƽpadding��С
int limitBoundingBoxWidth = 96; // Ŀ����޶����
int limitBoundingBoxHeight = 96; // Ŀ����޶��߶�


int main()
{
	string path;
	path = dataPath + "Path_Images.txt";
	ifstream fin(path);
	string name;
	while (getline(fin, name))
	{
		Mat ROI;
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
		imwrite(savePath + imageName + "_1.jpg", rotaionImage); // ������ת��ͼƬ
		saveLandmarks(ground_truth_shape, savePath + imageName + "_1.txt"); // ����landmarks

		//// 3.Random shift bbox and Extraction positive sample
		//double rateX = random_generator.uniform(-5, 5) / 100.0;
		//double rateY = random_generator.uniform(-5, 5) / 100.0;
		//cout << "shift X " << rateX * 100 << "%, Y " << rateY * 100 << "%" << endl;
		//shiftSample(rotaionImage, ROI, ground_truth_shape, rateX, rateY);

		///*namedWindow("show ROI", 0);
		//imshow("show ROI", ROI);
		//waitKey(0);*/
		//imwrite(savePath + imageName + "_1.jpg", ROI); // ������ת��ͼƬ

	}

	return 0;
}