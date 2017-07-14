/*******************************************************************************************************


  * @file   main.cpp

  * @brief  ������data argument

  * @date   2017:07:13 

  * @note
  * ͼ�����Ŷ���
  * 1.���ȣ�ȫͼ����������ӻ��ȥ�̶�����ֵ����ΧΪͼ�����ؾ�ֵ�ġ�15%֮�䣻
  * 2.�Աȶȣ�ȫͼ�����������һ��ϵ����ϵ����Χ[0.9,1.1];
  * 3.�ֱ��ʣ�ͼ����������²�������ͨ��˫���Բ�ֵ�ķ����Ŵ�ԭͼ��С���²���ϵ��ȡֵ��Χ[0.4,1.0]
  * ͼ��任�Ŷ�:
  * 1.���飺���ƽ�Ʊ�ע������С��Χ�е�[-5%,5%]��x��y�ֱ����

  * @version <1>

 *******************************************************************************************************/ 

#include "process2.h"
using namespace std;
using namespace cv;

string dataPath = "D:/Other_Dataets/Car/Tmp/part4/"; // part1~part8 ������ǰ�������Ӽ�
string savePath = "D:/Other_Dataets/Car/Positive/part20/"; // part9~part16 �������������Ӽ�
int landmarks_num = 10;
Scalar value(0, 0, 0); // �߽�padding
int extRows = 300; // ��ֱpadding��С
int extCols = 300; // ˮƽpadding��С
int limitBoundingBoxWidth = 96; // Ŀ����޶����
int limitBoundingBoxHeight = 96; // Ŀ����޶��߶�
double rateX = 0; // Ŀ���X�������λ��
double rateY = 0; // Ŀ���Y�������λ��
int main(int argc, char** argv)
{
	string path;
	path = dataPath + "Path_Images.txt";
	ifstream fin(path);
	string name;
	while (getline(fin, name))
	{
		name.erase(0, name.find_first_not_of("  ")); // ȥǰ��ո�
		name.erase(name.find_last_not_of("  ") + 1); // ȥ����ո�
		//cout << "file:" << name << endl;
		string imageName = name.substr(name.find_last_of("/") + 1, 5);
		// Read Image
		Mat image = cv::imread(name, 1);
		if (image.data == NULL){
			cerr << "could not load " << name << endl;
			continue;
		}
		RNG random_generator(getTickCount());

		// ��1��Read ground truth shapes
		name.replace(name.find_last_of("."), 4, ".txt"); // �滻��׺
		Mat_<double> ground_truth_shape = LoadGroundTruthShape(name);

		// ��2��Get Bounding box(Random shift bbox)
		rateX = random_generator.uniform(-5, 5) / 100.0;
		rateY = random_generator.uniform(-5, 5) / 100.0;
		BoundingBox bbx2 = CalculateBoundingBox2(ground_truth_shape);
		//DrawPredictedImage(image, ground_truth_shape, bbx2);

		// ��3��Extend boundary if need
		if (bbx2.start_x < 0 || bbx2.start_y < 0
			|| (bbx2.start_x + bbx2.width) > image.cols || (bbx2.start_y + bbx2.height) > image.rows)
		{
			PaddingImage(image, bbx2); // �߽���չ
		}

		// ��4��Get ROI
		Mat  ROI;
		ROI = image(Range(int(bbx2.start_y), int(bbx2.start_y) + bbx2.height), Range(int(bbx2.start_x), int(bbx2.start_x) + bbx2.width)); // ��ȡĿ��
		/*namedWindow("Object Image", 0);
		imshow("Object Image", ROI);*/

		// ��5��Random Adjust Image Brightness and Contrast
		Mat midImage;
		double beta = random_generator.uniform(-15, 15) / 100.0;
		double alpha = random_generator.uniform(90, 110) / 100.0;
		//cout << "Random adjust color:" << alpha << "* Image + (" << beta << " * mean)" << endl;
		adjustBright(ROI, midImage, alpha, beta);
		/*namedWindow("New Brightness and Contrast Image", 0);
		imshow("New Brightness and Contrast Image", midImage);*/

		// ��6��Random Adjust Image Resolution
		double ratio = random_generator.uniform(4, 10) / 10.0;
		//cout << "Random adjust scale:" << ratio * 100 << "%" << endl;
		Mat dstImage, tmpImage;
		dstImage.create(midImage.size(), midImage.type());
		adjustResolution(midImage, tmpImage, ratio); // ����²���
		recoverSize(tmpImage, dstImage, ratio); // ˫���Բ�ֵ�ķ����Ŵ�

		// ��7��Zoom Image to make BoundingBox size is 96*96
		ZoomImage(dstImage);

		/*namedWindow("show dstImage", 0);
		imshow("show dstImage", dstImage);*/

		waitKey(0);
		imwrite(savePath + imageName + "_5.jpg", dstImage); // ����������
	}

	return 0;
}