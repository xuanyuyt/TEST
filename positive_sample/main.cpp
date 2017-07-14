/*******************************************************************************************************


  * @file   main.cpp

  * @brief  ��ȡ������

  * @date   2017:07:13 

  * @note
  * 1.�õ�����10�����������Ӿ��ο򣬽����ο����ҷֱ��������ȵ�20%�����־��ο�����Ĳ��䣬
	�����ο�ĸ߶ȱ�Ϊ�Ϳ��һ�£�
  * 2.����һ���õ��ľ��ο�����������96*96��С����ͼ���ϣ����Ź��̲���˫���Բ�ֵ������
  * 3.���ο򳬳�ͼ��߽粿�ֵ�������0ֵ���档

  * @version <1>

 *******************************************************************************************************/ 



#include "head.h"

using namespace std;
using namespace cv;

string dataPath = "D:/Other_Dataets/Car/Ts1/part5/";
string savePath = "D:/Other_Dataets/Car/Test/";
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
		name.erase(0, name.find_first_not_of("  ")); // ȥǰ��ո�
		name.erase(name.find_last_not_of("  ") + 1); // ȥ����ո�
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
		name.replace(name.find_last_of("."), 4, ".txt"); // �滻��׺
		Mat_<double> ground_truth_shape = LoadGroundTruthShape(name);

		// Get Bounding box
		//BoundingBox bbx = CalculateBoundingBox(ground_truth_shape);
		BoundingBox bbx2 = CalculateBoundingBox2(ground_truth_shape);

		// Extend boundary if need
		if (bbx2.start_x < 0 || bbx2.start_y < 0
			|| (bbx2.start_x + bbx2.width) > image.cols || (bbx2.start_y + bbx2.height) > image.rows)
		{
			PaddingImage(image, bbx2); // �߽���չ
		}

		// Zoom Image to make BoundingBox size is 96*96
		ZoomImage(image, dstImage, bbx2, ground_truth_shape);

		// Get ROI
		ROI = dstImage.clone();
		ROI = ROI(Range(int(bbx2.start_y), int(bbx2.start_y) + bbx2.height), Range(int(bbx2.start_x), int(bbx2.start_x) + bbx2.width)); // ��ȡ������

		// Test result
		/*namedWindow("show image", 1);
		DrawPredictedImage(dstImage, ground_truth_shape, bbx, bbx2);
		namedWindow("show ROI", 1);
		DrawROIImage(ROI);
		waitKey(0);*/
		imwrite(savePath + imageName + ".jpg", dstImage); // ����������
	}

	return 0;
}
