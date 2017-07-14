#include "head.h"

using namespace cv;
using namespace std;

string dataPath = "D:/Other_Dataets/����/Tr1/part4/"; // part1~part4 ������ǰ�������Ӽ�
string savePath = "D:/Other_Dataets/����/Negative/part4/"; // part1~part4 �������������Ӽ�
int landmarks_num = 10; // ����������
int negative_num = 12; // ÿ��ͼƬ������������
double IOU_threshold = 0.3; // IOU ��ֵ
int limitBoundingBoxWidth = 96; // Ŀ����޶����
int limitBoundingBoxHeight = 96; // Ŀ����޶��߶�

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
		cout << "file:" << name << endl;
		string imageName = name.substr(name.find_last_of("/") + 1, 5);
		// Read Image
		Mat image = cv::imread(name, 1);
		if (image.data == NULL){
			cerr << "could not load " << name << endl;
			continue;
		}
		// Read ground truth shapes
		name.replace(name.find_last_of("."), 4, ".txt"); // �滻��׺
		Mat_<double> ground_truth_shape = LoadGroundTruthShape(name);

		// Get Bounding box
		BoundingBox bbx = CalculateBoundingBox(ground_truth_shape);
		// Get negative sample
		cv::RNG random_generator(cv::getTickCount());
		for (int j = 0; j < negative_num; j++)
		{
			Mat ROI, dstImage;
			double randX, randY, IOU;
			BoundingBox bbxNeg;

			// ÿ�����������������
			do{
				randX = random_generator.uniform(0, image.cols - int(bbx.width));
				randY = random_generator.uniform(0, image.rows - int(bbx.height));
				BoundingBox tmp(randX, randY, bbx.width, bbx.height);
				bbxNeg = tmp;
				IOU = computIOU(bbxNeg, bbx);
			} while (IOU >= IOU_threshold);

			/*Mat tmpImage = image.clone();
			namedWindow("show IOU", 0);
			DrawROIImage(tmpImage, bbx, bbxNeg);*/
			cout << "sample negative sample " << j + 1 << ": IOU = " << IOU << endl;
			ROI = image(Range(bbxNeg.start_y, bbxNeg.start_y + bbxNeg.height), Range(bbxNeg.start_x, bbxNeg.start_x + bbxNeg.width)); // ��ȡ������
			resize(ROI, dstImage, Size(limitBoundingBoxWidth, limitBoundingBoxHeight));
			imwrite(savePath + imageName + "_" + to_string(j + 1) + ".jpg", dstImage); // ���渺����
			/*waitKey(0);*/
		}
	}
	return 0;
}