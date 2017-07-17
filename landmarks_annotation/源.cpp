/*******************************************************************************************************


  * @file   Դ.cpp

  * @brief  �������ע

  * @date   2017:07:13 

  * @note   ��Ҫ��ע����Ϣ������ͼ����Ŀ�공������ͷ����10�������㣺���粣�������ϡ����ϡ����¡����½ǵ㣬
			�������Ƶ����ĵ㣬��ǰ�����½Ǻ����½������㣬�������ĵ㣬�������ĵ㡣

  * @version <1>

 *******************************************************************************************************/ 

#include <iostream>
#include <fstream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
using namespace cv;
using namespace std;

#define WINDOW_NAME "�����򴰿ڡ�"        //Ϊ���ڱ��ⶨ��ĺ� 

void on_MouseHandle(int event, int x, int y, int flags, void* param);
void DrawLine(cv::Mat& img, cv::Point pt);

Point g_pt;
vector<Point> landmarks;
ofstream  OutFile;
string dataPath = "D:/Other_Dataets/����/my_samples/";
int landmarksNum = 10;


int main(int argc, char** argv)
{
	ifstream fin(dataPath + "Path_Images.txt");
	string name;
	while (getline(fin, name))
	{
		name.erase(0, name.find_first_not_of("  ")); // ȥǰ��ո�
		name.erase(name.find_last_not_of("  ") + 1); // ȥ����ո�
		cout << "file:" << name << endl;

		//��1��׼������
		g_pt = Point(0, 0);
		Mat srcImage = imread(name, 1);
		name.replace(name.find_last_of("."), 4, ".txt"); // �滻��׺
		OutFile.open(name, ios::out | ios::binary);
		OutFile << "version: 1" << endl;
		OutFile << " n_points: " << landmarksNum << endl;
		OutFile << "{" << endl;
		//��2�������������ص�����
		namedWindow(WINDOW_NAME, 1);
		setMouseCallback(WINDOW_NAME, on_MouseHandle, (void*)&srcImage);
		//��3��������ѭ���������л��Ƶı�ʶ��Ϊ��ʱ�����л���
		while (1)
		{
			imshow(WINDOW_NAME, srcImage);
			if (waitKey(10) == 27) break;//����ESC���������˳�
		}
		if (landmarks.size() != 10) // ����Ƿ��������ٱ��
		{
			cerr << "Wrong! " << name << endl;
			return -1;
		}
		OutFile << "}" << endl;
		OutFile.close();
		landmarks.clear();
	}

	return 0;
}


///*****************************************************************************************************
// Method:    on_MouseHandle
// Access:    public 
// Returns:   void
// Parameter: int event
// Parameter: int x
// Parameter: int y
// Parameter: int flags
// Parameter: void * param
// Function:  ���ص����������ݲ�ͬ������¼����в�ͬ�Ĳ���
///*****************************************************************************************************
void on_MouseHandle(int event, int x, int y, int flags, void* param)
{

	Mat& image = *(cv::Mat*) param;
	if (event == EVENT_LBUTTONDOWN)
	{
							  g_pt = Point(x, y);
							  landmarks.push_back(g_pt);
							  OutFile << g_pt.x << "  " << g_pt.y << endl;
							  //���ú������л���
							  DrawLine(image, g_pt);//����
	}
}

///*****************************************************************************************************
// Method:    DrawLine
// Access:    public 
// Returns:   void
// Parameter: cv::Mat & img
// Parameter: cv::Point pt
// Function:  ��ʾ����������
///*****************************************************************************************************
void DrawLine(cv::Mat& img, cv::Point pt)
{
	cv::line(img, pt, pt, cv::Scalar(0, 0, 255),5,8,0);//�����ɫ
}