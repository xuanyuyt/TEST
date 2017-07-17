/*******************************************************************************************************


  * @file   源.cpp

  * @brief  特征点标注

  * @date   2017:07:13 

  * @note   需要标注的信息包含：图像中目标车辆（车头）的10个特征点：挡风玻璃的左上、右上、左下、右下角点，
			两个车灯的中心点，车前脸左下角和右下角两个点，车标中心点，车牌中心点。

  * @version <1>

 *******************************************************************************************************/ 

#include <iostream>
#include <fstream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
using namespace cv;
using namespace std;

#define WINDOW_NAME "【程序窗口】"        //为窗口标题定义的宏 

void on_MouseHandle(int event, int x, int y, int flags, void* param);
void DrawLine(cv::Mat& img, cv::Point pt);

Point g_pt;
vector<Point> landmarks;
ofstream  OutFile;
string dataPath = "D:/Other_Dataets/车辆/my_samples/";
int landmarksNum = 10;


int main(int argc, char** argv)
{
	ifstream fin(dataPath + "Path_Images.txt");
	string name;
	while (getline(fin, name))
	{
		name.erase(0, name.find_first_not_of("  ")); // 去前面空格
		name.erase(name.find_last_not_of("  ") + 1); // 去后面空格
		cout << "file:" << name << endl;

		//【1】准备参数
		g_pt = Point(0, 0);
		Mat srcImage = imread(name, 1);
		name.replace(name.find_last_of("."), 4, ".txt"); // 替换后缀
		OutFile.open(name, ios::out | ios::binary);
		OutFile << "version: 1" << endl;
		OutFile << " n_points: " << landmarksNum << endl;
		OutFile << "{" << endl;
		//【2】设置鼠标操作回调函数
		namedWindow(WINDOW_NAME, 1);
		setMouseCallback(WINDOW_NAME, on_MouseHandle, (void*)&srcImage);
		//【3】程序主循环，当进行绘制的标识符为真时，进行绘制
		while (1)
		{
			imshow(WINDOW_NAME, srcImage);
			if (waitKey(10) == 27) break;//按下ESC键，程序退出
		}
		if (landmarks.size() != 10) // 检查是否多标点或者少标点
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
// Function:  鼠标回调函数，根据不同的鼠标事件进行不同的操作
///*****************************************************************************************************
void on_MouseHandle(int event, int x, int y, int flags, void* param)
{

	Mat& image = *(cv::Mat*) param;
	if (event == EVENT_LBUTTONDOWN)
	{
							  g_pt = Point(x, y);
							  landmarks.push_back(g_pt);
							  OutFile << g_pt.x << "  " << g_pt.y << endl;
							  //调用函数进行绘制
							  DrawLine(image, g_pt);//画线
	}
}

///*****************************************************************************************************
// Method:    DrawLine
// Access:    public 
// Returns:   void
// Parameter: cv::Mat & img
// Parameter: cv::Point pt
// Function:  显示标点的特征点
///*****************************************************************************************************
void DrawLine(cv::Mat& img, cv::Point pt)
{
	cv::line(img, pt, pt, cv::Scalar(0, 0, 255),5,8,0);//随机颜色
}