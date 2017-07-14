/*******************************************************************************************************


  * @file   frame.cpp

  * @brief  目标采样

  * @date   2017:07:13 

  * @note   抽取出视频中包含车头的样本，按S键保存样本，N键换下一帧，Q键退出。

  * @version <1>

 *******************************************************************************************************/ 

#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "annotion.h"

using namespace std;
using namespace cv;

Annotate annotation;

int main()
{
	Mat im, img;
	VideoCapture cam;
	char str[1024];
	string ifile = "test2.mp4";
	int idx = 142;
	int flag = 1;
	int c = 0;
	//get data
	namedWindow(annotation.wname);
	cam.open(ifile); // 读取视频


	if (!cam.isOpened()){
		cout << "Failed opening video file." << endl
			<< "usage: ./annotate [-v video] [-m muct_dir] [-d output_dir]"
			<< endl;
		return 0;
	}
	cam >> im;
	annotation.set_capture_instructions();
	annotation.image = im.clone();
	annotation.draw_instructions();
	imshow(annotation.wname, annotation.image);

	while (im.empty() != true && flag != 0)
	{
		c = waitKey(0);
		switch ((char)c)
		{
		case 'q': // 退出
			cout << "Exiting ...\n";
			flag = 0;
			break;
		case 's': // 保存该帧
			annotation.image = im.clone();
			annotation.draw_instructions();
			imshow(annotation.wname, annotation.image);
			//idx = annotation.data.imnames.size();
			if (idx < 10)
				sprintf(str, "00%d.jpg", idx);
			else if (idx < 100)
				sprintf(str, "0%d.jpg", idx);
			else
				sprintf(str, "%d.jpg", idx);
			idx++;
			imwrite(str, im);
			//annotation.data.imnames.push_back(str);
			cam >> im;
			imshow(annotation.wname, annotation.image);
			break;
		case 'n': // 切换下一帧
			cam >> im;
			annotation.image = im.clone();
			annotation.draw_instructions();
			imshow(annotation.wname, annotation.image);
			break;
		}
	}


	return 0;
}