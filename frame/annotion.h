#ifndef ANNOTION_H
#define ANNOTION_H
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Annotate{
public:
	int idx;                       //index of image to annotate
	int pidx;                      //index of point to manipulate
	Mat image;                     //current image to display 
	Mat image_clean;               //clean image to display
	const char* wname;             //display window name
	vector<string> instructions;   //annotation instructions

	Annotate();
	void draw_instructions();
	void set_capture_instructions(); //设置操作提示
protected:
	void draw_strings(Mat img, const vector<string> &text);
	void draw_string(Mat img, const string text, const int level);

};


#endif