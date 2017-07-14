#include "annotion.h"

Annotate::Annotate()
{
	wname = "¡¾Ô´ÊÓÆµ¡¿";
	idx = 0; pidx = -1;
}

void Annotate::set_capture_instructions(){
	instructions.clear();
	instructions.push_back(string("Select expressive frames."));
	instructions.push_back(string("s - use this frame"));
	instructions.push_back(string("n - to the next frame"));
	instructions.push_back(string("q - done"));
}

void Annotate::draw_instructions(){
	if (image.empty())
		return;
	this->draw_strings(image, instructions);
}

void Annotate::draw_strings(Mat img, const vector<string> &text)
{
	for (int i = 0; i < int(text.size()); i++)
		this->draw_string(img, text[i], i + 1);
}

void Annotate::draw_string(Mat img,
	const string text,
	const int level)
{
	Size size = getTextSize(text, FONT_HERSHEY_COMPLEX, 0.6f, 1, NULL);
	putText(img, text, Point(0, level*size.height), FONT_HERSHEY_COMPLEX, 0.6f,
		Scalar::all(0), 1, CV_AA);
	putText(img, text, Point(1, level*size.height + 1), FONT_HERSHEY_COMPLEX, 0.6f,
		Scalar::all(255), 1, CV_AA);
}