#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/dnn/dnn.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <string>
#include <iostream>
#include <stdio.h>
#include <time.h>

#include <vector>
using namespace std;


int main(int argc,char* argv[]){
	//init time param
	clock_t t_start,t_end;

	//init dnn
	cv::Ptr<cv::dnn::Importer> importer;
	std::string modelTxt = "./deploy2.prototxt"; //"./deploy.prototxt"
	std::string modelBin = "./snapshot_iter_10000.caffemodel";// ./googlenet_finetune_web_car_iter_10000.caffemodel
	importer = cv::dnn::createCaffeImporter(modelTxt,modelBin);
	if(!importer){
		std::cerr << "Can't load network by using the following files: " << std::endl;
		std::cerr << "prototxt:   " << modelTxt << std::endl;
		std::cerr << "caffemodel: " << modelBin << std::endl;
		exit(-1);
	}
	cv::dnn::Net net;
	importer->populateNet(net);
	importer.release();

	//init cascades
	std::string cascadeModel = "./cars.xml";
	cv::CascadeClassifier cascade;
	if( !cascade.load(cascadeModel) ){
		std::cerr << "Can't load cascade model:" << std::endl;
		std::cerr << "cascade:   " << cascadeModel << std::endl;
		exit(-1);
	}

	cv::Mat frame;
	char image_path[32];

	/*int i = atoi(argv[1]);*/
	/*string path;
	path =  "D:/Other_Dataets/Car/Ts2/ImageName.txt";
	ifstream fin(path);
	string name;
	while (getline(fin, name)){*/
	for(int i=144;i<=1700;i++){

		t_start = clock();

		sprintf(image_path,"cars_input/in%06d.jpg",i);
		std::cout<<image_path<<std::endl;
		frame = cv::imread(image_path);
		std::vector<cv::Rect> cars;
		cascade.detectMultiScale(frame,cars,1.1,2,0);
		for(int j=0;j<cars.size();j++){
			cv::Mat car_candidate = frame(cars[j]);
			//cv::resize(car_candidate,car_candidate,cv::Size(224,224));
			cv::Mat inputBlob = cv::dnn::blobFromImage(car_candidate, 1, cv::Size(96, 96),
				cv::Scalar(104, 117, 123));   //Convert Mat to batch of images

			/*cv::dnn::Blob inputBlob = cv::dnn::Blob::fromImages(car_candidate);*/
			net.setInput(inputBlob, "data");
			net.forward();
			cv::Mat  prob = net.forward("prob");
;
			double classProb;
			cv::minMaxLoc(prob, NULL, &classProb);

			char temp[10];
			sprintf(temp, "%lf", classProb);
			std::string s(temp);
			cv::putText(frame,s,cv::Point(cars[j].x,cars[j].y-10),cv::FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,cv::Scalar::all(255),1,8);

			cv::rectangle(frame,cars[j],cv::Scalar(255,0,0));
		}

		t_end = clock();

		//show processing time
		std::cout<<"time:"<<double(t_end-t_start)/CLOCKS_PER_SEC<<std::endl;

		//show result
		cv::imshow("frame",frame);
		cv::waitKey(0);

	} //--end for loop
}
