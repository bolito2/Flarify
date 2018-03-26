#include <iostream>
#include <string>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

float multiplier = 8;
bool showDebug = true;

int main(int argc, char** argv) {
	Mat img = imread(argv[1], IMREAD_COLOR);
	Mat gris;
	cvtColor(img, gris, COLOR_BGR2GRAY);

	CascadeClassifier eyes_classifier("haarcascade_eye.xml");
	vector<Rect> eyes;

	eyes_classifier.detectMultiScale(gris, eyes, 1.3, 5);
	int maxEyeSize = 0;
	Mat debug;
	img.copyTo(debug);
	for (int i = 0; i < eyes.size(); i++) {
		rectangle(debug, eyes[i], Scalar(255, 0, 0));
		if (eyes[i].width > maxEyeSize)maxEyeSize = eyes[i].width;
	}
	if (showDebug)imshow("rectangulos", debug);

	if (maxEyeSize != 0) {
		float flareWidth = maxEyeSize*multiplier*1.5;
		float flareHeight = maxEyeSize*multiplier / 1.5;

		float flareCenterX = 655 * flareWidth / 1600;
		float flareCenterY = 445 * flareHeight / 889;

		Mat flare = imread("flare.png", IMREAD_UNCHANGED);
		resize(flare, flare, Size(flareWidth, flareHeight), 0, 0);
		vector<Mat> ch;
		split(flare, ch);
		ch[0] = ch[3];
		ch[1] = ch[3];
		ch[2] = ch[3];
		ch.pop_back();
		Mat flare_alpha;
		merge(ch, flare_alpha);

		flare = imread("flare.png", IMREAD_COLOR);
		resize(flare, flare, Size(flareWidth, flareHeight), 0, 0);

		flare.convertTo(flare, CV_32FC3, 1.0 / 255);
		flare_alpha.convertTo(flare_alpha, CV_32FC3, 1.0 / 255);
		img.convertTo(img, CV_32FC3, 1.0 / 255);

		multiply(flare_alpha, flare, flare);

		for (int i = 0; i < eyes.size(); i++) {
			float eyeCenterX = eyes[i].x + eyes[i].width / 2;
			float eyeCenterY = eyes[i].y + eyes[i].height / 2;

			Mat roi = img(Rect(eyeCenterX - flareCenterX, eyeCenterY - flareCenterY, flareWidth, flareHeight));

			multiply(Scalar::all(1.0) - flare_alpha, roi, roi);

			add(flare, roi, roi);
		}

	}
	img.convertTo(img, -1, 2, 0);
	char* heces = "_estallado.png";
	strcat(argv[1], heces);
	imshow("superposicion", img);
	
	img.convertTo(img, CV_8UC3, 255.0);
	imwrite(argv[1], img);

	waitKey(0);
	destroyAllWindows();
	return 0;
}