#include <iostream>
#include <string>
#include <algorithm>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

float multiplier = 8;
bool showDebug = false;

float constantSize = 0.03f;

int main(int argc, char** argv) {
	if (argc > 3) {
		cout << "Error: too many arguments." << endl;
		return 69;
	}
	if (argc == 3 && strcmp(argv[1], "-d") == 0)showDebug = true;

	Mat img = imread(argc == 3 ? argv[2] : argv[1], IMREAD_COLOR);
	if (img.empty()){
		cout << "No existe la foto que intentas abrir." << endl;
		return 420;
	}
	cout << (img.rows + img.cols) / 2 << endl;
	Mat gris;
	cvtColor(img, gris, COLOR_BGR2GRAY);
	equalizeHist(gris, gris);

	CascadeClassifier eyes_classifier("haarcascade_eye.xml"), face_classifier("haarcascade_frontalface_default.xml");
	vector<Rect> eyes, faces;

	Mat debug;
	img.copyTo(debug);

	img.convertTo(img, CV_32FC3, 1.0 / 255);

	face_classifier.detectMultiScale(gris,faces, 1.1, 6);
	for (int f = 0; f < faces.size(); f++) {
		Mat face_roi = gris(faces[f]);
		Mat debug_roi = debug(faces[f]);
		rectangle(debug, faces[f], Scalar(0, 255, 0));

		vector<int> rejectLevels;
		vector<double> levelWeights;

		eyes_classifier.detectMultiScale(face_roi, eyes, rejectLevels,levelWeights,  1.1, 1, 0, Size(), Size(), true);
		int maxEyeSize = 0;
		
		for (int i = 0; i < eyes.size(); i++) {
			rectangle(debug_roi, eyes[i], Scalar(255, 0, 0));
			if (eyes[i].width > maxEyeSize)maxEyeSize = eyes[i].width;
			putText(debug, to_string(levelWeights[i]), Point(faces[f].x + eyes[i].x, faces[f].y + eyes[i].y - 25), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
			line(debug_roi, Point(eyes[i].x, eyes[i].y - 25), Point(eyes[i].x, eyes[i].y), Scalar(255,0,0));
		}
		if (showDebug)imshow("rectangulos", debug);
		
		if (eyes.size() > 1)
		{
			vector<Rect> realEyes;
			int maxIndex = max_element(levelWeights.begin(), levelWeights.end()) - levelWeights.begin();
			realEyes.push_back(eyes[maxIndex]);
			eyes.erase(eyes.begin() + maxIndex);
			levelWeights.erase(levelWeights.begin() + maxIndex);

			maxIndex = max_element(levelWeights.begin(), levelWeights.end()) - levelWeights.begin();
			realEyes.push_back(eyes[maxIndex]);

			eyes = realEyes;
		}
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

			multiply(flare_alpha, flare, flare);

			for (int i = 0; i < eyes.size(); i++) {
				float eyeCenterX = faces[f].x + eyes[i].x + eyes[i].width / 2;
				float eyeCenterY = faces[f].y + eyes[i].y + eyes[i].height / 2;

				Rect img_rect = Rect(0, 0, img.cols, img.rows);
				Rect roi_rect = Rect(eyeCenterX - flareCenterX, eyeCenterY - flareCenterY, flareWidth, flareHeight);
				Rect roi_rect_bounded = roi_rect & img_rect;
				Rect flare_roi_rect = Rect(0, 0, roi_rect.width, roi_rect.height) & Rect(roi_rect_bounded.x - (eyeCenterX - flareCenterX), roi_rect_bounded.y - (eyeCenterY - flareCenterY), roi_rect_bounded.width, roi_rect_bounded.height);

				Mat flare_roi = flare(flare_roi_rect);
				Mat flare_alpha_roi = flare_alpha(flare_roi_rect);
				Mat roi = img(roi_rect_bounded);

				//imshow(to_string(i) + "-flare_alpha : ", flare_alpha_roi);
				//imshow(to_string(i) + "-roi : ", roi);
				cout << eyes[i].width << endl;
				multiply(Scalar::all(1.0) - flare_alpha_roi, roi, roi);

				add(flare_roi, roi, roi);
			}
		}
	}
	
	img.convertTo(img, -1, 2, 0);
	imshow("superposicion", img);
	
	img.convertTo(img, CV_8UC3, 255.0);
	imwrite((argc == 3 ? string(argv[2]) : string(argv[1])) + "_estallado.png", img);
	

	waitKey(0);
	destroyAllWindows();
	return 0;
}