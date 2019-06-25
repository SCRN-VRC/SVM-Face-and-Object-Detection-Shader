#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml.hpp>

#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include <bitset>
#include <dirent.h>

using namespace cv;
using namespace cv::ml;
using namespace std;

const string _DIR_P(".\\TestData\\Faces-Test\\Positives\\");
const string _DIR_N(".\\TestData\\Faces-Test\\Negatives\\");

const float fX[3][3] = {
	{-1, 0, 1},
	{-2, 0, 2},
	{-1, 0, 1}
};

const float fY[3][3] = {
	{-1, -2, -1},
	{ 0,  0,  0},
	{ 1,  2,  1}
};

vector<string> listFiles(string path);
int* getLabelArray(vector<string> pathArray);
float* getHistArray(vector<string> pathArray);
int* myPredict(Ptr<SVM> svm, float* testArray, int n, int* labelArray);
string type2str(int type);

int main(int argc, char** argv) {
	/*
	// XOR test

	// Set up training data
	int labels[4] = { 1, -1, -1, 1 };
	float trainingData[4][2] = { {10, 10}, {490, 10}, {10, 490}, {280, 400} };
	Mat trainingDataMat(4, 2, CV_32F, trainingData);
	Mat labelsMat(4, 1, CV_32SC1, labels);
	// Train the SVM
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::SVM::RBF);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->trainAuto(trainingDataMat, ROW_SAMPLE, labelsMat);

	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);
	// Show the decision regions given by the SVM
	Vec3b green(0, 255, 0), blue(255, 0, 0);

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			//Mat sampleMat = (Mat_<float>(1, 2) << j, i);
			//float response = svm->predict(sampleMat);
			//if (response == 1)
			//	image.at<Vec3b>(i, j) = green;
			//else if (response == -1)
			//	image.at<Vec3b>(i, j) = blue;

			float testData[2];
			testData[0] = j; testData[1] = i;
			int* response = myPredict(svm, testData, 1, labels);
			if (response[0] == 1)
				image.at<Vec3b>(i, j) = green;
			else if (response[0] == -1)
				image.at<Vec3b>(i, j) = blue;
			free(response);
		}
	}

	Mat sampleMat = (Mat_<float>(1, 2) << 250, 250);
	cout << svm->predict(sampleMat) << endl;
	sampleMat = (Mat_<float>(1, 2) << 490, 490);
	cout << svm->predict(sampleMat) << endl;
	sampleMat = (Mat_<float>(1, 2) << 0, 0);
	cout << svm->predict(sampleMat) << endl;
	sampleMat = (Mat_<float>(1, 2) << 490, 0);
	cout << svm->predict(sampleMat) << endl;
	sampleMat = (Mat_<float>(1, 2) << 1, 1);
	cout << svm->predict(sampleMat) << endl;
	sampleMat = (Mat_<float>(1, 2) << 300, 300);
	cout << svm->predict(sampleMat) << endl;

	// Show the training data
	int thickness = -1;
	//circle(image, Point(10, 10), 5, Scalar(0, 0, 0), thickness);
	//circle(image, Point(490, 490), 5, Scalar(0, 0, 0), thickness);
	//circle(image, Point(10, 490), 5, Scalar(255, 255, 255), thickness);
	//circle(image, Point(250, 250), 5, Scalar(0, 0, 0), thickness);
	// Show support vectors
	thickness = 2;
	Mat sv = svm->getSupportVectors();
	for (int i = 0; i < sv.rows; i++)
	{
		const float* v = sv.ptr<float>(i);
		circle(image, Point((int)v[0], (int)v[1]), 5, Scalar(255, 255, 255), thickness);
	}
	imwrite("result.png", image);        // save the image
	imshow("SVM Simple Example", image); // show it to the user
	svm->save("SVM1.txt");

	int testLabels[6] = { 1, 1, 1, -1, 1, 1 };
	//float testData[3][2] = { {0, 0}, {490, 490}, {250, 250} };
	float testData[12] = { 250,250, 490,490, 0,0, 490,0, 1,1, 300,300 };
	int* response = myPredict(svm, testData, 6, labels);
	for (int i = 0; i < 6; i++)
		cout << testData[i*2] << ", " << testData[i*2+1] << ". " <<
			response[i] << " should be " << testLabels[i] << endl;

	waitKey();
	return 0;
	*/
	
	srand(time(NULL));
	Ptr<SVM> svm;

	vector<string> posList = listFiles(_DIR_P);
	posList.erase(posList.begin(), posList.begin() + 2);
	vector<string> negList = listFiles(_DIR_N);
	negList.erase(negList.begin(), negList.begin() + 2);

	vector<string> trainList;
	trainList.reserve(posList.size() + negList.size()); // preallocate memory
	trainList.insert(trainList.end(), posList.begin(), posList.end());
	trainList.insert(trainList.end(), negList.begin(), negList.end());

	//vector<string> testList;
	//testList.reserve(2);
	//testList.insert(testList.end(), trainList.begin(), trainList.begin() + 2);
	//trainList.erase(trainList.begin(), trainList.begin() + 2);

	// Set up training data
	float* trainData = getHistArray(trainList);
	int* labelArray = getLabelArray(trainList);
	//float* testData = getHistArray(testList);

//#define TRAIN
#ifdef TRAIN

	Mat trainingDataMat(trainList.size(), 1568, CV_32F, trainData);
	Mat labelsMat(trainList.size(), 1, CV_32SC1, labelArray);

	svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setC(5.0);
	svm->setGamma(1.5);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-3));
	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
	svm->save("SVM-Faces.yaml");

#else

	svm = svm->load("SVM-Faces.yaml");
	int* response = myPredict(svm, trainData, trainList.size(), labelArray);
	for (int i = 0; i < trainList.size(); i++)
		cout << i << ". " << response[i] << " should be " << labelArray[i] << endl;

	//int* response = myPredict(svm, trainData, trainList.size(), labelArray);
	//for (int i = 0; i < trainList.size(); i++)
	//	cout << i << ". " << response[i] << endl;

#endif

	cout << "Done!" << endl;
	getchar();

#ifdef TRAIN
	free(trainData);
	free(labelArray);
	//free(testData);
#else
	free(response);
#endif
	
	return 0;
}

int* myPredict(Ptr<SVM> svm, float* testArray, int n, int* labelArray) {
	int* labels = (int*)malloc(sizeof(int) * n);

	//string ty = type2str(alpha.type());
	//printf("Matrix: %s %dx%d \n", ty.c_str(), alpha.cols, alpha.rows);

	Mat sv = svm->getSupportVectors();
	//Ptr<Kernel> kernel = svm->getKernel();
	//Mat buffer(1, sv.rows, CV_32F);
	//kernel->calc(sv.rows, sv.cols, sv.ptr<float>(), data.ptr<float>(), buffer.ptr<float>());  // apply kernel on data (CV_32F vector) and support vectors
	
	Mat alpha, svidx;
	Mat buffer(1, sv.rows, CV_32F);
	
	double rho = svm->getDecisionFunction(0, alpha, svidx);
	double gamma = svm->getGamma();

	for (int x = 0; x < n; x++) {

		for (int k = 0; k < sv.rows; k++) {
			float s = 0;
			for (int l = 0; l < sv.cols; l++) {
				float t0 = sv.at<float>(k, l) - testArray[x * 1568 + l];
				//float t0 = sv.at<float>(k, l) - testArray[x * 2 + l];
				s += t0 * t0;
			}
			cout << s << " " << -gamma << " " << s * -gamma << " " << exp(s*-gamma) << endl;
			buffer.at<float>(k) = exp(s*-gamma);
		}

		float total = -rho;
		for (int k = 0; k < sv.rows; k++)
			total += alpha.at<double>(k) * buffer.at<float>(svidx.at<int>(k));

		labels[x] = total > 0 ? -1 : 1;
	}

	return labels;
}

inline int clip(int n, int lower, int upper) {
	return max(lower, min(n, upper));
}

float* getHistArray(vector<string> pathArray) {
	float* rtnArray = (float*)malloc(sizeof(float) * pathArray.size() * 1568);
	if (!rtnArray) {
		cerr << "Memory allocation failure" << endl;
		exit(-1);
	}

	int c = 0;
	for (auto path : pathArray) {
		Mat image;
		image = imread(path, IMREAD_COLOR); // Read the file
		if (image.empty())                      // Check for invalid input
		{
			cerr << "Could not open " << path << endl;
			continue;
		}
		//image.convertTo(image, CV_32F, 1 / 255.0);
		// Manually converting to lumanience 
		Mat imgLuma = Mat::zeros(image.rows, image.cols, CV_32F);
		for (int i = 0; i < image.rows; i++)
		{
			for (int j = 0; j < image.cols; j++)
			{
				Vec3b color = image.at<Vec3b>(Point(j, i));
				// Using 2 instead of 2.19921875 cause its faster
				float lin_r = (color[0] / 255.0);
				float lin_g = (color[1] / 255.0);
				float lin_b = (color[2] / 255.0);
				imgLuma.at<float>(i, j) = 0.2126*lin_r + 0.7152*lin_g + 0.0722*lin_b;
			}
		}
		// Manually calculating derivs
		Mat gx = Mat::zeros(imgLuma.rows, imgLuma.cols, CV_32F);
		Mat gy = Mat::zeros(imgLuma.rows, imgLuma.cols, CV_32F);
		Mat mag = Mat::zeros(imgLuma.rows, imgLuma.cols, CV_32F);
		Mat dir = Mat::zeros(imgLuma.rows, imgLuma.cols, CV_32F);

		for (int i = 0; i < imgLuma.rows; i++)
		{
			for (int j = 0; j < imgLuma.cols; j++)
			{
				int xL = clip(i - 1, 0, imgLuma.rows - 1);
				int xH = clip(i + 1, 0, imgLuma.rows - 1);
				int yL = clip(j - 1, 0, imgLuma.cols - 1);
				int yH = clip(j + 1, 0, imgLuma.cols - 1);
				float _00 = imgLuma.at<float>(xL, yL);
				float _01 = imgLuma.at<float>(xL, j);
				float _02 = imgLuma.at<float>(xL, yH);
				float _10 = imgLuma.at<float>(i, yL);
				float _11 = imgLuma.at<float>(i, j);
				float _12 = imgLuma.at<float>(i, yH);
				float _20 = imgLuma.at<float>(xH, yL);
				float _21 = imgLuma.at<float>(xH, j);
				float _22 = imgLuma.at<float>(xH, yH);
				gy.at<float>(i, j) = _00 * fY[0][0] + _01 * fY[0][1] + _02 * fY[0][2] +
					_10 * fY[1][0] + _11 * fY[1][1] + _12 * fY[1][2] +
					_20 * fY[2][0] + _21 * fY[2][1] + _22 * fY[2][2];
				gx.at<float>(i, j) = _00 * fX[0][0] + _01 * fX[0][1] + _02 * fX[0][2] +
					_10 * fX[1][0] + _11 * fX[1][1] + _12 * fX[1][2] +
					_20 * fX[2][0] + _21 * fX[2][1] + _22 * fX[2][2];
				mag.at<float>(i, j) = sqrt(pow(gx.at<float>(i, j), 2) +
					pow(gy.at<float>(i, j), 2));
				// Changing range to 0 - 180
				float dir_c = atan2(gy.at<float>(i, j), gx.at<float>(i, j))
					* 180.0 / 3.14159265358979323846;
				dir_c = dir_c < 0 ? dir_c + 360 : dir_c;
				dir_c = dir_c > 180 ? dir_c - 180 : dir_c;
				dir.at<float>(i, j) = dir_c;
			}
		}

		// https://www.learnopencv.com/histogram-of-oriented-gradients/

		// 8x8 binning
		// Modified HOG descriptors, 8 instead of 9 bins
		// Each bin is 22.5 degrees
		float bins[8][8][8];
		// 8x8 block loop
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				for (int k = 0; k < 8; k++) bins[i][j][k] = 0.0;
				// 8x8 pixel per block loop
				for (int x = 0; x < 8; x++) {
					for (int y = 0; y < 8; y++) {
						int ix = i * 8 + x;
						int jy = j * 8 + y;
						float direction = dir.at<float>(ix, jy);
						int binNo = ((int)(floor(direction / 22.5))) % 8;
						float ratio = fmod(dir.at<float>(ix, jy), 22.5) / 22.5;
						bins[i][j][binNo] += mag.at<float>(ix, jy) * (1.0 - ratio);
						bins[i][j][(binNo + 1) % 8] += mag.at<float>(ix, jy) * ratio;
					}
				}
			}
		}

		// 16x16 Normalization
		// GPU implementation - Can't fit 7x7x32 into a 7x7 image
		// Scaled to 14x14x8, each pixel of 14x14 image will hold 8 values
		
		//float binsNorm[14][14][8];
		for (int i = 0; i < 7; i++) {
			for (int j = 0; j < 7; j++) {
				float t = 0.0;
				for (int k = 0; k < 8; k++) {
					t += bins[i][j][k];
					t += bins[i + 1][j][k];
					t += bins[i][j + 1][k];
					t += bins[i + 1][j + 1][k];
				}

				for (int k = 0; k < 8; k++) {
					//binsNorm[i * 2][j * 2][k] = bins[i][j][k] / t;
					//binsNorm[i * 2 + 1][j * 2][k] = bins[i + 1][j][k] / t;
					//binsNorm[i * 2][j * 2 + 1][k] = bins[i][j + 1][k] / t;
					//binsNorm[i * 2 + 1][j * 2 + 1][k] = bins[i + 1][j + 1][k] / t;

					// Save to output format
					int ind = c * 1568;
					rtnArray[ind + ((i * 2) * 14 + (j * 2)) * 8 + k] = bins[i][j][k] / t;
					rtnArray[ind + ((i * 2 + 1) * 14 + (j * 2)) * 8 + k] = bins[i + 1][j][k] / t;
					rtnArray[ind + ((i * 2) * 14 + (j * 2 + 1)) * 8 + k] = bins[i][j + 1][k] / t;
					rtnArray[ind + ((i * 2 + 1) * 14 + (j * 2 + 1)) * 8 + k] =
						bins[i + 1][j + 1][k] / t;
				}
			}
		}
		c++;
		////Debugging

		////for (int i = 0; i < 1568; i++) cout << rtnArray[i] << " ";
		////getchar();

		//Mat concatImg[4];
		//concatImg[0] = gx;
		//concatImg[1] = gy;
		//concatImg[2] = mag;
		//concatImg[3] = dir;

		//Mat concatImg2[4];
		//Mat gx2, gy2, mag2, dir2;
		//image.convertTo(image, CV_32F, 1 / 255.0);
		//Sobel(image, gx2, CV_32F, 1, 0, 1);
		//Sobel(image, gy2, CV_32F, 0, 1, 1);
		//cartToPolar(gx2, gy2, mag2, dir2, 1);

		//concatImg2[0] = gx2;
		//concatImg2[1] = gy2;
		//concatImg2[2] = mag2;
		//concatImg2[3] = dir2;

		//int c = 0;
		//for (int i = 0; i < dir.rows; i++)
		//{
		//	for (int j = 0; j < dir.cols; j++)
		//	{
		//		float dir_c = dir2.at<float>(i, j);
		//		dir_c = dir_c > 180 ? dir_c - 180 : dir_c;

		//		dir2.at<float>(i, j) = dir_c / 180.0;

		//		dir.at<float>(i, j) = dir.at<float>(i, j) / 180.0;

		//		//if (abs(dir.at<float>(i, j) - dir_c) > 0.001) {
		//		//	cout << i << "," << j << " Should be " << dir_c << " instead of " <<
		//		//	dir.at<float>(i, j) << endl;
		//		//}
		//		//else c++;
		//		//cout.precision(5);
		//		//cout << dir.at<float>(i, j) << " ";
		//	}
		//	//cout << endl;
		//}

		////cout << c << "/" << dir.rows * dir.cols << " right" << endl;

		//Mat dst;
		//hconcat(concatImg, 4, dst);
		//Mat dst2;
		//hconcat(concatImg2, 4, dst2);

		//imshow("Mine", dst);
		//imshow("Default", dst2);
		//waitKey();
	}
	return rtnArray;
}

int* getLabelArray(vector<string> pathArray) {
	int* rtnArray = (int*)malloc(sizeof(int) * pathArray.size());
	if (!rtnArray) {
		cerr << "Memory allocation failure" << endl;
		exit(-1);
	}
	for (int ind = 0; ind < pathArray.size(); ind++) {
		bool eq = pathArray[ind].find("Positives") == string::npos;
		rtnArray[ind] = (eq == 0) ? 1 : -1;
	}
	return rtnArray;
}

char* concat(const char *s1, const char *s2)
{
	char *result = (char *)malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
	if (!result) {
		cerr << "Memory allocation failure" << endl;
		exit(-1);
	}
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

// Cause OpenCV is a piece of shit
string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

vector<string> listFiles(string path) {

	vector<string> rtnList;
	char char_array[100];
	strncpy(char_array, path.c_str(), 100);

	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(char_array)) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			char* s = concat(path.c_str(), ent->d_name);
			rtnList.push_back(s);
			free(s);
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		cerr << "Could not find directory " << path << endl;
		return { "" };
	}
	return rtnList;
}