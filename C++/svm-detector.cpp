/*
	The purpose of this file was to help me implement SVM functions in Cg
	fragment shaders by coding based on how data will be passed around in the
	render textures. Proof of concept only, will optimize in the future.

	https://www.learnopencv.com/histogram-of-oriented-gradients/
*/


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <stdlib.h>
#include <string.h>
#include <random>
#include <math.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

const float PI_F = 3.14159265358979f;

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

inline int clip(int n, int lower, int upper) {
	return max(lower, min(n, upper));
}

void** createArray(int i, int j, size_t size)
{
	void** r = (void**)calloc(i, sizeof(void*));
	for (int x = 0; x < i; x++) {
		r[x] = (void*)calloc(j, size);
	}
	return r;
}

float test(uint x, uint y, uint z, uint maxNum)
{
	float r;
	if (z == 0)
		r = (x / 64.0f) * (y / 64.0f) * 2.0f - 1.0f;
	else if (z == 1)
		r = ((64.0f - x) / 64.0f) * (y / 64.0f) * 2.0f - 1.0f;
	else
		r = (x / 64.0f) * ((64.0f - y) / 64.0f) * 2.0f - 1.0f;
	r = x > maxNum ? 0.0f : r;
	r = y > maxNum ? 0.0f : r;
	return r;
}

void freeArray(int i, int j, void** a)
{
	for (int x = 0; x < i; x++) {
		free(a[x]);
	}
	free(a);
}

void train(float **features, vector<int> imgC, bool t_auto,
	float C, float gamma, int iters, String name)
{
	int *lbl = (int*)malloc(sizeof(int) * imgC.size());
	float *f_block = (float*)malloc(sizeof(float) * 1568 * imgC.size());
	if (!lbl || !f_block) {
		std::clog << "No memory" << std::endl;
		exit(-1);
	}
	for (int i = 0; i < imgC.size(); i++) lbl[i] = imgC[i];
	for (int i = 0; i < imgC.size() * 1568; i++) f_block[i] = features[i / 1568][i % 1568];
	
	Mat trainingMat(imgC.size(), 1568, CV_32F, f_block);
	Mat labelsMat(imgC.size(), 1, CV_32SC1, lbl);

	Ptr<SVM> svm;
	svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);

	if (!t_auto) {
		svm->setC(C);
		svm->setGamma(gamma);
	}
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, iters, 1e-6));

	if (!t_auto) svm->train(trainingMat, ROW_SAMPLE, labelsMat);
	else svm->trainAuto(trainingMat, ROW_SAMPLE, labelsMat);
	svm->save(name);
	std::clog << "File saved " << name << std::endl;

	free(lbl);
	free(f_block);
}

float* predict(float** features, int n, String name) {
	Ptr<SVM> svm = Algorithm::load<SVM>(name);
	float* lout = (float*)malloc(sizeof(float) * n);
	if (!lout || !svm) {
		clog << "No memory" << std::endl;
		exit(-1);
	}
	Mat sv = svm->getSupportVectors();
	Mat alpha, svidx;
	Mat buffer(1, sv.rows, CV_32F);

	float rho = svm->getDecisionFunction(0, alpha, svidx);
	float gamma = svm->getGamma();

	// .at is slow as heck, might as well just put it in arrays
	float **fsv = (float**)createArray(sv.rows, sv.cols, sizeof(float));
	float *falpha = (float*)malloc(alpha.cols * sizeof(float));
	int *isvidx = (int*)malloc(svidx.cols * sizeof(int));
	float *buf = (float*)malloc(sv.rows * sizeof(float));

	for (int i = 0; i < sv.rows; i++)
	{
		for (int j = 0; j < sv.cols; j++)
		{
			fsv[i][j] = sv.at<float>(i, j);
		}
	}
	for (int i = 0; i < alpha.cols; i++) falpha[i] = alpha.at<double>(i);
	for (int i = 0; i < svidx.cols; i++) isvidx[i] = svidx.at<int>(i);

	for (int x = 0; x < n; x++) {
		for (int k = 0; k < sv.rows; k++) {
			float s = 0;
			for (int l = 0; l < sv.cols; l++) {
				float t0 = fsv[k][l] - features[x][l];
				s += t0 * t0;
			}
			buf[k] = exp(s * -gamma);
		}

		float s = -rho;
		for (int k = 0; k < sv.rows; k++)
			s += falpha[k] * buf[isvidx[k]];

		lout[x] = s;
	}

	freeArray(sv.rows, sv.cols, (void**)fsv);
	free(falpha);
	free(isvidx);
	free(buf);
	return lout;
}

// extract features
void getHogs(vector<Mat> images, float** features)
{
	int c = 0;
	for (auto img : images) {

		float imgL[64][64]; // Luma
		for (int i = 0; i < 64; i++)
		{
			for (int j = 0; j < 64; j++)
			{
				Vec3b color = img.at<Vec3b>(i, j);
				float lin_r = (color[2] / 255.0f);
				float lin_g = (color[1] / 255.0f);
				float lin_b = (color[0] / 255.0f);
				//float lin_r = test(i, j, 0, 64);
				//float lin_g = test(i, j, 1, 64);
				//float lin_b = test(i, j, 2, 64);
				imgL[i][j] = 0.2126f*lin_r + 0.7152f*lin_g + 0.0722f*lin_b;
			}
		}

		// Manually calculating gradients
		float gx[64][64]; // x edges
		float gy[64][64]; // y edges
		float mag[64][64]; // magnitude
		float dir[64][64]; // direction
		for (int i = 0; i < 64; i++)
		{
			for (int j = 0; j < 64; j++)
			{
				int xL = clip(i - 1, 0, 63);
				int xH = clip(i + 1, 0, 63);
				int yL = clip(j - 1, 0, 63);
				int yH = clip(j + 1, 0, 63);
				float _00 = imgL[xL][yL];
				float _01 = imgL[xL][j];
				float _02 = imgL[xL][yH];
				float _10 = imgL[i][yL];
				float _11 = imgL[i][j];
				float _12 = imgL[i][yH];
				float _20 = imgL[xH][yL];
				float _21 = imgL[xH][j];
				float _22 = imgL[xH][yH];

				gy[i][j] = _00 * fY[0][0] + _01 * fY[0][1] + _02 * fY[0][2] +
					_10 * fY[1][0] + _11 * fY[1][1] + _12 * fY[1][2] +
					_20 * fY[2][0] + _21 * fY[2][1] + _22 * fY[2][2];
				gx[i][j] = _00 * fX[0][0] + _01 * fX[0][1] + _02 * fX[0][2] +
					_10 * fX[1][0] + _11 * fX[1][1] + _12 * fX[1][2] +
					_20 * fX[2][0] + _21 * fX[2][1] + _22 * fX[2][2];

				mag[i][j] = sqrt(pow(gx[i][j], 2) + pow(gy[i][j], 2));
				dir[i][j] = atan2(gy[i][j], gx[i][j]) * 180.0f / PI_F; // -180, 180
				// unsigned magnitude
				dir[i][j] = dir[i][j] < 0.0f ? dir[i][j] + 180.0f : dir[i][j];
				//if (i == 0 && j == 0)
				//{
				//	std::clog << gx[i][j] << " ";
				//	std::clog << gy[i][j] << " ";
				//	std::clog << mag[i][j] << " ";
				//	std::clog << dir[i][j] << std::endl;
				//}
			}
		}

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
						int id = ((int)floor(dir[ix][jy] / 22.5f)) % 8;
						// Last bin wraps around to bin 0, ratio of magnitude to next bin
						float binR = dir[ix][jy] > 157.5f ? (dir[ix][jy] - 157.5f) / 22.5f: 0.0f;

						//if (i == 0 && j == 0 && x == 0 && y == 0)
						//{
						//	std::clog << dir[ix][jy] << " ";
						//	std::clog << mag[ix][jy] << " ";
						//	std::clog << id << " ";
						//	std::clog << binR << std::endl;
						//}

						bins[i][j][id] += mag[ix][jy] * (1.0f - binR);
						bins[i][j][(id + 1) % 8] += mag[ix][jy] * binR;
					}
				}
			}
		}

		//for (int i = 0; i < 8; i++) {
		//	for (int j = 0; j < 8; j++) {
		//		std::clog << i << ", " << j << " ";
		//		for (int k = 0; k < 8; k++) {
		//			std::clog << bins[i][j][k] << " ";
		//		}
		//		std::clog << std::endl;
		//	}
		//}
		//std::clog << std::endl;

		//// normFactor Layer
		float sum[7][7];
		for (int i = 0; i < 7; i++) {
			for (int j = 0; j < 7; j++) {
				sum[i][j] = 0.0;
				for (int k = 0; k < 8; k++) {
					sum[i][j] += bins[i][j][k];
					sum[i][j] += bins[i + 1][j][k];
					sum[i][j] += bins[i][j + 1][k];
					sum[i][j] += bins[i + 1][j + 1][k];
				}
				// squash noise
				sum[i][j] = sum[i][j] < 4.0f ? 999999.0f : sum[i][j];
			}
		}

		//// HOGs (bins / normFactor) Layer
		float hogs[14][14][8];
		for (int i = 0; i < 14; i++) {
			for (int j = 0; j < 14; j++) {
				int si = floor(i / 2);
				int sj = floor(j / 2);
				int bi = si + (i % 2);
				int bj = sj + (j % 2);

				hogs[i][j][0] = bins[bi][bj][0] / (sum[si][sj]);
				hogs[i][j][1] = bins[bi][bj][1] / (sum[si][sj]);
				hogs[i][j][2] = bins[bi][bj][2] / (sum[si][sj]);
				hogs[i][j][3] = bins[bi][bj][3] / (sum[si][sj]);
				hogs[i][j][4] = bins[bi][bj][4] / (sum[si][sj]);
				hogs[i][j][5] = bins[bi][bj][5] / (sum[si][sj]);
				hogs[i][j][6] = bins[bi][bj][6] / (sum[si][sj]);
				hogs[i][j][7] = bins[bi][bj][7] / (sum[si][sj]);
			}
		}

		//for (int i = 0; i < 14; i++) {
		//	for (int j = 0; j < 14; j++) {
		//		std::clog << i << ", " << j << " ";
		//		for (int k = 0; k < 8; k++) {
		//			std::clog << hogs[i][j][k] << " ";
		//		}
		//		std::clog << std::endl;
		//	}
		//}
		//std::clog << std::endl;

		bool nan = false;
		for (uint i = 0; i < 1568; i++) {
			uint j = (i / 112) % 14;
			uint k = (i / 8) % 14;
			uint l = i % 8;
			features[c][i] = hogs[j][k][l];
			nan = nan || isnan(features[c][i]);
			//std::clog << features[c][i] << " ";
		}
		//std::clog << std::endl;
		if (nan)
		{
			std::clog << "Warning: #" << c << " is NaN" << std::endl;
			//imshow("NaN", img);
			//waitKey();
		}

		c++;
	}
}

int main(int argc, char** argv)
{
	String posdir = "\\positive\\*.*";
	String negdir = "\\negative\\*.*";
	String tesdir = "\\test\\*.*";
	String fname = "svm-out.xml";
	int iterations = 100;

	if (argc == 1)
	{
		std::clog << "Please enter a directory." << endl;
		std::system("pause");
		return 0;
	}
	else if (argc >= 2)
	{
		String argv1(argv[1]);
		if ('\\' == argv1.back()) argv1.pop_back();

		posdir.insert(0, argv1);
		negdir.insert(0, argv1);
		tesdir.insert(0, argv1);

		if (argc >= 3)
		{
			iterations = stoi(argv[2]);
		}
	}

	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	srand(seed);

	vector<cv::String> fn;
	float **features;

	// train
	if (1)
	{
		vector<Mat> img;
		vector<int> imgC; // image class

		// positives
		cv::glob(posdir, fn, false);
		for (size_t i = 0; i < fn.size(); i++)
		{
			Mat imgIn = imread(fn[i]);
			if ((rand() % 2) == 0) flip(imgIn, imgIn, 1);

			// Rotate for HLSL
			cv::Point2f pc(imgIn.cols / 2., imgIn.rows / 2.);
			cv::Mat r = cv::getRotationMatrix2D(pc, -90, 1.0);
			cv::warpAffine(imgIn, imgIn, r, imgIn.size());

			img.push_back(imgIn);
			imgC.push_back(0); // Class 0
		}
	
		std::clog << fn.size() << " images in " << posdir << std::endl;

		// negatives
		fn.clear();
		cv::glob(negdir, fn, false);
		for (size_t i = 0; i < fn.size(); i++)
		{
			Mat imgIn = imread(fn[i]);
			if ((rand() % 2) == 0) flip(imgIn, imgIn, 1);

			// Rotate for HLSL
			cv::Point2f pc(imgIn.cols / 2., imgIn.rows / 2.);
			cv::Mat r = cv::getRotationMatrix2D(pc, -90, 1.0);
			cv::warpAffine(imgIn, imgIn, r, imgIn.size());

			img.push_back(imgIn);
			imgC.push_back(1); // Class 1
		}

		std::clog << fn.size() << " images in " << negdir << std::endl;

		// shuffle
		std::shuffle(img.begin(), img.end(), default_random_engine(seed));
		std::shuffle(imgC.begin(), imgC.end(), default_random_engine(seed));

		// total features per image
		features = (float**)createArray(img.size(), 1568, sizeof(float));

		std::clog << "Extracting features" << std::endl;
		getHogs(img, features);

		// train
		std::clog << "Training..." << std::endl;
		train(features, imgC, true, 2.5, 0.50625, iterations, fname);

		freeArray(img.size(), 1568, (void**)features);
	}

	// predict
	vector<Mat> img_t;
	fn.clear();
	cv::glob(tesdir, fn, false);
	for (size_t i = 0; i < fn.size(); i++)
	{
		Mat imgIn = imread(fn[i]);
		if ((rand() % 2) == 0) flip(imgIn, imgIn, 1);
		cv::Point2f pc(imgIn.cols / 2., imgIn.rows / 2.);
		cv::Mat r = cv::getRotationMatrix2D(pc, -90, 1.0);
		cv::warpAffine(imgIn, imgIn, r, imgIn.size());
		img_t.push_back(imgIn);
	}
	std::clog << fn.size() << " images in " << tesdir << std::endl;

	features = (float**)createArray(img_t.size(), 1568, sizeof(float));
	std::clog << "Extracting features" << std::endl;
	getHogs(img_t, features);

	std::clog << "Predict test set..." << std::endl;
	float* lout = predict(features, img_t.size(), fname);

	for (int i = 0; i < img_t.size(); i++)
	{
		std::size_t found = fn[i].find_last_of("/\\");
		std::clog << fn[i].substr(found + 1) << " " << lout[i] << std::endl;
	}

	free(lout);

	std::system("pause");
	return 0;
}