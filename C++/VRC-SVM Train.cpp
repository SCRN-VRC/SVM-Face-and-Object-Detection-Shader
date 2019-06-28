/*
	The purpose of this file was to help me implement SVM functions in Cg
	fragment shaders by coding based on how data will be passed around in the
	render textures. Proof of concept only, will optimize in the future.
*/

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <stdlib.h>
#include <string.h>
#include <dirent.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

// Mimic f32 to f16 packing in a pixel
// without doing the actual conversion
typedef struct {
	float x[2];
	float y[2];
	float z[2];
	float w[2];
} float4;

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

static inline int is_odd_A(int x) { return x & 0x1; }

static inline int clip(int n, int lower, int upper) {
	return max(lower, min(n, upper));
}

void svmtrain(float *features, vector< int > & labels, bool train_auto, 
	float C, float gamma, int var_iters, int max_size, String name) {

	int *lbl = (int *)malloc(sizeof(int) * labels.size());
	for (int i = 0; i < labels.size(); i++) lbl[i] = labels[i];

	Mat trainingMat(min(max_size, labels.size()), 1568, CV_32F, features);
	Mat labelsMat(min(max_size, labels.size()), 1, CV_32SC1, lbl);
	
	Ptr<SVM> svm;
	svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	if (!train_auto) {
		svm->setC(C);
		svm->setGamma(gamma);
	}
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, var_iters, 1e-6));

	if (!train_auto)
	{
		svm->train(trainingMat, ROW_SAMPLE, labelsMat);
	}
	else
	{
		svm->trainAuto(trainingMat, ROW_SAMPLE, labelsMat);
	}
	svm->save(name);

	free(lbl);
}

int* myPredict(Ptr<SVM> & svm, float* testArray, int n) {
	int* labels = (int*)malloc(sizeof(int) * n);
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
			//cout << s << " " << -gamma << " " << s * -gamma << " " << exp(s*-gamma) << endl;
			buffer.at<float>(k) = exp(s*-gamma);
		}

		float total = -rho;
		for (int k = 0; k < sv.rows; k++)
			total += alpha.at<double>(k) * buffer.at<float>(svidx.at<int>(k));

		labels[x] = total > 0 ? 0 : 1;
	}

	return labels;
}

void svmtest(float *features, vector< int > & labels, String name) {
	Ptr<SVM> svm = Algorithm::load<SVM>(name);
	int *results = myPredict(svm, features, labels.size());
	int r = 0;
	for (int i = 0; i < labels.size(); i++) {
		if (results[i] == labels[i]) r++;
		clog << i << ". Expected " << labels[i] << " was " << results[i] << endl;
	}
	clog << r << "/" << labels.size() << endl;
	free(results);
}

void getHistArray(vector< Mat > & img_lst, float* features) {

	int c = 0;
	for (auto image : img_lst) {

		//// cartToPolar Layer
		//image.convertTo(image, CV_32F, 1 / 255.0);
		// Manually converting to lumanience 
		Mat imgLuma = Mat::zeros(image.rows, image.cols, CV_32F);
		for (int i = 0; i < image.rows; i++)
		{
			for (int j = 0; j < image.cols; j++)
			{
				Vec3b color = image.at<Vec3b>(Point(j, i));
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

		//// bins Layer
		float4 binsRT[8][8];

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

						// Storing it in the render tex
						binsRT[i][j].x[0] = bins[i][j][0];
						binsRT[i][j].x[1] = bins[i][j][1];
						binsRT[i][j].y[0] = bins[i][j][2];
						binsRT[i][j].y[1] = bins[i][j][3];
						binsRT[i][j].z[0] = bins[i][j][4];
						binsRT[i][j].z[1] = bins[i][j][5];
						binsRT[i][j].w[0] = bins[i][j][6];
						binsRT[i][j].w[1] = bins[i][j][7];
					}
				}
			}
		}

		// 16x16 Normalization
		// GPU implementation - Can't fit 7x7x32 into a 7x7 image
		// Scaled to 14x14x8, each pixel of 14x14 image will hold 8 values

		////float binsNorm[14][14][8];
		//for (int i = 0; i < 7; i++) {
		//	for (int j = 0; j < 7; j++) {
		//		float t = 0.0;
		//		for (int k = 0; k < 8; k++) {
		//			t += bins[i][j][k];
		//			t += bins[i + 1][j][k];
		//			t += bins[i][j + 1][k];
		//			t += bins[i + 1][j + 1][k];
		//		}

		//		for (int k = 0; k < 8; k++) {
		//			//binsNorm[i * 2][j * 2][k] = bins[i][j][k] / t;
		//			//binsNorm[i * 2 + 1][j * 2][k] = bins[i + 1][j][k] / t;
		//			//binsNorm[i * 2][j * 2 + 1][k] = bins[i][j + 1][k] / t;
		//			//binsNorm[i * 2 + 1][j * 2 + 1][k] = bins[i + 1][j + 1][k] / t;

		//			// Save to output format
		//			int ind = c * 1568;
		//			features[ind + ((i * 2) * 14 + (j * 2)) * 8 + k] = bins[i][j][k] / t;
		//			features[ind + ((i * 2 + 1) * 14 + (j * 2)) * 8 + k] = bins[i + 1][j][k] / t;
		//			features[ind + ((i * 2) * 14 + (j * 2 + 1)) * 8 + k] = bins[i][j + 1][k] / t;
		//			features[ind + ((i * 2 + 1) * 14 + (j * 2 + 1)) * 8 + k] =
		//				bins[i + 1][j + 1][k] / t;
		//		}
		//	}
		//}

		//for (int i = 0; i < 14; i++) {
		//	for (int j = 0; j < 14; j++) {
		//		cout << binsNorm[i][j][0] << " ";
		//		cout << binsNorm[i][j][1] << " ";
		//		cout << binsNorm[i][j][2] << " ";
		//		cout << binsNorm[i][j][3] << " ";
		//		cout << binsNorm[i][j][4] << " ";
		//		cout << binsNorm[i][j][5] << " ";
		//		cout << binsNorm[i][j][6] << " ";
		//		cout << binsNorm[i][j][7] << endl;
		//	}
		//}
		//getchar();

		
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
			}
		}

		//// HOGs (bins / normFactor) Layer
		float4 hogs[14][14];
		for (int i = 0; i < 14; i++) {
			for (int j = 0; j < 14; j++) {
				int si = floor(i / 2);
				int sj = floor(j / 2);
				int bi = si + (is_odd_A(i) ? 1 : 0);
				int bj = sj + (is_odd_A(j) ? 1 : 0);

				hogs[i][j].x[0] = binsRT[bi][bj].x[0] / sum[si][sj];
				hogs[i][j].x[1] = binsRT[bi][bj].x[1] / sum[si][sj];
				hogs[i][j].y[0] = binsRT[bi][bj].y[0] / sum[si][sj];
				hogs[i][j].y[1] = binsRT[bi][bj].y[1] / sum[si][sj];
				hogs[i][j].z[0] = binsRT[bi][bj].z[0] / sum[si][sj];
				hogs[i][j].z[1] = binsRT[bi][bj].z[1] / sum[si][sj];
				hogs[i][j].w[0] = binsRT[bi][bj].w[0] / sum[si][sj];
				hogs[i][j].w[1] = binsRT[bi][bj].w[1] / sum[si][sj];
			}
		}

		//for (int i = 0; i < 14; i++) {
		//	for (int j = 0; j < 14; j++) {
		//		cout << hogs[i][j].x[0] << " ";
		//		cout << hogs[i][j].x[1] << " ";
		//		cout << hogs[i][j].y[0] << " ";
		//		cout << hogs[i][j].y[1] << " ";
		//		cout << hogs[i][j].z[0] << " ";
		//		cout << hogs[i][j].z[1] << " ";
		//		cout << hogs[i][j].w[0] << " ";
		//		cout << hogs[i][j].w[1] << endl;
		//	}
		//}
		//getchar();

		//// Extract the data into one big array
		// 1x1568 render tex

		for (uint i = 0; i < 1568; i++) {
			uint x = i % 14;
			uint y = ((int) floor(i / 14)) % 14;
			uint z = i % 8;
			uint k = z >> 1;

			float f =
				k > 1 ?
				k > 2 ? hogs[x][y].w[is_odd_A(z) ? 1 : 0] :
				hogs[x][y].z[is_odd_A(z) ? 1 : 0] :
				k > 0 ? hogs[x][y].y[is_odd_A(z) ? 1 : 0] :
				hogs[x][y].x[is_odd_A(z) ? 1 : 0];
			features[c * 1568 + i] = f;
		}

		
		//for (uint i = 0; i < 1568; i++) {
		//	cout << features[c * 1568 + i] << " ";
		//}

		//getchar();

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
		c++;
	}
}

void list_dirs(const String & dirname, vector < String > & dir_list) {

	char char_array[200];
	strncpy_s(char_array, dirname.c_str(), 200);

	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(char_array)) != NULL) {

		while ((ent = readdir(dir)) != NULL) {
			if (ent->d_name[0] == '.') continue;
			dir_list.push_back(ent->d_name);
		}
		closedir(dir);
	}
}

void load_images(const String & dirname, vector< Mat > & img_lst, vector< int > & labels, int x)
{
	vector< String > files;
	glob(dirname, files);

	for (size_t i = 0; i < files.size(); ++i)
	{
		Mat img = imread(files[i]); // load the image
		if (img.empty())            // invalid image, skip it.
		{
			cout << files[i] << " is invalid!" << endl;
			continue;
		}
		img_lst.push_back(img);
		labels.push_back(x);
	}
}

int main(int argc, char** argv)
{
	const char* keys =
	{
		"{help h|     | show help message}"
		"{d     |     | directory containing images in folders separated by class}"
		"{auto  |false| train automatically to find optimal C and gamma values (OpenCV crashes a lot with this)}"
		"{c     |50   | sets C, a lower C will encourage a larger margin, therefore a simpler decision function, at the cost of training accuracy}"
		"{g     |0.05 | sets gamma, influence of single training sample with low values meaning 'far' and high values meaning 'close'}"
		"{test t|false| test a trained detector}"
		"{fn    |out.yaml  | file name of trained SVM}"
		"{i     |100  | training iterations}"
		"{s     |1000 | max samples to use for training}"
	};


	//// 0, 0 and loop start at bottom left
	//int width = 512, height = 512;
	//Mat image = Mat::zeros(height, width, CV_8UC3);
	//// Show the decision regions given by the SVM
	//Vec3b green(0, 255, 0), blue(255, 0, 0);
	//for (int i = image.rows - 1; i >= 0; i--)
	//{
	//	for (int j = 0; j < image.cols; j++)
	//	{
	//		image.at<Vec3b>(i, j) = Vec3b(image.rows - i - 1, j, 0);
	//	}
	//}

	imshow("SVM Simple Example", image);
	waitKey();
	return 0;

	CommandLineParser parser(argc, argv, keys);
	if (parser.has("help"))
	{
		parser.printMessage();
		cout << "Wrong number of parameters.\n\n"
			<< "Example command line:\n\"" << argv[0] << "\" -d=\"C:\\Users\\SCRN\\source\\repos\\VRC - SVM Train\\TestData\\Faces\" -c=50.0 -g=0.01"
			<< "\nExample command line for testing trained detector:\n\"" << argv[0] << "\" -d=\"C:\\Users\\SCRN\\source\\repos\\VRC - SVM Train\\TestData\\Faces\" -fn=\"C:\\Users\\SCRN\\source\\repos\\VRC - SVM Train\\Faces-SVM.yaml\" -t";
		exit(0);
	}

	String dir = parser.get< String >("d");
	String obj_det_filename = parser.get< String >("fn");
	double var_c = parser.get< double >("c");
	double var_gamma = parser.get< double >("g");
	int var_iters = parser.get< int >("i");
	int max_size = parser.get< int >("s");
	bool train_auto = parser.get< bool >("auto");
	bool test = parser.get< bool >("t");

	if (dir.empty())
	{
		parser.printMessage();
		cout << "Wrong number of parameters.\n\n"
			<< "Example command line:\n\"" << argv[0] << "\" -d=\"C:\\Users\\SCRN\\source\\repos\\VRC - SVM Train\\TestData\\Faces\" -c=50.0 -g=0.01"
			<< "\nExample command line for testing trained detector:\n\"" << argv[0] << "\" -d=\"C:\\Users\\SCRN\\source\\repos\\VRC - SVM Train\\TestData\\Faces\" -fn=\"C:\\Users\\SCRN\\source\\repos\\VRC - SVM Train\\Faces-SVM.yaml\" -t";
		exit(1);
	}
	vector< Mat > img_lst;
	vector< int > labels;
	vector< String > dir_list;

	float *features;

	list_dirs(dir, dir_list);

	if (dir_list.size() > 0)
	{
		clog << "\nClasses: " << dir_list.size() << endl;
		for (int i = 0; i < dir_list.size(); i++) {
			clog << "\t" << dir_list[i] << endl;
		}
	}
	else
	{
		clog << "no folders in " << dir << endl;
		return 1;
	}

	clog << "\nImages are being loaded...";

	for (int i = 0; i < dir_list.size(); i++)
		load_images(dir + "\\" + dir_list[i], img_lst, labels, i);

	if (img_lst.size() > 0)
	{
		clog << "...[done]" << endl;
	}
	else
	{
		clog << "no image in " << dir << endl;
		return 1;
	}

	features = (float*)malloc(sizeof(float) * 1568 * img_lst.size());
	clog << "Extracting features...";
	getHistArray(img_lst, features);
	clog << "...[done]\n";

	if (test)
	{
		clog << "Testing " << obj_det_filename << " on " << dir << "...\n";
		svmtest(features, labels, obj_det_filename);
	}
	else
	{
		clog << "Training on " << dir << "...";
		svmtrain(features, labels, train_auto, var_c, var_gamma, var_iters, 
			max_size, obj_det_filename);
	}
	clog << "...[done]\n";
	free(features);
	return 0;
}