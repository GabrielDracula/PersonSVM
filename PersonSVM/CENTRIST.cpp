#include <CENTRIST.h>

void GetSobel(const cv::Mat& gray, cv::Mat& sobel) {
	cv::Mat sobelx, sobely;
	cv::Sobel(gray, sobelx, CV_16S, 1, 0);
	cv::Sobel(gray, sobely, CV_16S, 0, 1);
	cv::Mat abs_sobelx, abs_sobely;
	cv::convertScaleAbs(sobelx, abs_sobelx);
	cv::convertScaleAbs(sobely, abs_sobely);
	cv::addWeighted(abs_sobelx, 1, abs_sobely, 1, 0, sobel);
}

void GetCT(const cv::Mat& sobel, cv::Mat& CTimage) {
	CTimage = cv::Mat::zeros(sobel.size(), CV_8UC1);
	for (int i = 1; i < sobel.rows - 1; i++) {
		for (int j = 1; j < sobel.cols - 1; j++) {
			if (sobel.at<uchar>(i, j) >= sobel.at<uchar>(i - 1, j - 1))
				CTimage.at<uchar>(i, j) += 128;
			if (sobel.at<uchar>(i, j) >= sobel.at<uchar>(i - 1, j))
				CTimage.at<uchar>(i, j) += 64;
			if (sobel.at<uchar>(i, j) >= sobel.at<uchar>(i - 1, j + 1))
				CTimage.at<uchar>(i, j) += 32;
			if (sobel.at<uchar>(i, j) >= sobel.at<uchar>(i, j - 1))
				CTimage.at<uchar>(i, j) += 16;
			if (sobel.at<uchar>(i, j) >= sobel.at<uchar>(i, j + 1))
				CTimage.at<uchar>(i, j) += 8;
			if (sobel.at<uchar>(i, j) >= sobel.at<uchar>(i + 1, j - 1))
				CTimage.at<uchar>(i, j) += 4;
			if (sobel.at<uchar>(i, j) >= sobel.at<uchar>(i + 1, j))
				CTimage.at<uchar>(i, j) += 2;
			if (sobel.at<uchar>(i, j) >= sobel.at<uchar>(i + 1, j + 1))
				CTimage.at<uchar>(i, j) += 1;
		}
	}
}

/*rownumber是Block行方向的个数，colnumber是Block列方向的个数*/
void Get_Block_Histogram(cv::Mat CTimage, int rownumber, int colnumber, cv::Mat& descriptor) {
	const int channels[1] = { 0 };
	const int histSize[1] = { 256 };
	float Range[2] = { 0, 256 };
	const float* Ranges[] = { Range };

	for (int i = 0; i < rownumber - 1; i++) {
		for (int j = 0; j < colnumber - 1; j++) {
			cv::Mat SuperBlock = cv::Mat(CTimage, cv::Rect(j * CTimage.cols / colnumber + 1,
				i * CTimage.rows / rownumber + 1,
				CTimage.cols / colnumber - 2,
				CTimage.rows / rownumber - 2));
			cv::Mat Block_descriptor;
			cv::calcHist(&SuperBlock, 1, channels, cv::Mat(),
				Block_descriptor,
				1, histSize, Ranges, true, false);

			cv::calcHist(&SuperBlock, 1, channels, cv::Mat(),
				Block_descriptor,
				1, histSize, Ranges, true, false);
			cv::transpose(Block_descriptor, Block_descriptor);

			if (descriptor.empty())
				descriptor = Block_descriptor.clone();
			else
				cv::hconcat(descriptor, Block_descriptor, descriptor);
		}
	}
}