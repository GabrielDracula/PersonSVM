#pragma once

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

void GetSobel(const cv::Mat& gray, cv::Mat& sobel);

void GetCT(const cv::Mat& sobel, cv::Mat& CTimage);

void Get_Block_Histogram(cv::Mat CTimage, int rownumber, int colnumber, cv::Mat& descriptor);
