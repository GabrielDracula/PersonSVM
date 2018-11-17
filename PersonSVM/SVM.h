#pragma once
#include <CENTRIST.h>
#include <files.h>

bool train_INRIA();
void test_INRIA(std::string pos_path, std::string neg_path);

bool train_INRIA_HOG();
void test_INRIA_HOG(std::string pos_path, std::string neg_path);