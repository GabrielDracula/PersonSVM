#include <CENTRIST.h>
#include <SVM.h>

int main(int argc, char** argv) {
	train_INRIA();
	std::string pos_path = ".\\INRIADATA\\normalized_images\\test\\pos";
	std::string neg_path = ".\\INRIADATA\\original_images\\test\\neg";
	test_INRIA(pos_path, neg_path);
	system("Pause");
	return 0;
}