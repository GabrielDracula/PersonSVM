#include <CENTRIST.h>
#include <SVM.h>

bool train_INRIA() {
	std::fstream trainposlst, trainneglst;
	trainposlst.open(".\\INRIADATA\\normalized_images\\train\\pos.lst", std::ios::in);
	trainneglst.open(".\\INRIADATA\\normalized_images\\train\\neg.lst", std::ios::in);
	std::vector<std::string> trainpos, trainneg;
	while (!trainposlst.eof()) {
		std::string filename;
		trainposlst >> filename;
		filename = filename.substr(10);
		filename = ".\\INRIADATA\\normalized_images\\train\\pos\\" + filename;
		trainpos.push_back(filename);
	}
	trainposlst.close();
	
	while (!trainneglst.eof()) {
		std::string filename;
		trainneglst >> filename;
		filename = filename.substr(10);
		filename = ".\\INRIADATA\\normalized_images\\train\\neg\\" + filename;
		trainneg.push_back(filename);
	}
	trainneglst.close();
	
	std::cout << "����" << trainpos.size() << "��������ͼƬ" << std::endl;
	std::cout << "����" << trainneg.size() << "��������ͼƬ" << std::endl;

	cv::Mat pos_descriptors, pos_labels;
	std::cout << "������ȡ��������Ϣ��" << std::endl;
	for (int i = 0; i < trainpos.size(); i++) {
		cv::Mat image = cv::imread(trainpos[i]);
		image = cv::Mat(image, cv::Rect(18, 15, 60, 120));//���ݼ�ż���·�����ʶ���ȥ�·�һЩ
		//����ʹ������Ϊ120*60��ͼƬ�������Ϊ10*6��12*10������ͳ��ֱ��ͼ
		cv::Mat gray;
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
		cv::Mat sobel;
		GetSobel(gray, sobel);
		cv::Mat CTimage;
		GetCT(sobel, CTimage);
		cv::Mat descriptor;
		Get_Block_Histogram(CTimage, 6, 4, descriptor);
		//����˵�i��ͼƬ��descriptor
		if (pos_descriptors.empty())
			pos_descriptors = descriptor.clone();
		else
			cv::vconcat(pos_descriptors, descriptor, pos_descriptors);

		if (i % (trainpos.size() / 10) == 0)
			std::cout << i*100/trainpos.size() << "% ";
	}//��ȡ��ȫ��pos��descriptor
	std::cout << std::endl;
	 //labelsҪô��CV_32F��Ҫô��CV_32S�����к�����������
	pos_labels = cv::Mat(pos_descriptors.rows, 1, CV_32S, 1);
	//�����������������ȡ���
	std::cout << "������������Ϣ��ȡ��ϣ�" << std::endl;
	
	cv::Mat neg_descriptors, neg_labels;
	std::cout << "������ȡ��������Ϣ��" << std::endl;
	for (int i = 0; i < trainneg.size(); i++) {
		cv::Mat negimage = cv::imread(trainpos[i]);
		cv::Mat gray;
		cv::cvtColor(negimage, gray, cv::COLOR_BGR2GRAY);
		cv::Mat sobel;
		GetSobel(gray, sobel);
		cv::Mat CTimage;
		GetCT(sobel, CTimage);
		
		std::vector<cv::Mat> CTimage_Wins;
		/*CTimage_Wins.push_back(cv::Mat(CTimage, cv::Rect(0, 0, 96, 160)));
		CTimage_Wins.push_back(cv::Mat(CTimage, cv::Rect(CTimage.cols - 96, 0, 96, 160)));
		CTimage_Wins.push_back(cv::Mat(CTimage, cv::Rect(0, CTimage.rows - 160, 96, 160)));
		CTimage_Wins.push_back(cv::Mat(CTimage, cv::Rect(CTimage.cols - 96, CTimage.rows - 160, 96, 160)));*/
		CTimage_Wins.push_back(cv::Mat(CTimage, cv::Rect(CTimage.cols / 2 - 30, 0, 60, 120)));
		CTimage_Wins.push_back(cv::Mat(CTimage, cv::Rect(0, CTimage.rows / 2 - 60, 60, 120)));
		CTimage_Wins.push_back(cv::Mat(CTimage, cv::Rect(CTimage.cols / 2 - 30, CTimage.rows - 120, 60, 120)));
		CTimage_Wins.push_back(cv::Mat(CTimage, cv::Rect(CTimage.cols - 60, CTimage.rows / 2 - 60, 60, 120)));
		CTimage_Wins.push_back(cv::Mat(CTimage, cv::Rect(CTimage.cols / 2 - 30, CTimage.rows / 2 - 60, 60, 120)));
		if (CTimage.cols / 2 >= 50) {
			CTimage_Wins.push_back(cv::Mat(CTimage, cv::Rect(CTimage.cols / 2 - 50, CTimage.rows / 2 - 80, 96, 160)));
			CTimage_Wins.push_back(cv::Mat(CTimage, cv::Rect(CTimage.cols / 2 - 46, CTimage.rows / 2 - 80, 96, 160)));
		}
		if (CTimage.rows / 2 >= 82) {
			CTimage_Wins.push_back(cv::Mat(CTimage, cv::Rect(CTimage.cols / 2 - 48, CTimage.rows / 2 - 82, 96, 160)));
			CTimage_Wins.push_back(cv::Mat(CTimage, cv::Rect(CTimage.cols / 2 - 48, CTimage.rows / 2 - 78, 96, 160)));
		}
		
		
		for (int i = 0; i < CTimage_Wins.size(); i++) {
			cv::Mat descriptor;
			Get_Block_Histogram(CTimage_Wins[i], 6, 4, descriptor);
			if (neg_descriptors.empty())
				neg_descriptors = descriptor.clone();
			else
				cv::vconcat(neg_descriptors, descriptor, neg_descriptors);
		}

		if (i % (trainneg.size() / 10) == 0)
			std::cout << i * 100 / trainneg.size() << "% ";
	}//��ȡ��neg��descriptor
	std::cout << std::endl;
	neg_labels = cv::Mat(neg_descriptors.rows, 1, CV_32S, -1);
	//�����������������ȡ���
	std::cout << "������������Ϣ��ȡ��ϣ�" << std::endl;
	
	cv::Mat descriptors, labels;
	cv::vconcat(pos_descriptors, neg_descriptors, descriptors);
	cv::vconcat(pos_labels, neg_labels, labels);
	pos_descriptors.release();
	neg_descriptors.release();
	pos_labels.release();
	neg_labels.release();

	//Ptr��cv�����ռ��
	cv::Ptr<cv::ml::SVM> svm_model = cv::ml::SVM::create();
	svm_model->setType(cv::ml::SVM::C_SVC);
	svm_model->setKernel(cv::ml::SVM::LINEAR);
	//svm_model->setC(100.0);

	std::cout << "һ���� " << descriptors.rows << " ������" << std::endl;
	std::cout << "���������� " << descriptors.cols << " ά" << std::endl;
	
	std::cout << "����ѵ��SVMģ�� ..." << std::endl;
	svm_model->trainAuto(descriptors, cv::ml::ROW_SAMPLE, labels);
	if (svm_model->isTrained()) {
		std::cout << "ѵ���ɹ���" << std::endl;
		svm_model->save("centrist_person_svm_model.xml");
		std::cout << "������ɣ�" << std::endl;
		return true;
	}
	else {
		std::cout << "ѵ��ʧ�ܣ�" << std::endl;
		return false;
	}
	//return true;
}

void test_resize(cv::Mat& image) {
	image = cv::Mat(image, cv::Rect(1, 0, 67, 134));
	cv::resize(image, image, cv::Size(60, 120));
}

void compare(cv::Mat& labels, cv::Mat groundtruth) {
	int true_positive = 0, false_positive = 0, true_negative = 0, false_negative = 0;
	for (int i = 0; i < labels.rows; i++) {
		if (groundtruth.at<int>(i) > 0) {
			if (labels.at<int>(i) > 0)
				true_positive++;
			else
				false_negative++;
		}
		else {
			if (labels.at<int>(i) > 0)
				false_positive++;
			else
				true_negative++;
		}
	}

	float precision, recall;
	precision = (float)true_positive / (true_positive + false_positive);
	recall = (float)true_positive / (true_positive + false_negative);
	std::cout << "true positive: " << true_positive << std::endl;
	std::cout << "true negative: " << true_negative << std::endl;
	std::cout << "false positive: " << false_positive << std::endl;
	std::cout << "false negative: " << false_negative << std::endl;
	std::cout << "׼ȷ�ʣ�" << precision * 100 << "%" << std::endl;
	std::cout << "�ٻ���: " << recall * 100 << "%" << std::endl;
}

void test_INRIA(std::string pos_path, std::string neg_path) {
	std::vector<std::string> pos_images;
	getFiles(pos_path, pos_images);
	std::cout << "���õ�" << pos_images.size() << "��������ͼƬ" << std::endl;

	std::vector<std::string> neg_images;
	getFiles(neg_path, neg_images);
	std::cout << "���õ�" << neg_images.size() << "��������ͼƬ" << std::endl;
	
	cv::Mat pos_descriptors, pos_labels;
	std::cout << "������ȡ��������Ϣ��" << std::endl;
	for (int i = 0; i < pos_images.size(); i++) {
		cv::Mat image = cv::imread(pos_images[i]);
		if (image.empty())
			continue;
		test_resize(image);
		cv::Mat gray;
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
		cv::Mat sobel;
		GetSobel(gray, sobel);
		cv::Mat CTimage;
		GetCT(sobel, CTimage);
		cv::Mat descriptor;
		Get_Block_Histogram(CTimage, 6, 4, descriptor);
		//����˵�i��ͼƬ��descriptor
		if (pos_descriptors.empty())
			pos_descriptors = descriptor.clone();
		else
			cv::vconcat(pos_descriptors, descriptor, pos_descriptors);

		if (i % (pos_images.size() / 10) == 0)
			std::cout << i * 100 / pos_images.size() << "% ";
	}//��ȡ��ȫ��pos��descriptor
	std::cout << std::endl;
	//labelsҪô��CV_32F��Ҫô��CV_32S�����к�����������
	pos_labels = cv::Mat(pos_descriptors.rows, 1, CV_32S, 1);
	//�����������������ȡ���
	std::cout << "������������Ϣ��ȡ��ϣ�" << std::endl;

	cv::Mat neg_descriptors, neg_labels;
	std::cout << "������ȡ��������Ϣ��" << std::endl;
	for (int i = 0; i < neg_images.size(); i++) {
		cv::Mat image = cv::imread(neg_images[i]);
		if (image.empty())
			continue;

		cv::Mat gray;
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
		cv::Mat sobel;
		GetSobel(gray, sobel);
		cv::Mat CTimage;
		GetCT(sobel, CTimage);
		
		std::vector<cv::Mat> CTimage_Wins;
		CTimage_Wins.push_back(cv::Mat(CTimage, cv::Rect(0, 0, 60, 120)));
		CTimage_Wins.push_back(cv::Mat(CTimage, cv::Rect(CTimage.cols - 60, 0, 60, 120)));
		CTimage_Wins.push_back(cv::Mat(CTimage, cv::Rect(0, CTimage.rows - 120, 60, 120)));
		CTimage_Wins.push_back(cv::Mat(CTimage, cv::Rect(CTimage.cols - 60, CTimage.rows - 120, 60, 120)));

		for (int i = 0; i < CTimage_Wins.size(); i++) {
			cv::Mat descriptor;
			Get_Block_Histogram(CTimage_Wins[i], 6, 4, descriptor);
			if (neg_descriptors.empty())
				neg_descriptors = descriptor.clone();
			else
				cv::vconcat(neg_descriptors, descriptor, neg_descriptors);
		}

		if (i % (neg_images.size() / 10) == 0)
			std::cout << i * 100 / neg_images.size() << "% ";
	}
	std::cout << std::endl;
	neg_labels = cv::Mat(neg_descriptors.rows, 1, CV_32S, -1);
	//�����������������ȡ���
	std::cout << "������������Ϣ��ȡ��ϣ�" << std::endl;

	cv::Mat descriptors, groundtruth;
	cv::vconcat(pos_descriptors, neg_descriptors, descriptors);
	cv::vconcat(pos_labels, neg_labels, groundtruth);
	std::cout << "һ���� " << descriptors.rows << " ������" << std::endl;
	std::cout << "���������� " << descriptors.cols << " ά" << std::endl;
	
	std::cout << "��������SVMģ��..." << std::endl;
	cv::Ptr<cv::ml::SVM> svm_model = cv::ml::SVM::load("centrist_person_svm_model.xml");
	cv::Mat labels;
	svm_model->predict(descriptors, labels);
	compare(labels, groundtruth);
}
