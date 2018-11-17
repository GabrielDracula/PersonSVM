"# PersonSVM" 
该程序仅使用OpenCV库实现，现已经实现基于CENTRIST描述子和SVM进行分类，判断有无行人，召回率较高但准确率仍较低。
Cite：
Wu J, Geyer C, Rehg J M. Real-time human detection using contour cues[C]//Robotics and Automation (ICRA), 2011 IEEE International Conference on. IEEE, 2011: 860-867.
尝试HOG特征并与其进行对比，

#TODO
加上滑窗算法以实现行人检测。若HOG准确率较高，则考虑使用CENTRIST和HOG进行级联。
