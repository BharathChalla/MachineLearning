
Environment Setup:
Used Python3(3.9) and Anaconda Environment & Packaging Tool
Ref: https://anaconda.org/conda-forge/scikit-learn-intelex
scikit-learn-intelex: Intel(R) Extension for Scikit-learn is a seamless way to speed up your Scikit-learn application.

Environment Setup Using Conda
conda create -n cs6375 python=3.9 numpy pandas scikit-learn 
conda activate cs6375
conda install -c conda-forge scikit-learn-intelex

Directory Structure:
hw2
-> data
		-> netflix (directory contains the netflix data files - TrainingRatings.txt, TestingRatings.txt)
		-> mnist (directory contains the mnist data files) # But MNIST data is downloaded and using directly
-> src
	-> check_modules.py
	-> part1 
			-> netflix_collaborative_filtering.py
	-> part2 
			-> MNIST_KNN_Classifier.py
			-> MNIST_MLP_Classifier.py
			-> MNIST_SVM_Classifier.py
-> out
	-> output files of the part1 and part2

Modules Check:
Run "check_modules.py" to check whether all the dependency modules installed or not.
$python_path $project_path/src/check_modules.py

For Part1:
Please download the Netflix dataset and extract the files to the folder "netflix" in "hw2/data" folder.

Need to pass the netflix ratings dataset folder path as argument to the file.
$python_path $project_path/src/part1/netflix_collaborative_filtering.py $project_path/data/netflix

For Part2:
MNIST data is downloaded and loaded programmatically.

Run the below files for all the parameters settings are configired in the file
$python_path $project_path/src/part2/MNIST_SVM_Classifier.py
$python_path $project_path/src/part2/MNIST_MLP_Classifier.py
$python_path $project_path/src/part2/MNIST_KNN_Classifier.py


Else run the "run.sh" shell script by configuring the python path, project paths and 
the corresponding source and dataset paths in the script file.

For the set of outputs we can refer the files in the output folder.
