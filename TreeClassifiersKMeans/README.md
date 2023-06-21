Environment Setup:
Used Python3(3.9) and Anaconda Environment & Packaging Tool
Ref: https://anaconda.org/conda-forge/scikit-learn-intelex
scikit-learn-intelex: Intel(R) Extension for Scikit-learn is a seamless way to speed up your Scikit-learn application.

Environment Setup Using Conda
conda create -n cs6375 python=3.9 numpy pandas scikit-learn 
conda activate cs6375
conda install -c conda-forge scikit-learn-intelex

hw3
-> data
		-> tree (directory contains the CNF Boolean data files - train*, test* and valid* - train c[i] d[j]:csv)
		-> mnist (directory contains the mnist data files) # But MNIST data is downloaded and using directly
		-> kmeans (directory contains the two image files - Koala.jpg & Penguins.jpg)
-> src
	-> check_modules.py
	-> part1 
			-> CNF_BaggingClassifier.py
			-> CNF_DecisionTreeClassifier.py
			-> CNF_GradientBoostingClassifier.py
			-> CNF_RandomForestClassifier.py
			-> MNIST_BaggingClassifier.py
			-> MNIST_DecisionTreeClassifier.py
			-> MNIST_GradientBoostingClassifier.py
			-> MNIST_RandomForestClassifier.py
	-> part2 
			-> KMeans.java
			-> KMeans.py
			-> KMeansJava.py
-> output
	-> part1 
			-> Files containing the output content of part1
	-> part2
			-> Files containing the output content of part2
			-> images
					-> Contains the Compressed images where each image has naming as [ImageName]-K[k]-S[seed].jpg
						where k is no. of clusters and seed is the seed for the random number generator.

Modules Check:
Run "check_modules.py" to check whether all the dependency modules installed or not.
$python_path $project_path/src/check_modules.py


# To run individually
Please setup the python_path, project_path, src_path & data_path as environment variables before running.

python_path=~/anaconda3/envs/cs6375/bin/python
project_path=~/Projects/PyCharm/CS6375ML/hw3
src_path=$project_path/src
data_path=$project_path/data
output_path=$project_path/output

For Part1:
Please download the CNF Boolean dataset and extract the files to the folder "tree" in "hw3/data" folder.
MNIST data is downloaded and loaded programmatically.

# To run individually
Run the below files for all the parameters settings which are configured in the corresponding file
Need to pass the CNF Boolean dataset folder path as argument to the file.
$python_path -u $src_path/part1/CNF_DecisionTreeClassifier.py $data_path/tree > $output_path/part1/CNF_DTC.txt 2>&1
$python_path -u $src_path/part1/CNF_BaggingClassifier.py $data_path/tree > $output_path/part1/CNF_BC.txt 2>&1
$python_path -u $src_path/part1/CNF_RandomForestClassifier.py $data_path/tree > $output_path/part1/CNF_RFC.txt 2>&1
$python_path -u $src_path/part1/CNF_GradientBoostingClassifier.py $data_path/tree > $output_path/part1/CNF_GBC.txt 2>&1

$python_path -u $src_path/part1/MNIST_DecisionTreeClassifier.py $data_path/tree > $output_path/part1/MNIST_DTC.txt 2>&1
$python_path -u $src_path/part1/MNIST_BaggingClassifier.py $data_path/tree > $output_path/part1/MNIST_BC.txt 2>&1
$python_path -u $src_path/part1/MNIST_RandomForestClassifier.py $data_path/tree > $output_path/part1/MNIST_RFC.txt 2>&1
$python_path -u $src_path/part1/MNIST_GradientBoostingClassifier.py $data_path/tree > $output_path/part1/MNIST_GBC.txt 2>&1

For Part2:
Please download the images "Koala.jpg" & "Penguins.jpg" to the folder "kmeans" in "hw3/data" folder.
# To run individually
$python_path -u $src_path/part2/KMeans.py $data_path/kmeans/Koala.jpg 5 $output_path/part2/kmeans/Koala-K5.jpg
$python_path -u $src_path/part2/KMeans.py $data_path/kmeans/Penguins.jpg 5 $output_path/part2/kmeans/Peguins-K5.jpg

If we don't pass any parameters runs on all parameters configured in the file.
$python_path -u $src_path/part2/KMeans.py > $output_path/part2/Total 2>&1


Else run the "run.sh" shell script by configuring the python path, project paths and 
the corresponding source and dataset paths in the script file to run all the parts.
Also we can run individually by commenting the rest other than the file to run in "run.sh".

For the set of outputs we can refer the files in the output folder.
