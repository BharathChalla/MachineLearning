#!/bin/bash
python_path=~/anaconda3/envs/cs6375/bin/python
project_path=~/Projects/PyCharm/CS6375ML/hw3
src_path=$project_path/src
data_path=$project_path/data
output_path=$project_path/output
echo "Training Tree Classifiers on CNF Dataset"
echo "DecisionTreeClassifier - CNF Boolean Dataset"
$python_path -u $src_path/part1/CNF_DecisionTreeClassifier.py $data_path/tree > $output_path/part1/CNF_DTC.txt 2>&1
echo "BaggingClassifier - CNF Boolean Dataset"
$python_path -u $src_path/part1/CNF_BaggingClassifier.py $data_path/tree > $output_path/part1/CNF_BC.txt 2>&1
echo "RandomForestClassifier - CNF Boolean Dataset"
$python_path -u $src_path/part1/CNF_RandomForestClassifier.py $data_path/tree > $output_path/part1/CNF_RFC.txt 2>&1
echo "GradientBoostingClassifier - CNF Boolean Dataset"
$python_path -u $src_path/part1/CNF_GradientBoostingClassifier.py $data_path/tree > $output_path/part1/CNF_GBC.txt 2>&1
echo -e "\n\n"

echo "Tree Classifiers on MNIST Dataset - Extra Credit"
$python_path -u $src_path/part1/MNIST_DecisionTreeClassifier.py $data_path/tree > $output_path/part1/MNIST_DTC.txt 2>&1
$python_path -u $src_path/part1/MNIST_BaggingClassifier.py $data_path/tree > $output_path/part1/MNIST_BC.txt 2>&1
$python_path -u $src_path/part1/MNIST_RandomForestClassifier.py $data_path/tree > $output_path/part1/MNIST_RFC.txt 2>&1
$python_path -u $src_path/part1/MNIST_GradientBoostingClassifier.py $data_path/tree > $output_path/part1/MNIST_GBC.txt 2>&1
echo -e "\n\n"

# To run individually
#$python_path -u $src_path/part2/KMeans.py $data_path/kmeans/Koala.jpg 5 $output_path/part2/kmeans/Koala-K5.jpg
#$python_path -u $src_path/part2/KMeans.py $data_path/kmeans/Penguins.jpg 5 $output_path/part2/kmeans/Peguins-K5.jpg

# Will run on the all Parameters with Images=[Koala, Penguins] and K = [2, 5, 10, 15, 20]
echo "Running KMeans on all Parameters with Images=[Koala, Penguins] and K = [2, 5, 10, 15, 20] & Initialization Seeds"
$python_path -u $src_path/part2/KMeans.py > $output_path/part2/Total 2>&1
