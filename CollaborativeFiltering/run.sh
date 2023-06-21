#!/bin/bash
python_path=~/anaconda3/envs/cs6375/bin/python
project_path=~/PycharmProjects/CS6375ML/hw2
src_path=$project_path/src
data_path=$project_path/data
echo "Check Modules"
$python_path $src_path/check_modules.py
echo -e "\n\n"

echo "Assignment Part 1"
echo "Colloborative Filtering on Netflix Ratings Dataset"
$python_path $src_path/part1/netflix_collaborative_filtering.py $data_path/netflix
echo -e "\n\n"

echo "Assignment Part 2"
echo "SVM Classifier on MNIST Dataset"
$python_path $src_path/part2/MNIST_SVM_Classifier.py
echo -e "\n\n"
echo "MLP Classifier on MNIST Dataset"
$python_path $src_path/part2/MNIST_MLP_Classifier.py
echo -e "\n\n"
echo "KNN Classifier on MNIST Dataset"
$python_path $src_path/part2/MNIST_KNN_Classifier.py
echo -e "\n\n"