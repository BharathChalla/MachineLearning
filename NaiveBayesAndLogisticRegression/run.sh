#!/bin/bash
python_path=~/anaconda3/envs/cs6375/bin/python
project_path=~/PycharmProjects/CS6375ML/hw1/src
data_path=~/PycharmProjects/CS6375ML/hw1/data
echo "Multinomial Naive Bayes - Bag Of Words Dataset Model"
$python_path $project_path/Multinomial_Naive_Bayes.py $data_path/hw1
$python_path $project_path/Multinomial_Naive_Bayes.py $data_path/enron1
$python_path $project_path/Multinomial_Naive_Bayes.py $data_path/enron4
echo
echo "Discrete Naive Bayes - Bernoulli Dataset Model"
$python_path $project_path/Discrete_Naive_Bayes.py $data_path/hw1
$python_path $project_path/Discrete_Naive_Bayes.py $data_path/enron1
$python_path $project_path/Discrete_Naive_Bayes.py $data_path/enron4
echo -e "\n\n"

echo "MCAP Logistic Regression algorithm with L2 regularization - Bernoulli Dataset Model"
$python_path $project_path/LogisticRegression_Bernoulli.py $data_path/hw1
$python_path $project_path/LogisticRegression_Bernoulli.py $data_path/enron1
$python_path $project_path/LogisticRegression_Bernoulli.py $data_path/enron4

echo "MCAP Logistic Regression algorithm with L2 regularization - Bag Of Words Dataset Model"
$python_path $project_path/LogisticRegression_BagOfWords.py $data_path/hw1
$python_path $project_path/LogisticRegression_BagOfWords.py $data_path/enron1
$python_path $project_path/LogisticRegression_BagOfWords.py $data_path/enron4
echo -e "\n\n"

echo "SGDClassifier - Bernoulli Dataset Model"
$python_path $project_path/SGDC_Bernoulli.py $data_path/hw1
$python_path $project_path/SGDC_Bernoulli.py $data_path/enron1
$python_path $project_path/SGDC_Bernoulli.py $data_path/enron4
echo
echo "SGDClassifier - Bag Of Words Dataset Model"
$python_path $project_path/SGDC_BagOfWords.py $data_path/hw1
$python_path $project_path/SGDC_BagOfWords.py $data_path/enron1
$python_path $project_path/SGDC_BagOfWords.py $data_path/enron4
echo -e "\n\n"
