#!/bin/bash
python_path=~/anaconda3/envs/cs6375/bin/python
project_path=~/Projects/PyCharm/CS6375ML/hw4
src_path=$project_path/src
data_path=$project_path/data
output_path=$project_path/output

echo "Tree Bayesian networks"
$python_path -u $src_path/TreeBayesianNetworks.py $data_path/dataset > $output_path/TBN.txt 2>&1

echo "Mixtures of Tree Bayesian networks using EM"
$python_path -u $src_path/TreeBayesianNetworksMixEM.py $data_path/dataset > $output_path/MTBN_EM.txt 2>&1

echo "Mixtures of Tree Bayesian networks using Random Forests."
$python_path -u $src_path/TreeBayesianNetworksMixRF.py $data_path/dataset > $output_path/MTBN_RF.txt 2>&1

echo -e "\n\n"

# To run individually
#$python_path -u $src_path/TreeBayesianNetworks.py $data_path/dataset > $output_path/TBN.txt 2>&1
#$python_path -u $src_path/TreeBayesianNetworksMixEM.py $data_path/dataset > $output_path/MTBN_EM.txt 2>&1
#$python_path -u $src_path/TreeBayesianNetworksMixRF.py $data_path/dataset > $output_path/MTBN_RF.txt 2>&1#$python_path -u $src_path/TreeBayesianNetworksMixRF.py $data_path/dataset 2>&1 | tee > $output_path/MTBN_RF.txt
