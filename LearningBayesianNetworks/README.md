Environment Setup:
Used Python3(3.9) and Anaconda Environment & Packaging Tool

Environment Setup Using Conda
conda create -n cs6375 python=3.9 numpy pandas scipy
conda activate cs6375

Notation:
TBN     - Tree Bayesian networks
MTBN_EM - Mixtures of Tree Bayesian networks using EM
MTBN_RF - Mixtures of Tree Bayesian networks using Random Forests

hw4
	-> data
			-> dataset (directory contains the 10 different datasets and each having - *.ts*(train), *.valid*(valid) and *.test*(test)
	-> src
			-> check_modules.py				- Checks the dependencies installed or not
			-> CLT_class.py					- Modified and added learn_by_setting_r_MI_zeroes funtion for Random Forests
			-> DatasetUtils.py				- DatasetUtils contains the dataset names in various orders and can uncomment and run easily
			-> Logger.py					- Logger to easily save the logging in case of multi-processing (can ignore)
			-> MIXTURE_CLT.py				- Implementation of the Mixtures of Tree Bayesian networks using EM (MTBN_EM)
			-> MIXTURE_CLT_RF.py			- Implementation of the Mixtures of Tree Bayesian networks using Random Forests (MTBN_RF)
			-> ParallelProcessing.py		- Examples of various methods of multi-processing (can ignore)
			-> TreeBayesianNetworks.py      - Helper file for running the Tree Bayesian networks using the CLT_class.py
			-> TreeBayesianNetworksMixEM.py - Helper file for running the MTBN_EM using the MIXTURE_CLT.py
			-> TreeBayesianNetworksMixRF.py	- Helper file for running the MTBN_RF using the MIXTURE_CLT_RF.py
			-> Util.py
	-> output
			-> TBN.txt      - Output for the Tree Bayesian Networks
			-> MTBN_EM.txt  - Mixtures of Tree Bayesian networks using EM
			-> MTBN_RF.txt  - Mixtures of Tree Bayesian networks using Random Forests.

Modules Check:
Run "check_modules.py" to check whether all the dependency modules installed or not.
$python_path $project_path/src/check_modules.py

## Note: Need to configure in the DatasetUtils.py for dataset names to run on
## Sorted the datasets to run in parallel

For Data:
Please download the 10 datasets shared on Teams and extract the files to the folder "dataset" in "hw4/data" folder.


For Parameters need to set in the corresponding file before running 

Please set up the Environment: python_path, project_path, src_path & data_path as environment variables before running.
python_path=~/anaconda3/envs/cs6375/bin/python
project_path=~/Projects/PyCharm/CS6375ML/hw4
src_path=$project_path/src
data_path=$project_path/data
output_path=$project_path/output

# To run individually need to set the parameters in the corresponding file before running
Run the below files for all the parameters settings which are configured in the corresponding file
#$python_path -u $src_path/TreeBayesianNetworks.py $data_path/dataset 2>&1 | tee > $output_path/TBN.txt
#$python_path -u $src_path/TreeBayesianNetworksMixEM.py $data_path/dataset 2>&1 | tee > $output_path/MTBN_EM.txt
#$python_path -u $src_path/TreeBayesianNetworksMixRF.py $data_path/dataset 2>&1 | tee > $output_path/MTBN_RF.txt


Else run the "run.sh" shell script by configuring the python path, project paths and 
the corresponding source and dataset paths in the script file to run all the parts.
Also, we can run individually by commenting the rest other than the file to run in "run.sh".

For the set of outputs we can refer the files in the output folder.
