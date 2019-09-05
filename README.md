# Dublinbikes-Redistribution-plan-Optimisation-Capstone

##################################### File Guide #####################################

SAS Code Scripts:
		Activity Code.txt
		Hour Grouping Code.txt
		Weights and Totals Code.txt
 		Average Availability and Activity Code.txt
		
SAS Code PDFs:
		Activity Code.pdf
		Hour Grouping Code.pdf
		Weights and Totals Code.pdf
 		Average Availability and Activity Code.pdf

Input Data Files:
		position.csv
		DISTANCE_PROCESSED_CLUSTER.csv

Driving Distance Code:
		getdistance.py

Case 1 - 5 am on the 11th of February 2019 Files:

	Sample Dataset: clusterinput1102.csv

	Python Files: 	clustering_1102.py
		      	Optimisation_1102.py


Case 2 - 5 pm on the 20th of March 2019 Files:

	Sample Dataset: clusterinput2003.csv

	Python Files: 	clustering_2003.py
			Optimisation_2003.py

## Each code file contains detailed comments outlining the processes used ##


################################### SAS Code Files ###################################

The SAS code has been provided as text and PDF files - the have been provided in both
formats becasue they cannot be run without access to the SAS Viya system and are only 
provided as reference of the code used.  The PDF's provide a clearer outline of the 
code and the associated comments.

The SAS code files were run in the following order:
1. Activity Code
2. Hour Grouping Code
3. Weights and Totals Code
4. Average Availability and Activity Code


########################### To Run Driving Distances Code ############################

It is recommended that this code should be run using Jupyter Notebook through Anaconda.

This code requires the following files to be stored in the same directory as the 
respective code files:
			position.csv

getdistance.py implements the driving distances data retrieval process

The following steps outline how the code can be run using Jupyter Notebook:

Step 1: The osmnx and networkx packages may need to installed which can be completed 
	using the below links as installation guides:
		https://osmnx.readthedocs.io/en/stable/
		https://networkx.github.io/documentation/stable/install.html

Step 2: Activate the ox environment in Anaconda

Step 3: Modify the start-up directory in Jupyter notebook to match the location of the
	getdistance.py file

Step 4: Run Jupyter notebook through Anaconda

Step 5: Run the code file

Data outputs will be stored in a file called: raw_distance.csv


############################## To Run Clustering Code ################################

This code can be run using Windows Powershell, MacOS Terminal or Anaconda

This code requires the following files to be stored in the same directory as the 
respective code files:
			clusterinput1102.csv
			clusterinput2003.csv
			DISTANCE_PROCESSED_CLUSTER.csv

clustering_1102.py implements the clustering process for the 11th February 2019 
			(with the input data taken from clusterinput1102.csv)

clustering_2003.py implements the clustering process for the 20th March 2019 
			(with the input data taken from clusterinput2003.csv)

The following steps outline how the code can be run using Windows Powershell:

Step 1: Set the directory of Windows PowerShell to match the location of the 
	clustering_1102.py or clustering_2003.py file

Step 2: The input data needs to be stored in the same directory without changing the 
	file name (If the file name is changed - the code file will fail to run)

Step 3: Run the code file (This may take a few minutes to run)

Data will be output as either: cluster_output_data_1102.csv 
			       or cluster_output_data_2003.csv


############################# To Run Optimisation Code ###############################

##It is recommended that this code is run after running the clustering code##

This code can be run using Windows Powershell, MacOS Terminal or Anaconda

This code requires the following files to be stored in the same directory as the 
respective code files:
			cluster_output_data_1102.csv
			cluster_output_data_2003.csv
			DISTANCE_PROCESSED_CLUSTER.csv

Optimisation_1102.py implements the optimisation process for the 11th February 2019 
			(with the input data taken from cluster_output_data_1102.csv)

Optimisation_2003.py implements the optimisation process for the 20th March 2019 
			(with the input data taken from cluster_output_data_2003.csv)

The following steps outline how the code can be run using Windows Powershell:

Step 1: Set the directory of Windows PowerShell to match the location of the 
	Optimisation_1102.py or Optimisation_2003.py file

Step 2: The input data needs to be stored in the same directory without changing the 
	file name (If the file name is changed - the code file will fail to run)

Step 3: Run the code file (This may take a few minutes to run)

Data will be output as either: output_data_opted_1102.csv
			       or output_data_opted_2003.csv
