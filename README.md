# Melanoma_Federated_Learning
A federated learning framework for melanoma classification using SIIM-ISIC 2020 data

Instructions to run and load code: 
Jupyter notebooks are organized to address each project objective. The following seven .ipynb notebooks are submitted: 
Objective 1: Developing a classification model for Melanoma
1.	FL_Melanoma_obj1 Data analysis and Image based DataSet training
2.	FL_Melanoma_obj1 Feature Based DataSet training
Objective 2: Implementing FedAvg federated learning framework
3.	FL_Melanoma_obj2 Scenario split for Fedearated learning
4.	FL_Melanoma_obj2  FedAvg iid and non iid testing 1
5.	FL_Melanoma_obj2  FedAvg iid clients
Objective 3: Exploration of data distribution assumptions and novel application of federated learning aggregation strategies for our melanoma classification.
6.	FL_Melanoma_obj3  FedAdam and FedProx non iid testing 1
7.	FL_Melanoma_obj3  FedProx real case scenario split

The above notebooks use the following modules we implemented from scratch as the code behind our project. These modules are imported by the Import packages and modules script in the notebooks:
-	FedAvg
-	FLAdjustment
-	FLScenario
-	FLTrainVal
-	FLutils
-	FLWorker
-	FocalLoss
-	MelanomaDataset
-	MelanomaEfficientNet
-	Plot
-	ResizeImages
-	Test
-	TrainVal
-	utils
-	Visualise
Data
Source data can be downloaded from the link below, note we only use the training JPEG data:
-	https://challenge2020.isic-archive.com/
