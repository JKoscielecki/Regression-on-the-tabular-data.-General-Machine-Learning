# Regression-on-the-tabular-data.-General-Machine-Learning

This repository contains:

● jupyter notebook with exploratory data analysis (EDA)

● train.py python script for model training

● predict.py python script for model inference on test data

● Train dataset (train.csv) contains 53 anonymized features and a target column and dataset (hidden_test.csv) that contains only 53 features to be predicted. Task is to build a model that predicts a target based on the proposed features.

● Saved model (model.pkl) 

● File with prediction results (hidden_test_predictions.csv)

● requirements.txt files

Description
The EDA includes exploratory analysis, checking for missing values, data distributions, linearity, variable types, etc. Plots such as scatterplots, line plots, and box plots are also created. The correlation is checked and a correlation matrix is drawn. The data is first normalized using methods such as MinMaxScaler and then STDscaler, and the results are compared. Then, linear regression is performed for each factor and the results are analyzed, but the results are weak. Attempts were made to train the model on several best factors using feature selection methods such as correlation-based or RFE or PCA, but the results were still weak. Other models such as random forest regressor and neural networks with different layers and neurons were also tried, but the results were still poor. Finally, the method of PolynomialFeatures(degree=2, include_bias=False) with normal regression was used, and it brought excellent results. The trained model was then evaluated, and the following are the metrics for the test dataset: RMSE: 9.204990941349006e-12, Mean squared error: 8.473185823031727e-23, R-squared: 1.0.


● Installation

1. Clone the repository
2. If you want to run all files including EDA then install requirements.txt. 
If you only want to run train.py and predict.py, all you need to install is requirements_train_predict.txt
2. Install the required packages using pip install -r requirements.txt

● Usage 
1. Run python train.py to execute the training script
2. Run python predict.py to execute the predict script
