

# IBM DATA SCIENCE PROJECT 


# Introduction
Welcome to our Data Science - Scrougify project!

This project is designed to work with various datasets and perform data-related tasks such as data collection, cleaning, feature engineering, modeling, and interpretation. It utilizes several Python libraries to achieve these tasks and is organized into separate modules for different stages of the data science pipeline.

# Getting Started
To use this project, you need to have Python installed on your system along with the required libraries. The main file to execute is main.py, which orchestrates the data processing workflow. Before running the code, ensure you have your dataset in a compatible format, and the file path will be given as input during execution.

$ python main.py

# Libraries 
The following Python libraries are required to run the project:

• pandas
• sklearn
• numpy
• os
• sys
• tkinter
• scipy
• seaborn
• matplotlib
• statsmodels.api
• time

You can install them using pip, such as:
$ pip install pandas scikit-learn


# Project Structure
The project is structured as follows:

• Data_Collection: Contains functions to import files and check data format.

• Data_Cleaning: Contains functions for handling missing values and duplicate records.

• Feature_Engineering: Includes modules for binning, handling numerical data, and creating dummy variables.

• EDA: Contains functions to calculate correlations between attributes.

• Modeling: Contains functions for Regression, Classification and Clustering. Visualise the outcomes using Python libraries.

• Evaluation: Evaluates modeling using over and underfitting functions. 

• User_Interface: Includes functions for preparing and organizing data for modeling.


# Usage
1) Run the main.py file.
2) Provide the file path to your dataset when prompted.
3) Choose the target attribute and the group of attributes you want to work with, if required.
4) The project will then perform the following tasks in sequence:
   
  -Clean missing values and duplicates.
   
  -Convert numeric columns to numerical format.
  
  -Perform data preprocessing steps such as dropping outliers, normalization, and standardization.
  
  -Handle target attribute encoding if it is non-numeric.
  
  -Create dummy variables or apply binning as needed for categorical attributes.
  
  -Perform feature selection.
  
  -Calculate correlation with the target attribute.
  
  -Perform multiple linear regression modeling.
  
  -Perform K-Nearest Neighbors (KNN) classification.
  
  -Perform decision tree classification and clustering.
  
  -Save the processed dataset to a CSV file.


# Conclusion
This project provides a comprehensive set of tools for working with various datasets and performing data-related tasks. It covers data collection, cleaning, feature engineering, and modeling, allowing you to gain insights and make informed decisions based on the data.

Feel free to modify and customize the code to suit your specific needs and datasets. Happy data analysis and modeling!
