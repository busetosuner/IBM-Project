

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

• Feature_Engineering: Includes modules for binning, handling numerical data, and creating dummy variables.

• Data_Cleaning: Contains functions for handling missing values and duplicate records.
Modeling: Includes modules for regression, classification, and clustering tasks.

• EDA: Contains functions to calculate correlations between attributes.

• User_Interface: Includes functions for preparing and organizing data for modeling.


# Usage
1) Run the main.py file.
2) Provide the file path to your dataset when prompted.
3) Choose the target attribute and the group of attributes you want to work with, if required.
4) The project will then perform the following tasks in sequence:
5) Clean missing values and duplicates.
6) Convert numeric columns to numerical format.
7) Perform data preprocessing steps such as dropping outliers, normalization, and standardization.
8) Handle target attribute encoding if it is non-numeric.
9) Create dummy variables or apply binning as needed for categorical attributes.
10) Perform feature selection.
11) Calculate correlation with the target attribute.
12) Perform multiple linear regression modeling.
13) Perform K-Nearest Neighbors (KNN) classification.
14) Perform decision tree classification.
15) Save the processed dataset to a CSV file.


# Conclusion
This project provides a comprehensive set of tools for working with various datasets and performing data-related tasks. It covers data collection, cleaning, feature engineering, and modeling, allowing you to gain insights and make informed decisions based on the data.

Feel free to modify and customize the code to suit your specific needs and datasets. Happy data analysis and modeling!
