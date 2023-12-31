import pandas as pd
import warnings
from sklearn.preprocessing import OrdinalEncoder
warnings.filterwarnings("ignore")

import Data_Collection.Import_File as Import_File
import Feature_Engineering.Binning as Binning
import Feature_Engineering.Numerical_Data as Numerical_Data
import Data_Cleaning.Duplicates as Duplicates
import Modeling.Regression as Regression
import EDA.Correlation as Correlation
import EDA.Anova as Anova


import Data_Cleaning.Handle_Missing_Values as Handle_Missing_Values
import Feature_Engineering.dummy_variables as Dummy_Variables
import Modeling.Classification as Classification
import Modeling.Clustering as Clustering
import Feature_Engineering.feature_selection as Feature_Selection
import User_Interface.User_Interface as User_Interface
import Evaluation.Over_UnderFitting as Over_UnderFitting

# Import UI functions from separate files


# Path will be given by the user

warnings.filterwarnings("ignore")

# Import the necessary modules and functions (your import statements)

# Path will be given by the user
df, file_path = Import_File.check_data_format()


# Get input from the user for the target attribute and the group list
headers = df.columns.values

for i, header in enumerate(headers):
    print(i, " ", header)

target = headers[int(input("Please enter index of target attribute: "))]
group_list_input = input("Please enter the index of attributes you want(leave blank for all): ").strip()
group_list = [headers[int(item)] for item in group_list_input.split()] if group_list_input else []



User_Interface.prepare_data(df, target)


if len(group_list) != 0:
    if not target in group_list:
        group_list.append(target)
    df = df[group_list]
elif "Unnamed: 0" in df.columns.values:
    df = df.drop("Unnamed: 0", axis=1)


# Clean missing values and duplicates
df = Handle_Missing_Values.clean_missing(df, target)
df = Duplicates.clean_duplicates(df)


# Convert numeric columns to numeric in pandas for further operations
for column in df.columns.values:
    if not Binning.is_numeric(df[column]):
        continue
    df[column] = pd.to_numeric(df[column])

# Data preprocessing steps
df = Numerical_Data.drop_outliers(df)
df = Numerical_Data.normalization(df)
df = Numerical_Data.standardization(df)

for attribute in df.columns.values:
    if attribute == target:
        if not Binning.is_numeric(df[target]):
            encoder = OrdinalEncoder()
            df[target] = encoder.fit_transform(df[[target]])
        continue

    if df[attribute].nunique() <= 5:
        # Pass numerical variables for the sake of simplicity
        if Binning.is_numeric(df[attribute]):
            continue
        df = dummy_variables.create_dummies(df, attribute)
    else:
        df = Binning.make_bins(df, attribute)

print("\n New columns: ", df.columns.values)

df = feature_selection.decide(df, target)

df_numeric = df[Numerical_Data.numeric_columns(df)]

target_correlation = Correlation.calculate_correlation(df, target)

Anova.perform_anova(df, target)

model, mse, r2, df_numeric = Regression.perform_multiple_linear_regression(df_numeric, target)

if len(df_numeric.axes[1]) < 20:
    Classification.KNN(df_numeric, target, 3)
else:
    print("Sorry, this is too much for KNN classification :(")

Classification.decision_trees(df, target)

Clustering.cluster(df_numeric, 3)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_numeric, df[target], test_size=0.2, random_state=42)


# Evaluate overfitting for the regression model
regression_model = model
mse_train, mse_test = Over_UnderFitting.evaluate_overfitting(regression_model, X_train, y_train, X_test, y_test)


if int(mse_train) == 0.00 and int(mse_test) == 0.00:
    print("Regression Model indicates overfitting.")
else:
    print("Regression Model has good performance.")
    print(f"Regression Model - MSE for Training: {mse_train:.2f}, MSE for Test: {mse_test:.2f}")

df.to_csv(file_path, index=False)
