import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import Numerical_Data


def perform_multiple_linear_regression(df, target_col, group_list, mp):
    # Convert the target variable to numeric if it is categorical
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        df[target_col] = pd.factorize(df[target_col])[0]
    
    # Use the dictionary "mp" to get the correct column names
    feature_cols = [mp[col] if col in mp else col for col in group_list]

    # Select only numeric columns for the feature variables

    numeric_columns = Numerical_Data.numeric_columns(df)
    
    # Split the data into features (X) and target (y)
    X = df[numeric_columns]
    y = df[target_col]

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict the target variable on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    
    
    return model, mse, r2, df
