import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import Numerical_Data
import Binning
import matplotlib.pyplot as plt
import seaborn as sns

def perform_multiple_linear_regression(df, target_col):
    # Convert the target variable to numeric if it is categorical
    if not Binning.is_numeric(df[target_col]):
        df[target_col] = pd.factorize(target_col)[0]
    

    # Split the data into features (X) and target (y)
    X = df.drop(target_col, axis= 1)
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
    # Print regression results
    print("\nMultiple Linear Regression Results:")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)

    # Plot regression results
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Actual Target Values')
    plt.ylabel('Predicted Values')
    plt.title('Regression Plot')
    plt.show()
    
    return model, mse, r2, df
