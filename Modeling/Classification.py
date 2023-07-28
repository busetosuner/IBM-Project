from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,     plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# KNN Algorithm
def KNN(df, target, k):
    X = df
    y = df[target]
    
    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)

    neighbors = np.arange(1, 9)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    for i, k in enumerate(neighbors):
        knn =     knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)

    plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')

    plt.legend()
    plt.xlabel('n_neigbors')
    plt.ylabel('Accuracy')
    plt.show()

    knn = KNeighborsClassifier(n_neighbors=pd.Series(test_accuracy).idxmax() + 1)
    knn.fit(X_train, y_train)

    # print("\nPredict (KNN): ", knn.predict(X_test))
    print("\nKNN Score: ", knn.score(X_test, y_test))
# Decision Trees Algorithm
def decision_trees(df, target):
    X = df
    y = df[target]

    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=44)

    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)

    # print("\nPredict (Decision Trees): ", dtree.predict(X_test))
    print("Decision Trees Score: ", dtree.score(X_test, y_test))

    return dtree.score(X_test, y_test)
