from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import preprocessing

# KNN Algorithm
def KNN(df, target, k):
    X = df
    y = df[target]

    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=44)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # print("\nPredict (KNN): ", knn.predict(X_test))
    print("KNN Score: ", knn.score(X_test, y_test))

    return knn.score(X_test, y_test)

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
