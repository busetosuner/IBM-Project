from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_classification
from sklearn import tree

from sklearn import preprocessing

# KNN Algorithm

def KNN(df, target, k):
    X = df
    y = target

    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=44)

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    print("\nPredict: ", knn.predict(X_test))
    print("\nKNN Score: ", knn.score(X_test, y_test))
