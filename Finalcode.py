import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, VotingClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('CreditCard.csv')
print(df.head(10))
print(len(df))
print(df.describe())
a = df.select_dtypes('float64').copy()
b = df.select_dtypes('int64').copy()
c = df.select_dtypes('object').copy()
dummy_df = pd.get_dummies(a)
yy = pd.concat([dummy_df, a, b], axis=1)

X = np.array(yy.drop('Fraud', axis=1))
y = np.array(yy['Fraud'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=25)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test)

print("linear kernel")
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)
y_pred1 = classifier.predict(x_test)
xx = accuracy_score(y_test, y_pred1)
print(xx)
print("Recall:",metrics.recall_score(y_test, y_pred1))

print('\n Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))
print('\n Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))
print('\n Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))

print("rbf kernel")
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train, y_train)
y_pred2 = classifier.predict(x_test)
xy = accuracy_score(y_test, y_pred2)
print(xy)

print("poly kernel")
classifier = SVC(kernel='poly', random_state=0)
classifier.fit(x_train, y_train)
y_pred3 = classifier.predict(x_test)
xz = accuracy_score(y_test, y_pred3)
print(xz)

print("sigmoid kernel")
classifier = SVC(kernel='sigmoid', random_state=0)
classifier.fit(x_train, y_train)
y_pred4 = classifier.predict(x_test)
xyz = accuracy_score(y_test, y_pred4)
print(xyz)

print("naive bayes:")
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred5 = gnb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred5))

print("KNN:")
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred6 = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred6))
print("Recall:",metrics.recall_score(y_test, y_pred6))
print('\n Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred6))
print('\n Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred6))
print('\n Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred6)))


print("LOGISTIC REGRESSION:")
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred6))
print('\n Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred6))
print('\n Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred6))
print('\n Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred6)))

print("DECISION TREE:")
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred3 = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred3))
print("Recall:",metrics.recall_score(y_test, y_pred3))

print('\n Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred3))
print('\n Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred3))
print('\n Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred3)))


print("LINEAR REGRESSION:")
linear = LinearRegression().fit(X_train, y_train)
#linear._estimator_type = "classifier"
y_pred2 =np.round(linear.predict(X_test))
# Model Accuracy: how often is the classifier correct?
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Recall:",metrics.recall_score(y_test, y_pred2))
print('\n Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred2))
print('\n Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred2))
print('\n Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))



print("RANDOM FOREST:")
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred4 = np.round(regressor.predict(X_test))
print("Accuracy:", accuracy_score(y_test, y_pred4))
print("Recall:",metrics.recall_score(y_test, y_pred4))
print('\n Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred4))
print('\n Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred4))
print('\n Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred4)))



print("Voting Classifier:")

# Define classifiers
svm_poly = SVC(kernel='poly')
svm_linear = SVC(kernel='linear')
svm_rbf = SVC(kernel='rbf')
svm_sigmoid = SVC(kernel='sigmoid')
gnb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=7)
logreg = LogisticRegression(random_state=1)
clf = DecisionTreeClassifier(max_depth=4)

# Define the voting classifier
voting_clf = VotingClassifier(estimators=[
    ('svm_poly', svm_poly),
    ('svm_linear', svm_linear),
    ('svm_rbf', svm_rbf),
    ('svm_sigmoid', svm_sigmoid),
    ('gnb', gnb),
    ('knn', knn),
    ('logreg', logreg),
    ('clf', clf)
], voting='hard')

# Perform k-fold cross-validation
num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True)
accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the voting classifier on the training data
    voting_clf.fit(X_train, y_train)

    # Predict on the test data
    y_pred = voting_clf.predict(X_test)

    # Calculate accuracy and append to list
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Calculate and print the average accuracy
max_accuracy = np.max(accuracies)
print(accuracies)
print("Maximum Accuracy:", max_accuracy)
