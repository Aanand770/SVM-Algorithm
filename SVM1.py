import pandas as pd
from pandas import read_csv, DataFrame
import sklearn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
import numpy as np

# importing datasets
soil = r'/home/wall-e/diabetes.csv'
data = read_csv(soil)

df = pd.DataFrame(data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome'])

# Extracting Independent and dependent Variable
X = df.iloc[:, [1, 8]].values
y = df.Outcome


# Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Fitting the SVM classifier to the training set
classifier = SVC(kernel='linear', random_state=0)
classifier = classifier.fit(X_train, y_train)
print(classifier)
pred = classifier.predict(X_test)
print('Prediction = ', pred)
b = classifier.score(X_test, y_test)
print('Accuracy = ', b)

#Displying Confusion Matrix and Classification Report
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix

result = confusion_matrix(y_test, pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, pred)
print("Classification Report:",)
print (result1)
print("Mean Squared Error:",)
print(result2)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]

Svc_classifier = svm.SVC(kernel='linear', C=1.0).fit(X, y)

Z = Svc_classifier.predict(X_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlim(xx.min(), xx.max())

plt.show()


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test, display_labels=y, cmap=plt.cm.Blues, normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

w=classifier.coef_[0]
print(w)

a= -w[0]/w[1]

xx= np.linspace(0,12)
yy= a* xx - classifier.intercept_[0]/ w[1]

h0= plt.plot(xx, yy, 'k-')

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

