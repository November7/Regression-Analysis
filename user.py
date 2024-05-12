import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dane = pd.read_csv('User_Data.csv')

print(dane.head())
print(dane.info())

dane['Gender'] = dane['Gender'].map(lambda x: 1 if x == "Male" else 0)

print(dane.head())

X = np.array(dane[["Gender","Age","EstimatedSalary"]])
y = np.array(dane["Purchased"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(model.score(X_train,y_train))
print(model.score(X_test,y_test))
print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# import joblib
# joblib.dump(model, "model.joblib")


import seaborn as sn
import matplotlib.pyplot as plt


sn.heatmap(confusion_matrix(y_test, y_pred),annot=True)
           
plt.show()

