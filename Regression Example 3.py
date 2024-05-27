# 1 import bibliotek

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sn
import matplotlib.pyplot as plt

# 2 załadowanie danych i wyświetlenie 5 wierszy oraz informacji o danych

dane = pd.read_csv('users.csv', sep=';',decimal=',')

print(dane.head())
print(dane.info())

# 3 usunięcie kolumny User ID

del dane['User ID']

# 4 konwersja wartości estimated salary na wartość liczbową

def dollars(val:str) -> float:
    ret = val.replace('$', '')
    ret = ret.replace(' ', '')
    ret = ret.replace(',', '.')
    return float(ret)

dane['EstimatedSalary'] = dane['EstimatedSalary'].map(dollars)

print(dane.head())

# 5 wykresy 
w = dane.groupby('Gender').agg(es = ('EstimatedSalary','mean'))
w = w.reset_index()
plt.pie(w['es'],labels=w['Gender'])
plt.show()

w2 = dane.groupby('Age').agg(es = ('EstimatedSalary','mean'))
w2 = w2.reset_index()
plt.bar(w2['Age'],w2['es'])
plt.show()

# 6 zamiana kolumny Gender na wartości liczbowe

dane['Gender'] = dane['Gender'].map(lambda x: 0 if x == "Male" else 1)
print(dane.head())

# 7 zamiana kolumny Purchased na wartości liczbowe

dane['Purchased'] = dane['Purchased'].map(lambda x: 1 if x == "Yes" else 0)
print(dane.head())

# 8 wybór zmiennych zależnych i niezależnych

X = np.array(dane[["Gender","Age","EstimatedSalary"]])
y = np.array(dane["Purchased"])

# 9 podział danych na zbiór treningowy i testowy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 10 regresja logistyczna

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 11 predykcja

y_pred = model.predict(X_test)

# 12 ocena modelu

print(model.score(X_train,y_train))
print(model.score(X_test,y_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
sn.heatmap(confusion_matrix(y_test, y_pred),annot=True)
plt.show()

#13 Odpowiedź

d = np.single([1,18,240000])
d = d.reshape(1,-1)
print(model.predict(d))

d = np.single([0,28,123000])
d = d.reshape(1,-1)
print(model.predict(d))


d = np.single([0,43,234000])
d = d.reshape(1,-1)
print(model.predict(d))