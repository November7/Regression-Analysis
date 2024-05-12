import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dane = pd.read_csv('weatherAUS.csv')

dane["LocationID"] = pd.Categorical(dane["Location"]).codes

'''
0 — wiatr północny (N)
22.5 — wiatr północno-wschodni (NNE)
45 — wiatr północno-wschodni (NE)
67.5 — wiatr wschodnio-północno-wschodni (ENE)
90 — wiatr wschodni (E)
112.5 — wiatr południowo-wschodni (ESE)
135 — wiatr południowo-wschodni (SE)
157.5 — wiatr południowo-południowo-wschodni (SSE)
180 — wiatr południowy (S)
202.5 — wiatr południowo-południowo-zachodni (SSW)
225 — wiatr południowo-zachodni (SW)
247.5 — wiatr południowo-zachodni (WSW)
270 — wiatr zachodni (W)
292.5 — wiatr zachodnio-północno-zachodni (WNW)
315 — wiatr północno-zachodni (NW)
337.5 — wiatr północno-północno-zachodni (NNW)
360 — wiatr północny (N)
'''
def WindDir(angle):
    
    angles = {
        'N': 0,
        'NNE':22.5,
        'NE':45, 
        'ENE':67.5, 
        'E':90, 
        'ESE':112.5, 
        'SE':135, 
        'SSE':157.5, 
        'S':180, 
        'SSW':202.5,
        'SW':225,
        'WSW':247.5,
        'W':270,
        'WNW':292.5,
        'NW':315,
        'NNW':337.5
    }
    try:
        return angles[angle]
    except:
        return None

def YesNo(value:str):
    if value == "Yes":
        return 1
    else:
        return 0

dane['WindGustDir'] = dane['WindGustDir'].map(WindDir)
dane['RainToday'] = dane['RainToday'].map(YesNo)
dane['RainTomorrow'] = dane['RainTomorrow'].map(YesNo)

test = dane[['MinTemp', 'MaxTemp']][::100]

# import seaborn as sns
# from matplotlib import pyplot as plt

# sns.pairplot(test,diag_kind="scatter")

# plt.show()

print(dane.head())
print(dane.info())


