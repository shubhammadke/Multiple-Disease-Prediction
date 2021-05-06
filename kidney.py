import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline


import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv(r"C:\Users\shubh\OneDrive\Desktop\DISEASE\csv\kidney_disease.csv")
dataset = data.copy()
dataset.head()

data.classification.unique()
data.classification=data.classification.replace("ckd\t","ckd")
data.classification.unique()

data.drop('id', axis = 1, inplace = True)

data['classification'] = data['classification'].replace(['ckd','notckd'], [1,0])

df = data.dropna(axis = 0)
print(f"Before dropping all NaN values: {data.shape}")
print(f"After dropping all NaN values: {df.shape}")

df.index = range(0,len(df),1)
df.head()

for i in df['wc']:
    print(i)

df['wc']=df['wc'].replace(["\t6200","\t8400"],[6200,8400])

for i in df['wc']:
    print(i)

df.info()


df['pcv']=df['pcv'].astype(int)
df['wc']=df['wc'].astype(int)
df['rc']=df['rc'].astype(float)
df.info()

object_dtypes = df.select_dtypes(include = 'object')
object_dtypes.head()

dictonary = {
        "rbc": {
        "abnormal":1,
        "normal": 0,
    },
        "pc":{
        "abnormal":1,
        "normal": 0,
    },
        "pcc":{
        "present":1,
        "notpresent":0,
    },
        "ba":{
        "notpresent":0,
        "present": 1,
    },
        "htn":{
        "yes":1,
        "no": 0,
    },
        "dm":{
        "yes":1,
        "no":0,
    },
        "cad":{
        "yes":1,
        "no": 0,
    },
        "appet":{
        "good":1,
        "poor": 0,
    },
        "pe":{
        "yes":1,
        "no":0,
    },
        "ane":{
        "yes":1,
        "no":0,
    }
}

df=df.replace(dictonary)

X = df.drop(['classification', 'sg', 'appet', 'rc', 'pcv', 'hemo', 'sod'], axis = 1)
y = df['classification']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 20)
model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score

confusion_matrix(y_test, model.predict(X_test))

print(f"Accuracy is {round(accuracy_score(y_test, model.predict(X_test))*100-2, 2)}%")
pickle.dump(model, open('model3.pkl','wb'))