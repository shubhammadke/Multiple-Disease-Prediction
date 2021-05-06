import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv(r"C:\Users\shubh\OneDrive\Desktop\DISEASE\csv\heart.csv")
dataset = data.copy()
dataset.head()

X = dataset.drop(['target'], axis = 1)
y = dataset['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)

pred = model.predict(X_test)
pred[:10]

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)

from sklearn.metrics import accuracy_score
print(f"Accuracy of model is {round(accuracy_score(y_test, pred)*100, 2)}%")

pickle.dump(model, open('model2.pkl','wb'))

