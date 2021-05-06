import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
import warnings
import pickle
warnings.filterwarnings("ignore")

#%matplotlib inline
import seaborn as sns

data = pd.read_csv(r"C:\Users\shubh\OneDrive\Desktop\DISEASE\csv\diabetes.csv")

data.head()


X = data.iloc[:,:-1]
y = data['Outcome']

y = y.astype('float64')
X = X.astype('float64')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

print("Train Set: ", X_train.shape, y_train.shape)
print("Test Set: ", X_test.shape, y_test.shape)

from sklearn.ensemble import RandomForestClassifier
diabetes = RandomForestClassifier(n_estimators=20)
diabetes.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, diabetes.predict(X_test))*100)

pickle.dump(diabetes, open('model.pkl','wb'))
