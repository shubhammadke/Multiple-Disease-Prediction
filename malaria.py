import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics

import warnings
import pickle
warnings.filterwarnings("ignore")

##Step1: Load Dataset

dataframe = pd.read_csv(r"C:\Users\shubh\OneDrive\Desktop\DISEASE\csv\dataset.csv")
#print(dataframe.head())

#Step2: Split into training and test data
x = dataframe.drop(["Label"],axis=1)
y = dataframe["Label"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

##Step4: Build a model

model = RandomForestClassifier(n_estimators=100,max_depth=5)
model.fit(x_train,y_train)

from sklearn.metrics import confusion_matrix, accuracy_score

confusion_matrix(y_test, model.predict(x_test))

predictions = model.predict(x_test)

print(metrics.classification_report(predictions,y_test))
print(model.score(x_test,y_test))

pickle.dump(model, open('model4.pkl','wb'))