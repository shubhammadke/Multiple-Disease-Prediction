import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pickle
warnings.filterwarnings("ignore")
#%matplotlib inline
import seaborn as sns

data = pd.read_csv(r"C:\Users\shubh\OneDrive\Desktop\DISEASE\csv\breast.csv")

dataset = data
dataset['diagnosis'].replace(['M','B'], [1,0], inplace = True)

dataset.drop('Unnamed: 32',axis = 1, inplace = True)
dataset.drop(['id','symmetry_se','smoothness_se','texture_se','fractal_dimension_mean'], axis = 1, inplace = True)

X = dataset.drop('diagnosis', axis = 1)
y = dataset['diagnosis']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print("Train Set: ", X_train.shape, y_train.shape)
print("Test Set: ", X_test.shape, y_test.shape)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score

confusion_matrix(y_test, model.predict(X_test))

print(f"Accuracy is {round(accuracy_score(y_test, model.predict(X_test))*100,2)}")

from sklearn.model_selection import RandomizedSearchCV

classifier = RandomForestClassifier(n_jobs = -1)

from scipy.stats import randint
param_dist={'max_depth':[3,5,10,None],
              'n_estimators':[10,100,200,300,400,500],
              'max_features':randint(1,27),
               'criterion':['gini','entropy'],
               'bootstrap':[True,False],
               'min_samples_leaf':randint(1,27),
              }

search_clfr = RandomizedSearchCV(classifier, param_distributions = param_dist, n_jobs=-1, n_iter = 40, cv = 9)
search_clfr.fit(X_train, y_train)

params = search_clfr.best_params_
score = search_clfr.best_score_
print(params)
print(score)

claasifier=RandomForestClassifier(n_jobs=-1, n_estimators=200,bootstrap= True,criterion='gini',max_depth=20,max_features=8,min_samples_leaf= 1)
classifier.fit(X_train, y_train)

print(f"Accuracy is {round(accuracy_score(y_test, classifier.predict(X_test))*100,2)}%")

pickle.dump(classifier, open('model1.pkl','wb'))
