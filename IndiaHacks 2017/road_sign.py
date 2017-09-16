# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:30:43 2017

@author: Ketan1
"""

#Importing Train Dataset
import pandas as pd
dataset=pd.read_csv('train.csv')
X=dataset.iloc[:, [1,2,3,4,5]].values
y=dataset.iloc[:, 6].values              
  
#importing Test Dataset              
dataset=pd.read_csv('test.csv')
X_test=dataset.iloc[:, [1,2,3,4,5]].values              
 
#encoding tranning data to integer                   
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()    
X[:, 0]=labelencoder_X.fit_transform(X[:, 0])
onehotencoder=OneHotEncoder(categorical_features= [0])          
X=onehotencoder.fit_transform(X).toarray()

#encoding test data to integer
labelencoder_X2=LabelEncoder()
X_test[:, 0]=labelencoder_X2.fit_transform(X_test[:, 0])
onehotencoder=OneHotEncoder(categorical_features= [0])          
X_test=onehotencoder.fit_transform(X_test).toarray()

#encoding target to integer
labelencoder_y=LabelEncoder() 
y=pd.DataFrame(y)   
y=labelencoder_y.fit_transform(y)



#train Machine Learning model using gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
params = {'n_estimators': 100, 'max_depth': 6,
        'learning_rate': 0.1}
clf = GradientBoostingClassifier(**params).fit(X, y)

#predict on test data
y_pred = clf.predict_proba(X_test)

#Save prediction to sample_submission file
y_pred=pd.DataFrame(y_pred)
y_pred=round(y_pred, 4)
csv_input = pd.read_csv('sample_submission.csv')
csv_input['Front'] = y_pred[0]
csv_input['Left'] = y_pred[1]
csv_input['Rear'] = y_pred[2]
csv_input['Right'] = y_pred[3]
csv_input.to_csv('sample_submission.csv', index=False)
