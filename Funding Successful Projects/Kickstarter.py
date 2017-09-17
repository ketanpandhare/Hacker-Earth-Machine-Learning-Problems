# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 03:56:05 2017

@author: Ketan1
"""
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb




dataset1=pd.read_csv('train.csv')

X=dataset1.iloc[:, [8,9,10,11,2]].values
X=pd.DataFrame(X)              
y=dataset1.iloc[:, 13].values
              
y=pd.DataFrame(y)   



dataset2=pd.read_csv('test.csv')
X_test=dataset2.iloc[:, [8,9,10,11,2]].values
X_test=pd.DataFrame(X_test)             



X[0] = pd.to_datetime(X[0],unit='s')
X[1] = pd.to_datetime(X[1],unit='s')
X[2] = pd.to_datetime(X[2],unit='s')
X[3] = pd.to_datetime(X[3],unit='s')



colum_1=pd.DataFrame({"d_year":X[0].dt.year,
                      "d_month":X[0].dt.month,
                      "d_day":X[0].dt.day,
                      "d_hour":X[0].dt.hour,
                      "d_minute":X[0].dt.minute,
                      "d_second":X[0].dt.second,})

colum_2=pd.DataFrame({"s_year":X[1].dt.year,
                      "s_month":X[1].dt.month,
                      "s_day":X[1].dt.day,
                      "s_hour":X[1].dt.hour,
                      "s_minute":X[1].dt.minute,
                      "s_second":X[1].dt.second,})

colum_3=pd.DataFrame({"c_year":X[2].dt.year,
                      "c_month":X[2].dt.month,
                      "c_day":X[2].dt.day,
                      "c_hour":X[2].dt.hour,
                      "c_minute":X[2].dt.minute,
                      "c_second":X[2].dt.second,})

colum_4=pd.DataFrame({"l_year":X[3].dt.year,
                      "l_month":X[3].dt.month,
                      "l_day":X[3].dt.day,
                      "l_hour":X[3].dt.hour,
                      "l_minute":X[3].dt.minute,
                      "l_second":X[3].dt.second,})


X.drop([0,1,2,3], inplace=True, axis=1)
X=X.join(colum_1)
X=X.join(colum_2)
X=X.join(colum_3)
X=X.join(colum_4)





X_test[0] = pd.to_datetime(X_test[0],unit='s')
X_test[1] = pd.to_datetime(X_test[1],unit='s')
X_test[2] = pd.to_datetime(X_test[2],unit='s')
X_test[3] = pd.to_datetime(X_test[3],unit='s')


colum_1=pd.DataFrame({"d_year":X_test[0].dt.year,
                      "d_month":X_test[0].dt.month,
                      "d_day":X_test[0].dt.day,
                      "d_hour":X_test[0].dt.hour,
                      "d_minute":X_test[0].dt.minute,
                      "d_second":X_test[0].dt.second,})

colum_2=pd.DataFrame({"s_year":X_test[1].dt.year,
                      "s_month":X_test[1].dt.month,
                      "s_day":X_test[1].dt.day,
                      "s_hour":X_test[1].dt.hour,
                      "s_minute":X_test[1].dt.minute,
                      "s_second":X_test[1].dt.second,})

colum_3=pd.DataFrame({"c_year":X_test[2].dt.year,
                      "c_month":X_test[2].dt.month,
                      "c_day":X_test[2].dt.day,
                      "c_hour":X_test[2].dt.hour,
                      "c_minute":X_test[2].dt.minute,
                      "c_second":X_test[2].dt.second,})

colum_4=pd.DataFrame({"l_year":X_test[3].dt.year,
                      "l_month":X_test[3].dt.month,
                      "l_day":X_test[3].dt.day,
                      "l_hour":X_test[3].dt.hour,
                      "l_minute":X_test[3].dt.minute,
                      "l_second":X_test[3].dt.second,})


X_test.drop([0,1,2,3], inplace=True, axis=1)
X_test=X_test.join(colum_1)
X_test=X_test.join(colum_2)
X_test=X_test.join(colum_3)
X_test=X_test.join(colum_4)




# creating a full list of descriptions from train and etst
kickdesc = pd.Series(X[4].tolist() + X_test[4].tolist()).astype(str)


# this function cleans punctuations, digits and irregular tabs. Then converts the sentences to lower
def desc_clean(word):
    p1 = re.sub(pattern='(\W+)|(\d+)|(\s+)',repl=' ',string=word)
    p1 = p1.lower()
    return p1

kickdesc = kickdesc.map(desc_clean)




stop = set(stopwords.words('english'))
kickdesc = [[x for x in x.split() if x not in stop] for x in kickdesc]

stemmer = SnowballStemmer(language='english')
kickdesc = [[stemmer.stem(x) for x in x] for x in kickdesc]

kickdesc = [[x for x in x if len(x) > 2] for x in kickdesc]

kickdesc = [' '.join(x) for x in kickdesc]


# Due to memory error, limited the number of features to 650
cv = CountVectorizer(max_features=650)



alldesc = cv.fit_transform(kickdesc).todense()



#create a data frame
combine = pd.DataFrame(alldesc)
combine.rename(columns= lambda x: 'variable_'+ str(x), inplace=True)


#split the text features

train_text = combine[:X.shape[0]]
test_text = combine[X.shape[0]:]

test_text.reset_index(drop=True,inplace=True)



X['goal']=dataset1['goal']
X_test['goal']=dataset2['goal']



X.drop(4,axis=1, inplace=True)

X_test.drop(4,axis=1, inplace=True)


X = pd.concat([X, train_text],axis=1)
X_test = pd.concat([X_test, test_text],axis=1)




cols_to_use = ['name','desc']
len_feats = ['name_len','desc_len']
count_feats = ['name_count','desc_count']

for i in np.arange(2):
    X[len_feats[i]] = dataset1[cols_to_use[i]].apply(str).apply(len)
    X[count_feats[i]] = dataset1[cols_to_use[i]].apply(str).apply(lambda x: len(x.split(' ')))


X['keywords_len'] = dataset1['keywords'].apply(str).apply(len)
X['keywords_count'] = dataset1['keywords'].apply(str).apply(lambda x: len(x.split('-')))



for i in np.arange(2):
    X_test[len_feats[i]] = dataset2[cols_to_use[i]].apply(str).apply(len)
    X_test[count_feats[i]] = dataset2[cols_to_use[i]].apply(str).apply(lambda x: len(x.split(' ')))


X_test['keywords_len'] = dataset2['keywords'].apply(str).apply(len)
X_test['keywords_count'] = dataset2['keywords'].apply(str).apply(lambda x: len(x.split('-')))



X['goal'] = np.log1p(X['goal'])
X_test['goal'] = np.log1p(X_test['goal'])



from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X,y)
y_pred=classifier.predict(X_test)

y_pred=pd.DataFrame(y_pred)
#y_pred=round(y_pred, 1)
csv_input = pd.read_csv('samplesubmission.csv')
csv_input['final_status'] = y_pred
csv_input.to_csv('samplesubmission.csv', index=False)

