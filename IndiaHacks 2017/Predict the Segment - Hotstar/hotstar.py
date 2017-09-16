# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:17:31 2017

@author: Ketan1
"""
import numpy as np
import pandas as pd
import re
import gc
dataset= pd.read_json('train_data.json',orient="index")
X=dataset.iloc[:, [0,1,2,4,5]].values
X=pd.DataFrame(X)              
y=dataset.iloc[:, 3].values
y=pd.DataFrame(y) 


#Read Test Dataset

X_test= pd.read_json('test_data.json',orient="index")
X_test=pd.DataFrame(X_test)  


del dataset 

gc.collect() 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y=LabelEncoder() 
y=pd.DataFrame(y)   
y=labelencoder_y.fit_transform(y)            


g1= [re.sub(pattern='\:\d+',repl='',string=x) for x in X[2]] 
g1=pd.DataFrame(g1)
g1[0]= g1[0].apply(lambda x: x.split(','))



g2= [re.sub(pattern='\:\d+',repl='',string=x) for x in X[1]] 
g2=pd.DataFrame(g2)
g2[0]= g2[0].apply(lambda x: x.split(','))

g3= [re.sub(pattern='\:\d+',repl='',string=x) for x in X[4]] 
g3=pd.DataFrame(g3)
g3[0]= g3[0].apply(lambda x: x.split(','))

     

g1 = pd.Series(g1[0]).apply(frozenset).to_frame(name='t_genre')
for t_genre in frozenset.union(*g1.t_genre):
    g1[t_genre] = g1.apply(lambda _: int(t_genre in _.t_genre), axis=1)
    

g2 = pd.Series(g2[0]).apply(frozenset).to_frame(name='t_dow')
for t_dow in frozenset.union(*g2.t_dow):
    g2[t_dow] = g2.apply(lambda _: int(t_dow in _.t_dow), axis=1)  
    
    
    


g3 = pd.Series(g3[0]).apply(frozenset).to_frame(name='t_tod')
for t_tod in frozenset.union(*g3.t_tod):
    g3[t_tod] = g3.apply(lambda _: int(t_tod in _.t_tod), axis=1)
    
g3.drop(['t_tod'], inplace=True, axis=1)    

X = pd.concat([X.reset_index(drop=True), g1], axis=1)
X = pd.concat([X.reset_index(drop=True), g2], axis=1)  
X = pd.concat([X.reset_index(drop=True), g3], axis=1)



#test

g1= [re.sub(pattern='\:\d+',repl='',string=x) for x in X_test['genres']] 
g1=pd.DataFrame(g1)
g1[0]= g1[0].apply(lambda x: x.split(','))

g2= [re.sub(pattern='\:\d+',repl='',string=x) for x in X_test['dow']] 
g2=pd.DataFrame(g2)
g2[0]= g2[0].apply(lambda x: x.split(','))

g1 = pd.Series(g1[0]).apply(frozenset).to_frame(name='t_genre')
for t_genre in frozenset.union(*g1.t_genre):
    g1[t_genre] = g1.apply(lambda _: int(t_genre in _.t_genre), axis=1)
    

g2 = pd.Series(g2[0]).apply(frozenset).to_frame(name='t_dow')
for t_dow in frozenset.union(*g2.t_dow):
    g2[t_dow] = g2.apply(lambda _: int(t_dow in _.t_dow), axis=1)  
    
    
    
g3= [re.sub(pattern='\:\d+',repl='',string=x) for x in X_test['tod']] 
g3=pd.DataFrame(g3)
g3[0]= g3[0].apply(lambda x: x.split(','))

g3 = pd.Series(g3[0]).apply(frozenset).to_frame(name='t_tod')
for t_tod in frozenset.union(*g3.t_tod):
    g3[t_tod] = g3.apply(lambda _: int(t_tod in _.t_tod), axis=1)

g3.drop(['t_tod'], inplace=True, axis=1) 
    

X_test = pd.concat([X_test.reset_index(drop=True), g1], axis=1)
X_test = pd.concat([X_test.reset_index(drop=True), g2], axis=1)  
X_test = pd.concat([X_test.reset_index(drop=True), g3], axis=1)

del g1
del g2
gc.collect() 
 
#title    
w1 = X[3]
w1 = w1.str.split(',')    


main = []
for i in np.arange(X.shape[0]):
    d1 = w1[i]
    nest = []
    nest = [re.sub(pattern = '.*\:', repl=' ', string= d1[k]) for k in list(np.arange(len(d1)))]
    main.append(nest)
    
blanks = []
for i in np.arange(len(main)):
    if '' in main[i]:
        print ("{} blanks found".format(len(blanks)))
        blanks.append(i)    
        
        
#replacing blanks with 0
for i in blanks:
    main[i] = [x.replace('','0') for x in main[i]]        
    
    
#converting string to integers
main = [[int(y) for y in x] for x in main]   


tosum = []
for i in np.arange(len(main)):
    s = sum(main[i])
    tosum.append(s) 
    
X['title_sum'] = tosum   
 
#genres 
w1 = X[2]
w1 = w1.str.split(',')    


main = []
for i in np.arange(X.shape[0]):
    d1 = w1[i]
    nest = []
    nest = [re.sub(pattern = '.*\:', repl=' ', string= d1[k]) for k in list(np.arange(len(d1)))]
    main.append(nest)
    
blanks = []
for i in np.arange(len(main)):
    if '' in main[i]:
        print ("{} blanks found".format(len(blanks)))
        blanks.append(i)    
        
        
#replacing blanks with 0
for i in blanks:
    main[i] = [x.replace('','0') for x in main[i]]        
    
    
#converting string to integers
main = [[int(y) for y in x] for x in main]   


tosum = []
for i in np.arange(len(main)):
    s = sum(main[i])
    tosum.append(s) 
    
X['genres_sum'] = tosum  

 
 
 
 
 
 
 
#city
w1 = X[0]
w1 = w1.str.split(',')    


main = []
for i in np.arange(X.shape[0]):
    d1 = w1[i]
    nest = []
    nest = [re.sub(pattern = '.*\:', repl=' ', string= d1[k]) for k in list(np.arange(len(d1)))]
    main.append(nest)
    
blanks = []
for i in np.arange(len(main)):
    if '' in main[i]:
        print ("{} blanks found".format(len(blanks)))
        blanks.append(i)    
        
        
#replacing blanks with 0
for i in blanks:
    main[i] = [x.replace('','0') for x in main[i]]        
    
    
#converting string to integers
main = [[int(y) for y in x] for x in main]   


tosum = []
for i in np.arange(len(main)):
    s = sum(main[i])
    tosum.append(s) 
    
X['city_sum'] = tosum 
 
 
 
 
 
 
 
#dow
w1 = X[1]
w1 = w1.str.split(',')    


main = []
for i in np.arange(X.shape[0]):
    d1 = w1[i]
    nest = []
    nest = [re.sub(pattern = '.*\:', repl=' ', string= d1[k]) for k in list(np.arange(len(d1)))]
    main.append(nest)
    
blanks = []
for i in np.arange(len(main)):
    if '' in main[i]:
        print ("{} blanks found".format(len(blanks)))
        blanks.append(i)    
        
        
#replacing blanks with 0
for i in blanks:
    main[i] = [x.replace('','0') for x in main[i]]        
    
    
#converting string to integers
main = [[int(y) for y in x] for x in main]   


tosum = []
for i in np.arange(len(main)):
    s = sum(main[i])
    tosum.append(s) 
    
X['dow_sum'] = tosum  
 
 
 
 
 
 #tod
w1 = X[4]
w1 = w1.str.split(',')    


main = []
for i in np.arange(X.shape[0]):
    d1 = w1[i]
    nest = []
    nest = [re.sub(pattern = '.*\:', repl=' ', string= d1[k]) for k in list(np.arange(len(d1)))]
    main.append(nest)
    
blanks = []
for i in np.arange(len(main)):
    if '' in main[i]:
        print ("{} blanks found".format(len(blanks)))
        blanks.append(i)    
        
        
#replacing blanks with 0
for i in blanks:
    main[i] = [x.replace('','0') for x in main[i]]        
    
    
#converting string to integers
main = [[int(y) for y in x] for x in main]   


tosum = []
for i in np.arange(len(main)):
    s = sum(main[i])
    tosum.append(s) 
    
X['tod_sum'] = tosum 





 



#making changes in test data
w1_te = X_test['titles']
w1_te = w1_te.str.split(',')




main_te = []
for i in np.arange(X_test.shape[0]):
    d1 = w1_te[i]
    nest = []
    nest = [re.sub(pattern = '.*\:', repl=' ', string= d1[k]) for k in list(np.arange(len(d1)))]
    main_te.append(nest)





blanks_te = []
for i in np.arange(len(main_te)):
    if '' in main_te[i]:
        print ("{} blanks found".format(len(blanks_te)))
        blanks_te.append(i)         
        
#replacing blanks with 0
for i in blanks_te:
    main_te[i] = [x.replace('','0') for x in main_te[i]]
    
#converting string to integers
main_te = [[int(y) for y in x] for x in main_te]

#adding the watch time
tosum_te = []
for i in np.arange(len(main_te)):
    s = sum(main_te[i])
    tosum_te.append(s)      

X_test['title_sum'] = tosum_te   
      

      
      

#genres
w1_te = X_test['genres']
w1_te = w1_te.str.split(',')




main_te = []
for i in np.arange(X_test.shape[0]):
    d1 = w1_te[i]
    nest = []
    nest = [re.sub(pattern = '.*\:', repl=' ', string= d1[k]) for k in list(np.arange(len(d1)))]
    main_te.append(nest)





blanks_te = []
for i in np.arange(len(main_te)):
    if '' in main_te[i]:
        print ("{} blanks found".format(len(blanks_te)))
        blanks_te.append(i)         
        
#replacing blanks with 0
for i in blanks_te:
    main_te[i] = [x.replace('','0') for x in main_te[i]]
    
#converting string to integers
main_te = [[int(y) for y in x] for x in main_te]

#adding the watch time
tosum_te = []
for i in np.arange(len(main_te)):
    s = sum(main_te[i])
    tosum_te.append(s)      





X_test['genres_sum'] = tosum_te      
      
      
      
      
      
      
#cities
w1_te = X_test['cities']
w1_te = w1_te.str.split(',')




main_te = []
for i in np.arange(X_test.shape[0]):
    d1 = w1_te[i]
    nest = []
    nest = [re.sub(pattern = '.*\:', repl=' ', string= d1[k]) for k in list(np.arange(len(d1)))]
    main_te.append(nest)





blanks_te = []
for i in np.arange(len(main_te)):
    if '' in main_te[i]:
        print ("{} blanks found".format(len(blanks_te)))
        blanks_te.append(i)         
        
#replacing blanks with 0
for i in blanks_te:
    main_te[i] = [x.replace('','0') for x in main_te[i]]
    
#converting string to integers
main_te = [[int(y) for y in x] for x in main_te]

#adding the watch time
tosum_te = []
for i in np.arange(len(main_te)):
    s = sum(main_te[i])
    tosum_te.append(s)      





X_test['cities_sum'] = tosum_te               


      
      
      
      
#dow
w1_te = X_test['dow']
w1_te = w1_te.str.split(',')




main_te = []
for i in np.arange(X_test.shape[0]):
    d1 = w1_te[i]
    nest = []
    nest = [re.sub(pattern = '.*\:', repl=' ', string= d1[k]) for k in list(np.arange(len(d1)))]
    main_te.append(nest)





blanks_te = []
for i in np.arange(len(main_te)):
    if '' in main_te[i]:
        print ("{} blanks found".format(len(blanks_te)))
        blanks_te.append(i)         
        
#replacing blanks with 0
for i in blanks_te:
    main_te[i] = [x.replace('','0') for x in main_te[i]]
    
#converting string to integers
main_te = [[int(y) for y in x] for x in main_te]

#adding the watch time
tosum_te = []
for i in np.arange(len(main_te)):
    s = sum(main_te[i])
    tosum_te.append(s)      

X_test['dow_sum'] = tosum_te         



      
      
      

#tod
w1_te = X_test['tod']
w1_te = w1_te.str.split(',')




main_te = []
for i in np.arange(X_test.shape[0]):
    d1 = w1_te[i]
    nest = []
    nest = [re.sub(pattern = '.*\:', repl=' ', string= d1[k]) for k in list(np.arange(len(d1)))]
    main_te.append(nest)





blanks_te = []
for i in np.arange(len(main_te)):
    if '' in main_te[i]:
        print ("{} blanks found".format(len(blanks_te)))
        blanks_te.append(i)         
        
#replacing blanks with 0
for i in blanks_te:
    main_te[i] = [x.replace('','0') for x in main_te[i]]
    
#converting string to integers
main_te = [[int(y) for y in x] for x in main_te]

#adding the watch time
tosum_te = []
for i in np.arange(len(main_te)):
    s = sum(main_te[i])
    tosum_te.append(s)      





X_test['tod_sum'] = tosum_te         
      
      
      
      

      
#count variables      
def wcount(p):
    return p.count(',')+1     


X['title_count'] = X[3].map(wcount)
X['genres_count'] = X[2].map(wcount)
X['cities_count'] = X[0].map(wcount)


X_test['title_count'] = X_test['titles'].map(wcount)
X_test['genres_count'] = X_test['genres'].map(wcount)
X_test['cities_count'] = X_test['cities'].map(wcount)



X.drop([0,1,2,3,4,'t_genre','t_dow'], inplace=True, axis=1)
X_test.drop(['cities','dow','genres','titles','tod','t_genre','t_dow'], inplace=True, axis=1)


from sklearn.ensemble import GradientBoostingRegressor
params = {'n_estimators': 100, 'max_depth': 6,
            'learning_rate': 0.1,'loss': 'huber','alpha':0.95}
rf_model = GradientBoostingRegressor(**params).fit(X, y)

rf_pred = rf_model.predict(X_test)

rf_pred=pd.DataFrame(rf_pred)
y_pred=round(y_pred, 4)
csv_input = pd.read_csv('sample_submission.csv')

csv_input['segment'] = rf_pred
csv_input.to_csv('sample_submission.csv', index=False)