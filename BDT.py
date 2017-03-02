from sklearn import tree

import pandas
import numpy
import math

dataframe_train = pandas.read_csv("training.csv", header=None)
dataset_train = dataframe_train.values

dataframe_test = pandas.read_csv("test.csv", header=None)
dataset_test = dataframe_test.values

for i in range(250001):
    if dataset_train [i,32]=='s':
        dataset_train [i,32]='1'
    else:
        dataset_train [i,32]='0'
        
print dataset_train.shape
print dataset_test.shape
X_test = dataset_train[1:10001,:31].astype(float)
Y_test = dataset_train[1:10001,32].astype(float)

X_train = dataset_train[10001:200001,:31].astype(float)
Y_train = dataset_train[10001:200001,32].astype(float)

W = dataset_train[1:10001,31].astype(float)


reg = tree.DecisionTreeRegressor ()
reg = reg.fit(X_train,Y_train)
R = reg.predict(X_test)
reg.score(X_test,Y_test)
cr=0.
sr=0
br=0
for i in range(len(X_test)):
    if R[i]== Y_test[i]:
        cr+=1
    if R[i]==1:
        if Y_test[i]==1:
            sr= sr + W[i]
        if Y_test[i]==0:
            br= br + W[i]
score_R=cr*100/len(X_test) 
radicandR = 2 *( (sr+br+10) * math.log (1.0 + sr/(br+10)) -sr)
AMS_R = math.sqrt(radicandR)
print score_R
print 'AMS_R =',AMS_R
#71.07
#AMS_R = 0.273223317766

from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor(base_estimator=reg,
                         n_estimators=50, 
                         learning_rate=0.1, random_state=0)
ada = ada.fit(X_train, Y_train)
A=ada.predict(X_test)
ca=0.
sa=0
ba=0
for i in range(len(X_test)):
    if A[i]== Y_test[i]:
        ca+=1
    if A[i]==1:
        if Y_test[i]==1:
            sa= sa + W[i]
        if Y_test[i]==0:
            ba= ba + W[i]
score_A=ca*100/len(X_test) 
radicandA = 2 *( (sa+ba+10) * math.log (1.0 + sa/(ba+10)) -sa)
AMS_A = math.sqrt(radicandA)
print score_A
print 'AMS_A =',AMS_A
#83 with 100 estimators
#82 with 50 estimators, AMS_A=0.505636131249
#79 with 10 estimators

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators = 50,random_state=0)
forest = forest.fit(X_train,Y_train)
F=forest.predict(X_test)
cf=0
for i in range(len(X_test)):
    if F[i]== Y_test[i]:
        cf+=1
score_F=cf*100/len(X_test)
print score_F
                         
clf = tree.DecisionTreeClassifier ()
clf = clf.fit(X_train,Y_train)

C = clf.predict(X_test)
cc=0.
sc=0
bc=0
for i in range(len(X_test)):
    if C[i]== Y_test[i]:
        cc+=1
    if C[i]==1:
        if Y_test[i]==1:
            sc= sc + W[i]
        if Y_test[i]==0:
            bc= bc + W[i]
score_C=cc*100/len(X_test) 
radicandC = 2 *( (sc+bc+10) * math.log (1.0 + sc/(bc+10)) -sc)
AMS_C = math.sqrt(radicandC)
print score_C
print 'AMS_C =',AMS_C
#70.65
#AMS_C = 0.272780740678

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 50,random_state=0)
forest = forest.fit(X_train,Y_train)
F=forest.predict(X_test)

cf=0.
sf=0
bf=0
for i in range(len(X_test)):
    if F[i]== Y_test[i]:
        cf+=1
    if F[i]==1:
        if Y_test[i]==1:
            sf= sf + W[i]
        if Y_test[i]==0:
            bf= bf + W[i]
score_F=cf*100/len(X_test) 
radicandF = 2 *( (sf+bf+10) * math.log (1.0 + sf/(bf+10)) -sf)
AMS_F = math.sqrt(radicandF)
print score_F
print 'AMS_F =',AMS_F
#83.2
#AMS_F = 0.536340014348

from sklearn.ensemble import GradientBoostingClassifier

gradient = GradientBoostingClassifier(n_estimators=50, random_state=0)
gradient = gradient.fit(X_train,Y_train)
G=gradient.predict(X_test)

cg=0.
sg=0
bg=0
for i in range(len(X_test)):
    if G[i]== Y_test[i]:
        cg+=1
    if G[i]==1:
        if Y_test[i]==1:
            sg= sg + W[i]
        if Y_test[i]==0:
            bg= bg + W[i]
score_G=cg*100/len(X_test) 
radicandG = 2 *( (sg+bg+10) * math.log (1.0 + sg/(bg+10)) -sg)
AMS_G = math.sqrt(radicandG)
print score_G
print 'AMS_G =',AMS_G
#82.33
#AMS_G = 0.51961354779

#En prenant comme ensemble de test le dataset_test, afin de comparer les résultats à ceux sur Kaggle
T=gradient.predict_proba(X_test)[:,0]
print (T)
print (T.shape)
print (Y_test[1])
temp = T.argsort()
ranks = numpy.empty(len(T), int)
ranks[temp] = numpy.arange(len(T)) +1

S=[]
for i in range(len(T)):
    if T[i]>0.5:
        S.append('s')
    else:
        S.append('b')
x=1
res = [["EventId","RankOrder","Class"]]
for i in range(len(G)):
    res.append([int(X_test[i,0]),ranks[i],S[i]])
    
print len(res)       
import csv
c= csv.writer(open("result.csv","wb"))

for i in res:
    c.writerow(i)
#Score AMS donné par Kaggle : 0.39744
