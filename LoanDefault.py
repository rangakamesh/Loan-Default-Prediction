import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score,mean_squared_error, roc_curve, auc
import time
from numpy import interp
from sklearn.model_selection import StratifiedKFold
import xgboost
import CleanData


data = pd.read_pickle('dataset/train_dataset.pkl')

train_features, train_target = CleanData.mk_data(data,10000,0,10000,'train')

data = pd.read_pickle('dataset/test_dataset.pkl')

val_features, val_target = CleanData.mk_data(data,500,10000,10500,'train')
test_features = CleanData.mk_data(data,5000,10000,15000,'test')



clf = xgboost.XGBClassifier(n_estimators=160,eval_metric="error",use_label_encoder=False)

tprs = np.zeros([10,100])
aucs = np.zeros([10])
mean_fpr = np.linspace(0, 1, 100)
cv = StratifiedKFold(n_splits=10)

i=0
for train, test in cv.split(train_features, train_target):
    probas_ = clf.fit(train_features[train], train_target[train]).predict_proba(train_features[test])
    fpr, tpr, thresholds = roc_curve(train_target[test], probas_[:, 1])
    tprs[i,:] = interp(mean_fpr, fpr, tpr)
    
    roc_auc = auc(fpr, tpr)
    aucs[i] = roc_auc
    i+=1

graph =  plt.plot(mean_fpr,np.mean(tprs,axis=0))
graphx = plt.plot([0,1],[0,1])
grapht = plt.title('Mean ROC Curve : '+str(np.mean(aucs)))

plt.show()

clf.fit(train_features,train_target)
ypred=clf.predict(val_features)

print('Validation Accuracy of the mdoel: ',accuracy_score(val_target,ypred))

probas_=clf.predict_proba(val_features)
fpr, tpr, thresholds = roc_curve(val_target, probas_[:, 1])
roc_auc = auc(fpr, tpr)
graph =  plt.plot(fpr,tpr)
graphx = plt.plot([0,1],[0,1])
grapht = plt.title('Validation ROC : '+str(roc_auc))


plt.show()


probas_=clf.predict_proba(test_features)

f = open("output.txt", "w")

i=0
for cust in np.arange(10000,15000):
  o_id=str(data[0][cust][1])
  o_prob=str(probas_[i][1])
  o_str=o_id+','+o_prob+'\n'
  f.write(o_str)
  i+=1
f.close()


print('Model Score : '+str(clf.score(train_features,train_target)))

