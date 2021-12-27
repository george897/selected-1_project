#!/usr/bin/env python
# coding: utf-8

# In[38]:


#porting Important Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,accuracy_score,auc,hinge_loss,roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,learning_curve

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D,MaxPooling2D, Flatten


# In[39]:


df= pd.read_csv('Diabetes.csv')
df.sample(5)


# In[40]:


title_mapping = {'YES':1,'NO':0}
df[' Class variable']=df[' Class variable'].map(title_mapping)


# In[41]:


df.dtypes


# In[42]:


col=['n_pregnant','glucose_conc','bp','skin_len','insulin','bmi','pedigree_fun','age','Output']
df.columns=col


# In[43]:


diabetes_true_count = len(df.loc[df['Output'] == True])
diabetes_false_count = len(df.loc[df['Output'] == False])


# In[44]:


col=['glucose_conc','bp','insulin','bmi','skin_len']
for i in col:
    df[i].replace(0, np.nan, inplace= True)


# In[45]:


def median_target(var):
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Output']].groupby(['Output'])[[var]].median().reset_index()
    return temp


# In[46]:


median_target('insulin')
median_target('glucose_conc')
#median_target('skin_len')
median_target('bp')
median_target('bmi')


# In[47]:


#Filling the NaN value with Median according to Output
df.loc[(df['Output'] == 0 ) & (df['insulin'].isnull()), 'insulin'] = 102.5
df.loc[(df['Output'] == 1 ) & (df['insulin'].isnull()), 'insulin'] = 169.5
df.loc[(df['Output'] == 0 ) & (df['glucose_conc'].isnull()), 'glucose_conc'] = 107
df.loc[(df['Output'] == 1 ) & (df['glucose_conc'].isnull()), 'glucose_conc'] = 140
df.loc[(df['Output'] == 0 ) & (df['skin_len'].isnull()), 'skin_len'] = 27
df.loc[(df['Output'] == 1 ) & (df['skin_len'].isnull()), 'skin_len'] = 32
df.loc[(df['Output'] == 0 ) & (df['bp'].isnull()), 'bp'] = 70
df.loc[(df['Output'] == 1 ) & (df['bp'].isnull()), 'bp'] = 74.5
df.loc[(df['Output'] == 0 ) & (df['bmi'].isnull()), 'bmi'] = 30.1
df.loc[(df['Output'] == 1 ) & (df['bmi'].isnull()), 'bmi'] = 34.3


# In[48]:


median_target('n_pregnant')
df.loc[(df['Output'] == 0 ) & (df['n_pregnant']>13), 'n_pregnant'] = 2
df.loc[(df['Output'] == 1 ) & (df['n_pregnant']>13), 'n_pregnant'] = 4


# In[49]:


median_target('bp')
df.loc[(df['Output'] == 0 ) & (df['bp']<40), 'bp'] = 70
df.loc[(df['Output'] == 1 ) & (df['bp']<40), 'bp'] = 74.5
df.loc[(df['Output'] == 0 ) & (df['bp']>103), 'bp'] = 70
df.loc[(df['Output'] == 1 ) & (df['bp']>103), 'bp'] = 74.5


# In[50]:


median_target('skin_len')
df.loc[(df['Output'] == 0 ) & (df['skin_len']>38), 'skin_len'] = 27
df.loc[(df['Output'] == 1 ) & (df['skin_len']>38), 'skin_len'] = 32
df.loc[(df['Output'] == 0 ) & (df['skin_len']<20), 'skin_len'] = 27
df.loc[(df['Output'] == 1 ) & (df['skin_len']<20), 'skin_len'] = 32


# In[51]:


median_target('bmi')
df.loc[(df['Output'] == 0 ) & (df['bmi']>48), 'bmi'] = 30.1
df.loc[(df['Output'] == 1 ) & (df['bmi']>48), 'bmi'] = 34.3


# In[52]:


median_target('pedigree_fun')
df.loc[(df['Output'] == 0 ) & (df['pedigree_fun']>1), 'pedigree_fun'] = 0.336
df.loc[(df['Output'] == 1 ) & (df['pedigree_fun']>1), 'pedigree_fun'] = 0.449


# In[53]:


median_target('age')
df.loc[(df['Output'] == 0 ) & (df['age']>61), 'age'] = 27
df.loc[(df['Output'] == 1 ) & (df['age']>61), 'age'] = 36


# In[54]:


df.dtypes


# In[55]:


df.head()


# In[56]:


df['Output'].unique()


# In[57]:


#data visulation
plt.plot(df['bp'],'o')
plt.plot(df['bmi'],'o')
plt.title('bp vs bmi')
plt.legend(['bp','bmi'], loc='upper left')
plt.show()


# In[58]:


scaler = MinMaxScaler()
df.col=['n_pregnant','glucose_conc','bp','skin_len','insulin','bmi','pedigree_fun','age','Output']
df[df.col]= scaler.fit_transform(df[df.col])


# In[59]:


df


# In[60]:


#Splitting the Data
X = df.drop(['Output'], 1)
y = df['Output']


# In[61]:


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.20,random_state=0)


# In[62]:


X_train.shape


# In[63]:


X_test.shape


# In[64]:


y_train.shape


# In[65]:


y_test.shape


# In[66]:


model = SVC(kernel='linear')
history=model.fit(X_train,y_train)


# In[67]:


y_pred=model.predict(X_test)


# In[68]:


print(classification_report(y_test,y_pred))


# In[90]:


train_sizes,train_scores, test_scores=learning_curve(RandomForestClassifier(),X_train, y_train, cv=5, scoring='accuracy',n_jobs=-1,train_sizes=np.linspace(0.01, 1, 50), verbose=1)


# In[91]:


train_mean= np.mean(train_scores, axis=1)
train_mean


# In[92]:


train_std= np.std(train_scores, axis=1)
train_std


# In[93]:


test_mean=np.mean(test_scores, axis=1)
test_mean


# In[94]:


test_std=np.std(test_scores, axis=1)
test_std


# In[95]:


plt.plot(train_sizes, train_mean,label='training scores')
plt.plot(train_sizes, test_mean, label='cross_validation score')

plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, color='#DDDDDD')

plt.fill_between(train_sizes, test_mean-test_std, test_mean+test_std, color='#DDDDDD')


plt.title('learning curve')
plt.xlabel('traning size')
plt.ylabel('accuracy score')
plt.legend(loc= 'best')


# In[78]:


cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[79]:


fpr,tpr,_=roc_curve(y_test,y_pred)
#calculate AUC
roc_auc=auc(fpr,tpr)
print('ROC AUC: %0.2f' % roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve(area= %2.f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# In[80]:


param_grid={'C':[0.1, 1, 10, 100],
            'gamma':[1,0.1,0.01,0.001],
            'kernel':['rbf']
           }

from sklearn.model_selection import GridSearchCV
optimal_params = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring='accuracy',
        verbose=0
)

optimal_params.fit(X_train, y_train)
print(optimal_params.best_params_)


# In[81]:


model=SVC(kernel='rbf', C=100, gamma=1) # C is 100 which means we will use regularization and the ideal value for gamma is 1
model.fit(X_train,y_train)


# In[82]:


print(f'accuracy - : {optimal_params.score(X_train,y_train):.3f}')


# In[ ]:




