#!/usr/bin/env python
# coding: utf-8

# In[94]:


#porting Important Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,accuracy_score,auc,hinge_loss,roc_auc_score
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D,MaxPooling2D, Flatten


# In[95]:


df= pd.read_csv('Diabetes.csv')
df.dtypes


# In[96]:


title_mapping = {'YES':1,'NO':0}
df[' Class variable']=df[' Class variable'].map(title_mapping)


# In[97]:


col=['n_pregnant','glucose_conc','bp','skin_len','insulin','bmi','pedigree_fun','age','Output']
df.columns=col


# In[98]:


diabetes_true_count = len(df.loc[df['Output'] == True])
diabetes_false_count = len(df.loc[df['Output'] == False])


# In[99]:


col=['glucose_conc','bp','insulin','bmi','skin_len']
for i in col:
    df[i].replace(0, np.nan, inplace= True)


# In[100]:


def median_target(var):
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Output']].groupby(['Output'])[[var]].median().reset_index()
    return temp


# In[101]:


median_target('insulin')
median_target('glucose_conc')
median_target('skin_len')
median_target('bp')
median_target('bmi')


# In[102]:


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


# In[103]:


median_target('n_pregnant')
df.loc[(df['Output'] == 0 ) & (df['n_pregnant']>13), 'n_pregnant'] = 2
df.loc[(df['Output'] == 1 ) & (df['n_pregnant']>13), 'n_pregnant'] = 4
median_target('bp')
df.loc[(df['Output'] == 0 ) & (df['bp']<40), 'bp'] = 70
df.loc[(df['Output'] == 1 ) & (df['bp']<40), 'bp'] = 74.5
df.loc[(df['Output'] == 0 ) & (df['bp']>103), 'bp'] = 70
df.loc[(df['Output'] == 1 ) & (df['bp']>103), 'bp'] = 74.5


# In[104]:


median_target('skin_len')
df.loc[(df['Output'] == 0 ) & (df['skin_len']>38), 'skin_len'] = 27
df.loc[(df['Output'] == 1 ) & (df['skin_len']>38), 'skin_len'] = 32
df.loc[(df['Output'] == 0 ) & (df['skin_len']<20), 'skin_len'] = 27
df.loc[(df['Output'] == 1 ) & (df['skin_len']<20), 'skin_len'] = 32
median_target('bmi')
df.loc[(df['Output'] == 0 ) & (df['bmi']>48), 'bmi'] = 30.1
df.loc[(df['Output'] == 1 ) & (df['bmi']>48), 'bmi'] = 34.3


# In[105]:


median_target('pedigree_fun')
df.loc[(df['Output'] == 0 ) & (df['pedigree_fun']>1), 'pedigree_fun'] = 0.336
df.loc[(df['Output'] == 1 ) & (df['pedigree_fun']>1), 'pedigree_fun'] = 0.449


# In[106]:


median_target('age')
df.loc[(df['Output'] == 0 ) & (df['age']>61), 'age'] = 27
df.loc[(df['Output'] == 1 ) & (df['age']>61), 'age'] = 36


# In[107]:


df.dtypes


# In[108]:


df.head()


# In[109]:


df['Output'].unique()


# In[110]:


#data visulation
plt.plot(df['bp'],'o')
plt.plot(df['bmi'],'o')
plt.title('bp vs bmi')
plt.legend(['bp','bmi'], loc='upper left')
plt.show()
#BLUE FOR BP
#ORANGE FOR BMI


# In[111]:


scaler = MinMaxScaler()
df.col=['n_pregnant','glucose_conc','bp','skin_len','insulin','bmi','pedigree_fun','age','Output']
df[df.col]= scaler.fit_transform(df[df.col])


# In[112]:


df


# In[113]:


#Splitting the Data
X = df.drop(['Output'], 1)
y = df['Output']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[114]:


X_train.shape


# In[115]:


X_test.shape


# In[134]:


model = keras.Sequential([
    #keras.layers.Dense(7, input_shape=(8,), activation='relu'),
    keras.layers.Dense(6, input_shape=(8,), activation='relu'),
    keras.layers.Dense(4, input_shape=(8,), activation='relu'),
    keras.layers.Dense(2, input_shape=(8,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),   
])

model.compile (optimizer = "adam",
               loss="binary_crossentropy",
               metrics=['accuracy'])

history=model.fit(X_train, y_train, epochs=100, validation_split=0.1)


# model.ptim

# In[135]:


yp=model.predict(X_test)
yp[:15]


# In[136]:


y_test[:10]


# In[137]:


y_pred = []
for element in yp:
    if element > .5:
        y_pred.append(1)
    else:
        y_pred.append(0)


# In[138]:


y_pred[:10]


# In[139]:


print(classification_report(y_test,y_pred))


# In[140]:


cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[146]:


round((96+39)/(96+39+8+11),2)


# In[147]:


plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training data'],loc='upper left')
plt.show()


# In[148]:


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training data'],loc='upper left')
plt.show()


# In[149]:


accuracy_score(y_test, y_pred)


# In[150]:


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


# In[ ]:





# In[ ]:




