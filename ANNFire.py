# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 20:23:16 2021

@author: youssif
"""
# Importing Libraries
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten ,GaussianNoise
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn import model_selection
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import seaborn
import pathlib
import glob
# %matplotlib inline
tf.__version__


 #Loading the data 
print(os.listdir("Desktop/D folder/college/selected/project/archive1/Data"))
fire_path = "Desktop/D folder/college/selected/project/archive1/Data/Train_Data/Fire"
non_fire_path = "Desktop/D folder/college/selected/project/archive1/Data/Train_Data/Non_Fire"
fire_path = pathlib.Path(fire_path)
non_fire_path = pathlib.Path(non_fire_path)




"""
 Preprocessing
"""
train_data_images = {
    "Fire" : list(fire_path.glob("*.jpg")),
    "NonFire" : list(non_fire_path.glob("*.jpg"))
}
train_labels = {
    "Fire": 0, "NonFire": 1
}
print(train_data_images)


"""# Reading images and storing it in an array"""
#data preprocessing&augmentation

X, y = [], []   # is the data and y is the labels
for label, images in train_data_images.items():
  for image in images:
    img = cv2.imread(str(image))   # Reading the image
    print(img)
    if img is not None:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = cv2.resize(img, (50, 50))
      X.append(img)
      y.append(train_labels[label])
print(X)

import numpy
data = numpy.array(X) 
labels = numpy.array(y)

np.save('Data', data)
np.save('Labels', labels)

print('forests : {} | labels : {}'.format(data.shape , labels.shape))

data = np.load("Data.npy")
labels = np.load("Labels.npy")

#ploting images and labels to understand the data
plt.figure(1 , figsize = (15 , 9)) 
n = 0 
for i in range(49):
    n += 1 
    r = np.random.randint(0 , data.shape[0] , 1)
    plt.subplot(7 , 7 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    plt.imshow(data[r[0]])
    plt.title('{} : {}'.format('Non Fire' if labels[r[0]] == 1 else 'Fire' ,
                               labels[r[0]]) )
    plt.xticks([]) , plt.yticks([])
    
plt.show()

# normalizing the images
data = data.astype(np.float32)
labels = labels.astype(np.int32)
data = (data/255) 
data[0].shape

# split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state = 42)

print(X_train.shape)

"""# Data Argumentation"""

data_argumentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomContrast(0.3),
    keras.layers.experimental.preprocessing.RandomRotation(0.2),
    keras.layers.experimental.preprocessing.RandomZoom(0.5) 
])


# compare between fire and nun fire images
plt.figure(1, figsize = (15, 7))
plt.imshow(data[0])
plt.title('fire')


plt.figure(1, figsize = (15, 7))
plt.imshow(data[4000])
plt.title('non fire')


"""# Model Building"""
ann = tf.keras.models.Sequential()

ann.add(keras.layers.Flatten(input_shape=(50, 50, 3)))
# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
checkpoint = keras.callbacks.ModelCheckpoint(filepath="weights.h5", verbose=1, save_best_only=True)
history = ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

ann.evaluate(X_test, y_test)

"""# Testing Dataset"""

test_fire_path = "Desktop/D folder/college/selected/project/Data/Test_Data/Fire"
test_non_fire_path = "Desktop/D folder/college/selected/project/Data/Test_Data/Non_Fire"
test_fire_path = pathlib.Path(test_fire_path)
test_non_fire_path = pathlib.Path(test_non_fire_path)
print(test_fire_path)

test = {
    "Fire": list(test_fire_path.glob("*.jpg")),
    "NonFire":list(test_fire_path.glob("*.jpg"))
}
print (test['Fire'])
test_array = []
for label, images in test.items():
    for image in images:
        img = cv2.imread(str(image)) # Reading the test image
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (50, 50))
            test_array.append(img)
test_array = numpy.array(test_array)

test_array = test_array/ 255
pred = ann.predict(test_array)
preds = pred.round(decimals=0).flatten()
results = []
for i in preds:
    if i==0:
        results.append("Fire")
    elif i == 1:
        results.append("Non Fire")

fire = 0
nonFire = 0
for i in results:
    if i=="Fire":
        fire += 1
    else:
        nonFire += 1
print(fire, "Fire images out of ", 25)
print(nonFire , 'Non fire images out of', 25)

font1 = {'family':'serif','color':'blue','size':20} # Custom font for my title
for i in range(50):
    plt.imshow(test_array[i])
    plt.title(results[i], fontdict=font1)
    plt.axis('off')
    plt.show()
    
    
#Evaluate the model

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = ann.evaluate(X_test,  y_test, verbose=2)

# plot loss per iteration

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
y_pred = ann.predict_classes(X_test)
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('predicted')
plt.ylabel('Truth')
print(cm)


# ROC curve

from sklearn.metrics import roc_curve, auc
import scikitplot as skplt
fpr, tpr,_ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print('Roc AUC: %0.2f' % roc_auc)
plt.figure(figsize=(5,5), dpi=100)
plt.plot(fpr, tpr, linestyle='-', label='ANN (auc = %0.3f)' %roc_auc)
#plt.plot(fpr,tpr,labels='ROC curve(area= %2.f)' %roc_auc)
plt.plot([0,1],[0,1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


