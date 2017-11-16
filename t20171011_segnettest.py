# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:23:15 2017

@author: johnb
"""

from keras import models
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
import json
from keras.optimizers import SGD
from keras.models import load_model
img_w = 640
img_h = 512
n_labels = 17

kernel = 3
import numpy as np
import scipy.io as spio
import pandas as pd
mat = spio.loadmat('C:/Users/johnb/Google Drive/Classes/ECE590Local/multilassoMethod/allMasks.mat', squeeze_me=True)
badInds=pd.read_excel('C:/Users/johnb/Google Drive/Classes/ECE590Local/badInds200.xlsx')
badInds=np.asarray(badInds['Bad Inds'])-1
allMasks = mat['allMasks']
allMasks=allMasks[:,:,:]
mat = spio.loadmat('C:/Users/johnb/Google Drive/Classes/ECE590Local/multilassoMethod/allIms.mat', squeeze_me=True)
allIms = mat['allIms']
allIms=allIms[:,:,:]
trainIms=[]
valIms=[]
testIms=[]
trainMasks=[]
valMasks=[]
testMasks=[]
for i in range(0,200):
    if i not in badInds:
        trainIms.append(list(np.reshape(allIms[:,0:img_w,i],(512,img_w,1))))
        trainMasks.append(list(np.reshape(allMasks[:,0:img_w,i],(512,img_w,1))))
for i in range(0,200):
    if i in badInds:
        testIms.append(list(np.reshape(allIms[:,0:img_w,i],(512,img_w,1))))
        testMasks.append(list(np.reshape(allMasks[:,0:img_w,i],(512,img_w,1))))
#for i in range(150,175):
#    if i==150:
#        valIms=np.reshape(allIms[:,:,i],(512,512,1,1))
#        valMasks=np.reshape(allMasks[:,:,i],(512,512,1,1))
#    else:
#        valIms=np.concatenate((valIms,np.reshape(allIms[:,:,i],(512,512,1,1))),axis=3)
#        valMasks=np.concatenate((valMasks,np.reshape(allMasks[:,:,i],(512,512,1,1))),axis=3)
#for i in range(175,150):
#    if i==175:
#        testIms=np.reshape(allIms[:,:,i],(512,512,1))
#        testMasks=np.reshape(allMasks[:,:,i],(512,512,1))
#    else:
#        testIms=np.concatenate((testIms,np.reshape(allIms[:,:,i],(512,512,1,1))),axis=3)
#        testMasks=np.concatenate((testMasks,np.reshape(allMasks[:,:,i],(512,512,1,1))),axis=3)
valIms=trainIms[150:]
valMasks=trainMasks[150:]
trainIms=trainIms[0:150]
trainMasks=trainMasks[0:150]
trainIms=np.asarray(trainIms)
trainMasks=np.asarray(trainMasks)
valIms=np.asarray(valIms)
valMasks=np.asarray(valMasks)
testMasks=np.asarray(testMasks)
testIms=np.asarray(testIms)
def label_map(labels):
    label_map = np.zeros([img_h, img_w, n_labels])    
    for r in range(img_h):
        for c in range(img_w):
            label_map[r, c, labels[r][c]] = 1
    return label_map

encoding_layers = [
    Convolution2D(32, kernel, kernel, border_mode='same', input_shape=(img_h, img_w,1)),
    #BatchNormalization(),
    Activation('relu'),
    Convolution2D(32, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Convolution2D(64, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    Convolution2D(64, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

]

autoencoder = models.Sequential()
autoencoder.encoding_layers = encoding_layers

for l in autoencoder.encoding_layers:
    autoencoder.add(l)

decoding_layers = [
    UpSampling2D(),
    Convolution2D(32, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    Convolution2D(32, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    
    UpSampling2D(),
    Convolution2D(64, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    Convolution2D(64, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    UpSampling2D(),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    UpSampling2D(),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    UpSampling2D(),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    #BatchNormalization(),
    Activation('relu'),
    Convolution2D(n_labels, 1, 1, border_mode='same'),
    #BatchNormalization(),
]
autoencoder.decoding_layers = decoding_layers
for l in autoencoder.decoding_layers:
    autoencoder.add(l)

autoencoder.add(Reshape((img_h * img_w, n_labels)))
#autoencoder.add(Permute((2, 1)))
autoencoder.add(Activation('softmax'))

autoencoder.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
batchSize=4
trainAccs=[]
valAccs=[]
numEpochs=20
for k in range(0,numEpochs):
    for i in range(0,int(len(trainIms)/batchSize)-1):
        curTrain=trainIms[i*batchSize:(i+1)*batchSize,:,:,:]
        curTrainMask=[]
        notStarted=True
        for j in range(i*batchSize,(i+1)*batchSize):
            if notStarted:
                curTrainMask=np.reshape(label_map(trainMasks[j,:,:,:]),(1,512,img_w,17))
            else:
                curTrainMask=np.concatenate((curTrainMask,np.reshape(label_map(trainMasks[j,:,:,:]),(1,512,img_w,17))),axis=0)
            notStarted=False
        curTrainMask=np.reshape(curTrainMask,(batchSize,img_w*512,17))
        hist=autoencoder.train_on_batch(curTrain,curTrainMask)
    print(k)
    print("train acc")
    print(hist)
    trainAccs.append(hist[1])
    curValAccs=[]
    for i in range(0,int(len(valIms)/batchSize)-1):
        curVal=valIms[i*batchSize:(i+1)*batchSize,:,:,:]
        curValMask=[]
        notStarted=True
        for j in range(i*batchSize,(i+1)*batchSize):
            if notStarted:
                curValMask=np.reshape(label_map(valMasks[j,:,:,:]),(1,512,img_w,17))
            else:
                curValMask=np.concatenate((curValMask,np.reshape(label_map(valMasks[j,:,:,:]),(1,512,img_w,17))),axis=0)
            notStarted=False
        curValMask=np.reshape(curValMask,(batchSize,img_w*512,17))
        hist=autoencoder.evaluate(curVal,curValMask)
        curValAccs.append(hist[1])
    valAccs.append(np.average(curValAccs))
    if np.average(curValAccs)==np.max(valAccs):
        print("saved model")
        autoencoder.save('C:/Users/johnb/Google Drive/Classes/ECE590Local/modelScaled2.h5')
    print("val acc")
    print(np.average(curValAccs))
autoencoder=load_model('C:/Users/johnb/Google Drive/Classes/ECE590Local/modelScaled2.h5')
def convertPredictionToMap(myPred):
    a=np.zeros((np.shape(myPred)[0:2]))
    for i in range(0,np.shape(myPred)[0]):
        for j in range(0,np.shape(myPred)[1]):
            a[i][j]=np.argmax(myPred[i,j,:])
    return a
#from PIL import Image
#img = Image.fromarray(convertPredictionToMap(mypred[1,:,:,:]))
#img.show()
testInd=1
valInd=1
from matplotlib import pyplot as plt
f, ax = plt.subplots(2)
ax[0].imshow(testIms[testInd,:,:,0], origin="lower")
ax[1].imshow(testMasks[testInd,:,:,0], origin="lower")


f, ax = plt.subplots(2)
ax[0].imshow(valIms[valInd,:,:,0], origin="lower")
ax[1].imshow(valMasks[valInd,:,:,0], origin="lower")

#Look at predictions for the same images

myoutTest=autoencoder.predict(np.reshape(testIms[testInd,:,:,:],(1,512,640,1)))
myoutTest=np.reshape(myoutTest,(512,640,17))
myoutTest=convertPredictionToMap(myoutTest)
myoutVal=autoencoder.predict(np.reshape(valIms[valInd,:,:,:],(1,512,640,1)))
myoutVal=np.reshape(myoutVal,(512,640,17))
myoutVal=convertPredictionToMap(myoutVal)
f, ax = plt.subplots(2)
ax[0].imshow(myoutTest, origin="lower")
ax[1].imshow(myoutVal, origin="lower")


for i in range(0,len(testIms)):
    myoutTest=autoencoder.predict(np.reshape(testIms[i,:,:,:],(1,512,640,1)))
    myoutTest=np.reshape(myoutTest,(512,640,17))
    myoutTest=convertPredictionToMap(myoutTest)
    f, ax = plt.subplots(3)
    ax[0].imshow(testIms[i,:,:,0], origin="lower")
    ax[1].imshow(testMasks[i,:,:,0], origin="lower")
    ax[2].imshow(myoutTest, origin="lower")
    saveStr='C:/Users/johnb/Google Drive/Classes/ECE590Local/testSegs/testIm'+str(i)+'.png'
    plt.savefig(saveStr)
import os
from PIL import Image

def rotateImages(rotationAmt):
  # for each image in the current directory
  for image in os.listdir('C:/Users/johnb/Google Drive/Classes/ECE590Local/testSegs'):
    # open the image
    img = Image.open('C:/Users/johnb/Google Drive/Classes/ECE590Local/testSegs/'+image)
    # rotate and save the image with the same filename
    img.rotate(rotationAmt).save('C:/Users/johnb/Google Drive/Classes/ECE590Local/testSegs/'+image)
    # close the image
    img.close()
rotateImages(90)