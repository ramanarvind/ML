# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:26:37 2019

@author: araman
"""



import numpy as np
import matplotlib.pyplot as plt
import cv2 as opencv
import digitRecognizer as DR

"""
Train the digit recognizer model
"""
# Read the digit images
fpTrain = open("./train/train-images.idx3-ubyte", "rb")
magicNum = int.from_bytes(fpTrain.read(4), byteorder='big')
assert(magicNum == 2051)
numTrainImgs = int.from_bytes(fpTrain.read(4), byteorder='big')
numTrainRows = int.from_bytes(fpTrain.read(4), byteorder='big')
numTrainCols = int.from_bytes(fpTrain.read(4), byteorder='big')
trainVecs = np.frombuffer(fpTrain.read(numTrainImgs*numTrainRows*numTrainCols), dtype='ubyte')
trainVecs = opencv.adaptiveThreshold(trainVecs, 255, opencv.ADAPTIVE_THRESH_MEAN_C, opencv.THRESH_BINARY_INV, 9, 2);
trainVecs = np.array(trainVecs).astype(np.float32)
trainVecs = trainVecs.reshape(numTrainImgs, numTrainRows*numTrainCols)
# Read the digit labels
fpTrainLabel = open("./train/train-labels.idx1-ubyte", "rb")
magicNum = int.from_bytes(fpTrainLabel.read(4), byteorder='big')
assert(magicNum == 2049)
numTrainLabels = int.from_bytes(fpTrainLabel.read(4), byteorder='big')
trainLabels = np.frombuffer(fpTrainLabel.read(numTrainLabels), dtype='ubyte')
trainLabels = np.array(trainLabels).astype(np.float32)
trainLabels = trainLabels.reshape(numTrainLabels, 1)
# Create an instance of the digit recognizer class and train it
dr = DR.digitRecognizer()
dr.train(trainVecs, trainLabels)

# Test the model
# Read the header information from test image
fpTest = open("./train/t10k-images.idx3-ubyte", "rb")
magicNum = int.from_bytes(fpTest.read(4), byteorder='big')
assert(magicNum == 2051)
numTestImgs = int.from_bytes(fpTest.read(4), byteorder='big')
numTestRows = int.from_bytes(fpTest.read(4), byteorder='big')
numTestCols = int.from_bytes(fpTest.read(4), byteorder='big')

# Read the header information test labels
fpTestLabel = open("./train/t10k-labels.idx1-ubyte", "rb")
magicNum = int.from_bytes(fpTestLabel.read(4), byteorder='big')
assert(magicNum == 2049)
numTestLabels = int.from_bytes(fpTestLabel.read(4), byteorder='big')

numTestImgs = 1
numTestLabels = 1

print("Reading test images...")
testVecs = np.frombuffer(fpTest.read(numTestImgs*numTestRows*numTestCols), dtype='ubyte')
testVecs = opencv.adaptiveThreshold(testVecs, 255, opencv.ADAPTIVE_THRESH_MEAN_C, opencv.THRESH_BINARY_INV, 9, 2);

imgData = testVecs.reshape([numTestRows, numTestCols])
plt.imshow(imgData, cmap='gray', interpolation='bicubic')    
plt.show()

testVecs = np.array(testVecs).astype(np.float32)
testVecs = testVecs.reshape(numTestImgs, numTestRows*numTestCols)

print("Reading test labels...")
testLabels = np.frombuffer(fpTestLabel.read(numTestLabels), dtype='ubyte')
testLabels = np.array(testLabels).astype(np.float32)
testLabels = testLabels.reshape(numTestLabels, 1)

print("Finding the nearest neighbours...")
ret, result, neighbours, dist = dr.test(testVecs)

matches = result ==  testLabels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size
print("Accuracy = ", accuracy)

 